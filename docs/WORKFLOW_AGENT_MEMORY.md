# DrivoR 工作流备忘（Agent Memory）

本文档汇总本仓库上 **train metric cache、BEV phase-1 训练、常见故障** 的上下文，供后续会话与人类查阅。路径以工作区 `wenzhet` 为根时的布局为准。

## 路径与数据

| 用途 | 典型路径 |
|------|----------|
| DrivoR 根目录 | `DrivoR/` |
| 训练/实验输出 | `DrivoR/exp/`（`NAVSIM_EXP_ROOT`） |
| OpenScene / NavSIM 数据 | `navsim_dataset/`（`OPENSCENE_DATA_ROOT`） |
| BEV 导出 | `navsim_bev_feature/exports_pretrained/{trainval|test}/**/*_decoder_neck.pt` |
| Train 用 metric cache | `DrivoR/exp/train_metric_cache/`（`*/*/metric_cache.pkl`） |
| BEV token 列表（训练 scene filter） | `DrivoR/exp/bev_feature_tokens/trainval_decoder_neck_tokens_full.txt` |

## Train metric caching

- **脚本**：`DrivoR/metric_caching.sh`（入口为 `navsim/planning/script/run_train_metric_caching.py`）。
- **续跑**：已有 `metric_cache.pkl` 时在 `train_cache_processor.py` 中会跳过（除非 force 重算）；中断后可原参数重跑。
- **后台与日志**：脚本默认 `BACKGROUND=1`，用 `nohup` 写 `exp/metric_caching_${TRAIN_TEST_SPLIT}.log`；可调 `BACKGROUND=0` 前台调试；`LOG_TO_DRIVER` 控制 Ray worker 日志是否打到终端。
- **全量对齐**：scene filter token 列表约 **152495**；一次完整跑成功后日志可出现约 **151778** features cached（剩余差集常为数据/filter 不可用场景，非“没跑完”）。

## BEV Phase-1 训练

- **主脚本**：`DrivoR/scripts/training/run_drivor_bev_phase1.sh`。
- **用法**：`bash scripts/training/run_drivor_bev_phase1.sh <baseline.pth> [experiment_name] [max_epochs]`。
- **Token 文件 env**：`BEV_TOKEN_FILTER_FILE`；若为文档占位 `/PATH/TO/...` 会误导 `mkdir`，脚本已对占位路径与不可写父目录做了回退到 `exp/bev_feature_tokens/..._tokens_full.txt`。
- **Hydra**：`scene_filter_token_file` 即每行一个 **scene token**，限制数据集与预计算 BEV 对齐。
- **默认策略**：`TRAINER_STRATEGY=ddp_find_unused_parameters_true`（BEV scorer 冻结主网后，仍有参数可能未参与某步 loss，纯 `ddp` 会报 unused parameters）。
- **W&B**：`USE_WANDB=1`；`WANDB_MODE=offline` 可离线；DDP 下只应在 rank 0 初始化 W&B logger。
- **多卡稳定性 env（guppy）**：`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`。裸 `torch.distributed` 两 rank NCCL probe 在默认 NCCL transport 下会卡/timeout，设置这两个变量后 `all_reduce`/`barrier` 正常。
- **Python env**：脚本里的 `PYTHON_BIN` 可被 shell 环境变量覆盖。若输出中 Python 不是预期 conda env，先 `unset PYTHON_BIN` 或显式 `PYTHON_BIN=/path/to/env/bin/python`。

## 自适应 GPU wrapper（可选）

- **文件**：若存在则 `DrivoR/train_drivor_bev_phase1.sh`：自动探测 `CUDA_VISIBLE_DEVICES`/`nvidia-smi` 的 GPU 数，并按 GPU 缩放 `NUM_WORKERS`（仍调用内层 `run_drivor_bev_phase1.sh`）。

## 已遇问题与原因

### 1. DataLoader：`stack expects each tensor to be equal size`

- **现象**：`[256,128,128]` vs `[1,256,128,128]`。
- **原因**：同批 BEV `.pt` 混存 **`C,H,W`** 与 **`1,C,H,W`** 两种形状（导出或读取分支不一致）。
- **方向**：在加载 BEV 处统一 `squeeze(0)` 或统一维度约定；或对导出脚本做一致性检查。

### 2. 双卡 NCCL / DDP freeze：`Broadcast` / `ALLREDUCE` watchdog 超时

- **现象**：单卡正常；2 GPU 时 rank 卡在 Lightning DDP setup、`broadcast`、`barrier` 或 `ALLREDUCE`，GPU util 可接近 100%，显存很小；最终出现 NCCL watchdog timeout。
- **关键定位**：裸 NCCL probe（无 DrivoR / Lightning / Ray / W&B / DataLoader）在 guppy 默认 NCCL transport 下也会失败；设置 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` 后 probe 通过。因此 guppy 上首要根因是 **NCCL transport / peer communication**，不是 W&B 本身。
- **脚本修复**：`scripts/training/run_drivor_bev_phase1.sh` 默认导出 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`、`NCCL_ASYNC_ERROR_HANDLING=1`、`TORCH_DISTRIBUTED_DEBUG=DETAIL`。
- **Lightning 修复**：`run_training_full.py` 使用自定义 `LocalMetadataDDPStrategy`，避免 Lightning 对本地相同 metadata 走 NCCL object broadcast（此前 stack dump 可见 `trainer.log_dir` / `DDPStrategy.broadcast_object_list` / pre-setup barrier）。
- **Ray 修复**：DDP 训练时 `agent.config.use_ray_score=false`，`DrivoRAgent` 在 `num_gpus > 1` 时不启用 Ray scorer；Ray worker 单机分支同时设置 `include_dashboard=False` 和 per-rank `RAY_TMPDIR`，避免每个 DDP rank 起本地 Ray 互相影响。

### 3. DDP + W&B online/offline：rank-safe logger 问题

- **现象**：单卡 + W&B 正常；多卡 + W&B online/offline 可卡住、crash，或在 epoch end 因 logger metadata 异常退出。
- **原因 A**：Hydra 传入的 W&B logger config 是带 `_target_` 的 DictConfig，不能原样交给 Lightning；且 DDP 每个 rank 都初始化 W&B 会造成额外服务进程/网络同步/metadata broadcast 风险。
- **修复 A**：`run_training_full.py` 将 `cfg.trainer.params` 转成普通 dict；若 logger config 带 `_target_`，只在 `LOCAL_RANK/RANK == 0` instantiate `WandbLogger`，非 0 rank 设 `logger=False`。
- **原因 B**：禁用 logger 的 rank 仍带 `LearningRateMonitor`，Lightning 会报 `Cannot use LearningRateMonitor callback with Trainer that has no logger`。
- **修复 B**：当 `trainer_params["logger"] is False` 时，从 callbacks 里过滤 `LearningRateMonitor`。
- **原因 C**：W&B 会往 progress metrics 里加字符串 `v_num`，旧进度打印器对所有 metric 用 `{v:.3f}`，导致 `ValueError: Unknown format code 'f' for object of type 'str'`，rank 1 随后表现为 barrier/peer error。
- **修复 C**：`LitProgressBar` 的 metric formatter 现在支持 tensor scalar、number 和 string。
- **验证**：2 GPU tiny smoke（`SCENE_FILTER_MAX_SCENES=64 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=0 NUM_WORKERS=0 BATCH_SIZE=1`）已验证：`USE_WANDB=0`、`WANDB_MODE=offline`、`WANDB_MODE=online` 都能过 DDP setup 并完成 1 个 train batch；online run 示例：`htxsxhuy`。

### 4. DataLoader worker 被 `Killed`（与 NCCL/W&B freeze 区分）

- **现象**：已通过 DDP setup 和 W&B init，但在第一批数据前/取 batch 时退出；典型 traceback：`DataLoader worker (pid ...) is killed by signal: Killed` 或日志无 traceback、无 checkpoint、W&B 只有 init。GPU 事后空闲，不是 CUDA OOM。
- **原因**：DDP 下 `dataloader.params.batch_size` 通常是 **每 rank/per GPU batch size**。例如 `BATCH_SIZE=96 NUM_WORKERS=8 NUM_GPUS=2` 等价于 global batch 约 192，且共有 16 个 DataLoader workers。BEV feature 单样本约 `256*128*128*float32 ~= 16 MB`，大 batch + prefetch + workers 很容易造成 CPU RAM / shared memory / worker 进程压力，被系统杀掉。
- **排查**：看 stdout 是否出现 `DataLoader worker ... killed by signal: Killed`；看 `nvidia-smi` 是否没有训练进程；看 `exp/ke/<exp>/<uid>/*.log` 是否停在 model summary 后、没有 batch/epoch 输出。普通用户可能看不到 kernel OOM 日志（`journalctl -k` 权限不足）。
- **缓解建议**：先用 `BATCH_SIZE=8 NUM_WORKERS=2 PREFETCH_FACTOR=1` 跑通，再逐步增大；调试 worker crash 时用 `NUM_WORKERS=0` 获得更直接的 Python traceback；避免一开始用 `BATCH_SIZE=32/96` + `NUM_WORKERS=8`。

### 5. 环境占位符 `BEV_TOKEN_FILTER_FILE=/PATH/TO/...`

- 已通过脚本内占位检测与 `mkdir` 失败回退避免误建 `/PATH`。

### 6. ModelCheckpoint `val/score_epoch` not found

- **现象**：Epoch 0 训练和验证都跑完了，但在保存 checkpoint 时 `ModelCheckpoint(monitor='val/score_epoch')` 报错找不到该 key。
- **原因**：`val/score` 被 log 为 `on_step=False, on_epoch=True`，PL 在此情况下不加 `_epoch` 后缀——key 就是 `val/score`。只有 `on_step=True, on_epoch=True` 同时为真时，PL 才会生成 `_step` 和 `_epoch` 双 key。
- **修复**：把 `val/score` 改为 `on_step=True, on_epoch=True`，这样 PL 自动产生 `val/score_epoch`，`ModelCheckpoint` 可正常 monitor。

### 7. WandB 只看到 `train/loss_step`，其他 train loss 不显示

- **现象**：WandB dashboard 里只有 `train/loss_step` 曲线，`train/trajectory_loss`、`train/score` 等不见。
- **原因**：其他 loss 被设为 `on_step=False, on_epoch=True`，整个 epoch 只发一个数据点给 WandB；若 epoch 没跑完或只有 1 个数据点，WandB 默认不自动建图。
- **修复**：把其他 loss 改为 `on_step=True, on_epoch=False`——每步都记录（名字无后缀、WandB 自动建图），但不需要 epoch 聚合。

### 8. Cache 与训练配置不匹配：`KeyError: 'trajectory_long'`

- **现象**：用旧 cache 跑训练时，DataLoader collate 报 `KeyError: 'trajectory_long'`。
- **原因**：cache 生成时未带 `agent.config.long_trajectory_additional_poses=2`（或其他改变 sample schema 的参数），导致 cache 中部分/全部样本缺少 `trajectory_long` 字段。训练脚本期望所有样本有此字段。
- **修复**：用与训练完全相同的 agent config 重新生成 cache（`scripts/training/run_dataset_caching.sh`），存到独立路径 `exp/navsim_cache_nommcv_same_as_training`。
- **关键参数必须匹配**：`long_trajectory_additional_poses`、`use_bev_feature`、`bev_feature_type`、`bev_channels`、`bev_features_root`、`bev_data_split`。

## PL Metric Naming 规则速查

| `on_step` | `on_epoch` | WandB 里的 key | 说明 |
|-----------|------------|----------------|------|
| True | True | `name_step` + `name_epoch` | 两份 |
| True | False | `name` (无后缀) | 逐步记录 |
| False | True | `name` (无后缀) | 仅 epoch 末 |

当前 agent_lightning_module.py 配置：
- `train/loss` → `on_step=True, on_epoch=True` → `train/loss_step` + `train/loss_epoch`
- 其他 train losses → `on_step=True, on_epoch=False` → 原名（如 `train/trajectory_loss`）
- `val/score` → `on_step=True, on_epoch=True` → `val/score_step` + `val/score_epoch`（用于 ModelCheckpoint monitor）
- 其他 val metrics → `on_step=False, on_epoch=True` → 原名（如 `val/l2`）

## Dataset Caching

- **脚本**：`scripts/training/run_dataset_caching.sh` → `navsim/planning/script/run_dataset_caching.py`
- **关键要求**：cache 必须用与训练**完全相同**的 agent config 生成，否则 sample schema 不一致会导致 DataLoader crash。
- **改变 schema 的关键参数**：`long_trajectory_additional_poses`、`use_bev_feature`、`bev_feature_type`、`bev_channels`、`bev_features_root`、`bev_data_split`
- **推荐 worker**：`worker=sequential`（Ray 在 caching 时容易 crash）
- **性能对比**：无 cache 初始化 ~33hr（SceneLoader over NFS）；有 cache 加载 ~3s（85k samples）
- **路径**：
  - 旧泛用 cache（schema 不一定匹配）：`exp/navsim_cache_nommcv_full`
  - 训练匹配 cache：`exp/navsim_cache_nommcv_same_as_training`

## 建议命令速查

```bash
# Train metric cache（全量 trainval token 列表示例）
cd DrivoR
MAX_RETRIES=200 SCENE_FILTER_TOKEN_FILE="$PWD/exp/bev_feature_tokens/trainval_decoder_neck_tokens_full.txt" bash metric_caching.sh

# Dataset cache（必须与训练 agent config 一致！）
bash scripts/training/run_dataset_caching.sh

# BEV phase-1 训练（示例）
unset BEV_TOKEN_FILTER_FILE   # 若 shell 曾 export 占位路径
unset PYTHON_BIN              # 若 shell 曾 export 到非预期 conda env
bash scripts/training/run_drivor_bev_phase1.sh \
  ./weights/checkpoints/drivor_Nav1_25epochs.pth finetune_drivor_bev_full_trainval 20

# 使用 cache 的全量训练（当前推荐配置）
CACHE_PATH="$PWD/exp/navsim_cache_nommcv_same_as_training" \
USE_CACHE_WITHOUT_DATASET=true \
BATCH_SIZE=16 NUM_WORKERS=8 PREFETCH_FACTOR=1 NUM_GPUS=2 \
bash scripts/training/run_drivor_bev_phase1.sh \
  ./weights/checkpoints/drivor_Nav1_25epochs.pth guppy-2gpu-16batch-8worker-cache 30

# 多卡 + W&B smoke test（先确认 DDP/W&B 通路）
USE_WANDB=1 WANDB_MODE=online \
SCENE_FILTER_MAX_SCENES=64 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=0 \
BATCH_SIZE=1 NUM_WORKERS=0 \
bash scripts/training/run_drivor_bev_phase1.sh \
  ./weights/checkpoints/drivor_Nav1_25epochs.pth debug_wandb_online_ddp 1

# 全量训练更稳的起点（之后再逐步增大 batch/workers）
BATCH_SIZE=8 NUM_WORKERS=2 PREFETCH_FACTOR=1 \
bash scripts/training/run_drivor_bev_phase1.sh \
  ./weights/checkpoints/drivor_Nav1_25epochs.pth finetune_drivor_bev_full_trainval 30
```

## 维护说明

- 数值（152495 / 151778、脚本行号）若与当前磁盘不一致，以实际 `wc -l`、`find`、`日志` 为准。
- 若升级 PyTorch Lightning / Ray / W&B / PyTorch NCCL，需重新核对：`strategy` 字符串、rank 0 logger instantiate、`LearningRateMonitor` 过滤、`ray.init()` 参数、`RAY_TMPDIR`、以及 `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` 是否仍需要。
- 更换 agent config 任何改变 sample structure 的参数后，必须重新生成 dataset cache。
