# DrivoR 环境搭建与 NAVSIM PDMS 评估指南

> 文档日期: 2026-04-01
> 基于 DrivoR 仓库 (valeoai/DrivoR) 在 NAVSIM-v1 navtest 分割上的 PDMS 评估流程。

---

## 1. 前提条件

本指南假设以下资源已准备就绪（由 WoTE 仓库提供）：

| 资源 | 路径 | 说明 |
|------|------|------|
| NAVSIM 数据集 (logs) | `/home/wenzhe/wm_ws/WoTE/dataset/navsim_logs/{test,trainval}` | navtest 分割使用 `test` 目录 (147 个 pkl, ~983MB) |
| 传感器数据 (sensor blobs) | `/home/wenzhe/wm_ws/WoTE/dataset/sensor_blobs/{test,trainval}` | 相机图像等原始传感器数据 |
| nuPlan 地图 | `/home/wenzhe/wm_ws/WoTE/dataset/maps/` | 4 个城市地图 + `nuplan-maps-v1.0.json` |
| Metric Cache | `/home/wenzhe/wm_ws/WoTE/exp/metric_cache/` | 已预计算，69,711 条目，lzma 压缩 pkl 格式 |

---

## 2. 环境搭建

### 2.1 创建 Conda 环境

```bash
conda create -n drivoR python=3.8 -y
```

### 2.2 安装 PyTorch (CUDA 12.1)

```bash
conda run -n drivoR pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 安装 nuplan-devkit 和 DrivoR

```bash
# nuplan-devkit (DrivoR 仓库内置的版本)
conda run -n drivoR pip install -e /home/wenzhe/wm_ws/DrivoR/nuplan-devkit

# DrivoR navsim 包 (版本 1.1.0)
conda run -n drivoR pip install -e /home/wenzhe/wm_ws/DrivoR
```

### 2.4 修复缺失依赖

初始安装后以下包可能缺失，需手动安装：

```bash
conda run -n drivoR pip install \
    numpy==1.24.4 \
    scipy==1.10.1 \
    sympy==1.13.3 \
    absl-py \
    pytorch-lightning==2.2.1 \
    selenium
```

> **重要**: 安装 `absl-py` 时 pip 可能会升级 `typing-extensions`，导致 numpy 被卸载。
> 务必在安装 absl-py 后验证 numpy 是否仍在。

### 2.5 验证安装

```bash
/home/wenzhe/miniconda/envs/drivoR/bin/python -c "
import numpy; print('numpy', numpy.__version__)
import torch; print('torch', torch.__version__)
import pytorch_lightning; print('pl', pytorch_lightning.__version__)
import timm; print('timm', timm.__version__)
import hydra; print('hydra', hydra.__version__)
from navsim.agents.drivoR.drivor_agent import DrivoRAgent; print('DrivoRAgent OK')
print('All imports successful!')
"
```

### 2.6 已验证的包版本清单

| 包 | 版本 |
|---|---|
| torch | 2.1.0+cu121 |
| torchvision | 0.16.0+cu121 |
| pytorch-lightning | 2.2.1 |
| timm | 1.0.15 |
| hydra-core | 1.3.2 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| navsim | 1.1.0 (editable, from DrivoR) |
| nuplan-devkit | 1.2.2 (editable) |
| absl-py | 2.3.1 |

---

## 3. 下载权重

### 3.1 DINOv2 Backbone 权重

```bash
mkdir -p weights/vit_small_patch14_reg4_dinov2.lvd142m
cd weights/vit_small_patch14_reg4_dinov2.lvd142m

pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='timm/vit_small_patch14_reg4_dinov2.lvd142m',
    local_dir='.', local_dir_use_symlinks=False
)
"
```

下载后目录包含: `config.json`, `model.safetensors`, `pytorch_model.bin`, `README.md`

### 3.2 DrivoR 模型权重 (GitHub Releases)

| 权重文件 | NAVSIM 版本 | 大小 | 下载地址 |
|----------|-------------|------|----------|
| `drivor_Nav1_25epochs.pth` | Nav1 (PDMS) | 292MB | `model_weights` release |
| `drivor_Nav2_10epochs.pth` | Nav2 (EPDMS) | 292MB | `model_weights` release |
| `drivor_Nav1_train_only_85k_25epochs.ckpt` | Nav1 (train-only) | 292MB | `model_weights` release |
| `nav1_30epochs_with_68k_simscale_103ktrainval.pth` | Nav1 + SimScale 68k | 292MB | `Scaling` release |
| `nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth` | Nav1 + SimScale 134k | 292MB | `Scaling` release |

下载 Nav1 基础权重:
```bash
cd /home/wenzhe/wm_ws/DrivoR/weights
wget "https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav1_25epochs.pth"
```

---

## 4. 环境变量配置

```bash
# 数据集路径 (复用 WoTE 的数据)
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/wenzhe/wm_ws/WoTE/dataset/maps"
export OPENSCENE_DATA_ROOT="/home/wenzhe/wm_ws/WoTE/dataset"

# DrivoR 代码和实验路径
export NAVSIM_DEVKIT_ROOT="/home/wenzhe/wm_ws/DrivoR"
export NAVSIM_EXP_ROOT="/home/wenzhe/wm_ws/DrivoR/exp"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
```

> **关键**: `NAVSIM_DEVKIT_ROOT` 必须指向 DrivoR 仓库（不是 WoTE），这样 Hydra 才能找到 DrivoR 的 agent 配置。

---

## 5. Metric Cache

### 5.1 复用 WoTE 的 Metric Cache

WoTE 已预计算了 metric cache，格式与 DrivoR 完全兼容：

```bash
mkdir -p /home/wenzhe/wm_ws/DrivoR/exp
ln -sf /home/wenzhe/wm_ws/WoTE/exp/metric_cache /home/wenzhe/wm_ws/DrivoR/exp/metric_cache
```

验证兼容性:
- 格式: lzma 压缩的 `MetricCache` pickle 对象
- 字段: `file_path`, `trajectory`, `ego_state`, `observation`, `centerline`, `route_lane_ids`, `drivable_area_map`
- metadata CSV: `/home/wenzhe/wm_ws/WoTE/exp/metric_cache/metadata/metric_cache_metadata_node_0.csv`
- 条目数: 69,711

### 5.2 重新生成 Metric Cache (如需)

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

---

## 6. 运行 PDMS 评估

### 6.1 DrivoR 模型配置参数

以下参数对应 `drivor_Nav1_25epochs.pth` 权重的架构：

```yaml
agent.config.proposal_num: 64          # 轨迹提案数量
agent.config.refiner_ls_values: 0.0
agent.config.image_backbone.focus_front_cam: false
agent.config.one_token_per_traj: true
agent.config.refiner_num_heads: 1
agent.config.tf_d_model: 256           # Transformer 隐藏维度
agent.config.tf_d_ffn: 1024            # FFN 维度
agent.config.area_pred: false
agent.config.agent_pred: false
agent.config.ref_num: 4                # register token 数量
agent.config.noc: 1                    # No-at-fault Collision 权重
agent.config.dac: 1                    # Drivable Area Compliance 权重
agent.config.ddc: 0.0                  # Driving Direction Compliance 权重 (=0)
agent.config.ttc: 5                    # Time to Collision 权重
agent.config.ep: 5                     # Ego Progress 权重
agent.config.comfort: 2                # Comfort 权重
```

### 6.2 单 GPU 评估 (推荐)

DDP 多 GPU 评估在 GPU 负载不均衡时容易出现 NCCL 超时。**推荐使用单 GPU**。

```bash
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 \
NUPLAN_MAP_VERSION="nuplan-maps-v1.0" \
NUPLAN_MAPS_ROOT="/home/wenzhe/wm_ws/WoTE/dataset/maps" \
OPENSCENE_DATA_ROOT="/home/wenzhe/wm_ws/WoTE/dataset" \
NAVSIM_DEVKIT_ROOT="/home/wenzhe/wm_ws/DrivoR" \
NAVSIM_EXP_ROOT="/home/wenzhe/wm_ws/DrivoR/exp" \
SUBSCORE_PATH="/home/wenzhe/wm_ws/DrivoR/exp" \
nohup /home/wenzhe/miniconda/envs/drivoR/bin/python -u \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py \
    train_test_split=navtest \
    agent=drivoR \
    agent.checkpoint_path="$NAVSIM_DEVKIT_ROOT/weights/drivor_Nav1_25epochs.pth" \
    experiment_name=drivoR_nav1_eval \
    trainer.params.devices=1 \
    trainer.params.strategy=auto \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.noc=1 \
    agent.config.dac=1 \
    agent.config.ddc=0.0 \
    agent.config.ttc=5 \
    agent.config.ep=5 \
    agent.config.comfort=2 \
    > $NAVSIM_EXP_ROOT/eval_output.log 2>&1 &

echo "PID: $!"
```

### 6.3 多 GPU 评估 (仅在所有 GPU 空闲时)

```bash
# 确保所有 GPU 空闲，否则会因 NCCL 超时失败
# 去掉 trainer.params.devices=1 和 trainer.params.strategy=auto
# 默认使用所有可见 GPU 的 DDP 策略
CUDA_VISIBLE_DEVICES=0,1,3 python -u \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py \
    ...  # 同上参数，去掉 trainer.params.devices/strategy
```

### 6.4 监控进度

```bash
# 实时日志
tail -f $NAVSIM_EXP_ROOT/eval_output.log

# 进程状态
ps -p <PID> -o pid,%cpu,etime

# GPU 使用
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# 推理进度 (从日志中)
grep "Predicting DataLoader" $NAVSIM_EXP_ROOT/eval_output.log | tail -5
```

### 6.5 预计时间线 (单 GPU, RTX 6000 Ada 48GB)

| 阶段 | 时间 | 说明 |
|------|------|------|
| Hydra 配置初始化 | ~10 分钟 | navtest.yaml 有 285k 字符 (12,146 token 列表) |
| 场景加载 | ~30 秒 | 加载 136 个 pkl 文件 → 12,146 个场景 |
| 模型初始化 | ~30 秒 | 加载 DINOv2 backbone + checkpoint |
| GPU 推理 | ~25-40 分钟 | 380 batches (batch_size=32) |
| PDM Scoring (CPU) | ~20-30 分钟 | Ray workers 并行评分 |
| **总计** | **~60-80 分钟** | |

---

## 7. 输出文件

评估完成后生成以下文件：

| 路径 | 说明 |
|------|------|
| `$NAVSIM_EXP_ROOT/navsim1_pdm_scores/drivoR_nav1_eval/<timestamp>.pkl` | 所有场景的预测轨迹 (pickle) |
| `$NAVSIM_EXP_ROOT/ke/drivoR_nav1_eval/<run_time>/<timestamp>.csv` | PDMS 评分结果 CSV |

CSV 包含以下列:
- `token`: 场景标识符 (最后一行为 `average`)
- `valid`: 评分是否成功
- `score`: 最终 PDMS 分数
- `no_at_fault_collisions`, `drivable_area_compliance`, `time_to_collision_within_bound`, `ego_progress`, `comfort`, `driving_direction_compliance`: 六个子分数

PDMS 计算公式: `PDMS = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12`

---

## 8. 已知问题与解决方案

### 8.1 `ModuleNotFoundError: No module named 'absl'`

**原因**: `~/.local/lib/python3.8/site-packages/` 中有 tensorboard 依赖 absl，但 drivoR 环境未安装。
**解决**: `pip install absl-py`

### 8.2 `ModuleNotFoundError: No module named 'numpy'` (安装 absl-py 后)

**原因**: pip 安装 absl-py 时升级了 typing-extensions，触发了依赖冲突导致 numpy 被移除。
**解决**: 重新安装 numpy: `pip install numpy==1.24.4`

### 8.3 NCCL Watchdog Timeout (DDP 多 GPU)

**错误**: `WorkNCCL(SeqNum=13, OpType=ALLREDUCE) ran for 1800242 milliseconds before timing out`
**原因**: GPU 负载不均衡（某个 GPU 被其他进程大量占用），DDP worker 之间同步等待超时 (默认 1800 秒)。
**解决**:
1. 使用单 GPU: `trainer.params.devices=1 trainer.params.strategy=auto`
2. 或确保所有 GPU 空闲后再运行多 GPU 评估
3. 或增大 `distributed_timeout_seconds` 配置

### 8.4 `conda run --no-banner` 不支持

**原因**: 旧版 conda 不支持 `--no-banner` 参数。
**解决**: 使用 `conda run` (不带 `--no-banner`) 或直接用 Python 路径。

### 8.5 输出缓冲导致看不到进度

**原因**: `conda run` 会缓冲所有 stdout/stderr，DDP 子进程的输出也不会显示在主终端。
**解决**:
1. 设置 `PYTHONUNBUFFERED=1`
2. 使用 `python -u` 运行
3. 直接用 Python 绝对路径而不是 `conda run`
4. 使用 `nohup` + 日志文件重定向

---

## 9. 目录结构

```
DrivoR/
├── weights/
│   ├── vit_small_patch14_reg4_dinov2.lvd142m/   # DINOv2 backbone
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── pytorch_model.bin
│   └── drivor_Nav1_25epochs.pth                  # DrivoR Nav1 checkpoint
├── exp/
│   ├── metric_cache -> /home/wenzhe/wm_ws/WoTE/exp/metric_cache  # 符号链接
│   ├── navsim1_pdm_scores/drivoR_nav1_eval/      # 预测轨迹 pkl
│   ├── ke/drivoR_nav1_eval/                       # Hydra 输出 + CSV 结果
│   └── eval_output.log                            # 评估日志
├── navsim/                                        # NAVSIM 源代码 (含 DrivoR agent)
├── nuplan-devkit/                                 # nuPlan devkit
├── scripts/evaluation/
│   └── run_drivor_nav1_pdms.sh                    # 评估脚本
└── docs/
    ├── setup_and_eval_guide.md                    # 本文档
    └── evaluation_pipeline_proposal.md            # 增强评估管线提案
```
