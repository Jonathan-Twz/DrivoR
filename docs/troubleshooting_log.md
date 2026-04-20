# DrivoR PDMS 评估 — 故障排查记录

> 日期: 2026-04-01
> 记录在 NAVSIM-v1 navtest 分割上评估 DrivoR 时遇到的所有问题和解决过程。

---

## 时间线总览

| 时间 (UTC+8) | 事件 | 结果 |
|---|---|---|
| 04:30 | 开始环境搭建 | - |
| 04:35 | 创建 conda env `drivoR`, python=3.8 | 成功 |
| 04:40 | 安装 torch 2.1.0+cu121 | 成功 |
| 04:42 | 安装 nuplan-devkit + DrivoR navsim | 成功，但部分依赖缺失 |
| 04:46 | 验证 DrivoRAgent 导入 | 成功 |
| 04:48 | 下载 DINOv2 backbone 权重 (HuggingFace) | 成功 (~2s) |
| 04:49 | 下载 drivor_Nav1_25epochs.pth (GitHub Releases, 292MB) | 成功 (~12s) |
| 04:52 | 创建 metric cache 符号链接 | 成功 |
| 04:52 | 验证 metric cache 兼容性 (69,711 条) | 成功 |
| 04:53 | **第 1 次评估尝试** (conda run, 4 GPU DDP) | `conda run --no-banner` 参数不支持 |
| 04:54 | **第 2 次评估尝试** (conda run, 4 GPU DDP) | 运行 ~60 分钟, exit_code=1, 输出被缓冲看不到错误 |
| 06:17 | **第 3 次评估尝试** (python -u, 4 GPU DDP) | 输出仍被缓冲 (DDP 子进程) |
| 06:48 | 单独测试场景加载 | 成功: Hydra init ~10min, 场景加载 ~30s, 12,146 scenes |
| 07:05 | **第 4 次评估尝试** (python -u, 3 GPU DDP, CUDA 0,1,3) | 到达 DDP init, 但 rank 0 崩溃: `No module named 'absl'` |
| 07:58 | 安装 absl-py | 同时导致 numpy 被卸载 |
| 08:12 | **第 5 次评估尝试** (PYTHONNOUSERSITE=1) | `No module named 'numpy'` — PYTHONNOUSERSITE 过于激进 |
| 08:43 | **第 6 次评估尝试** (去掉 PYTHONNOUSERSITE) | 仍然 `No module named 'numpy'` — numpy 确实没装 |
| 09:00 | 修复: 安装 numpy, sympy, scipy, pytorch-lightning | 全部安装成功 |
| 09:05 | 验证所有导入 | 全部成功 |
| 09:26 | **第 7 次评估尝试** (python -u, 3 GPU DDP, CUDA 0,1,3) | DDP 启动成功，推理开始 |
| 10:10 | 推理进度: 24% (31/127 batches) | 正在运行 |
| 11:30 | 推理进度: 47% (60/127 batches) | 正在运行 |
| 12:22 | 推理进度: 57% (73/127 batches) | **NCCL 超时崩溃** (Rank 2) |
| 13:37 | 确认进程终止 | exit_code=137 (SIGKILL) |
| 15:18 | **第 8 次评估尝试** (单 GPU, devices=1, strategy=auto) | 被用户 abort |
| 15:35 | **第 9 次评估尝试** (nohup 后台, 单 GPU) | 正在运行中... |

---

## 问题详解

### 问题 1: `conda run --no-banner` 不兼容

```
conda: error: unrecognized arguments: --no-banner
```

- **根因**: 服务器上 conda 版本较老，不支持 `--no-banner`
- **解决**: 去掉 `--no-banner` 参数

### 问题 2: `conda run` 缓冲所有输出

- **现象**: 评估运行 60 分钟，终端文件始终只有 7 行 (header)
- **根因**: `conda run` 捕获子进程的 stdout/stderr 并在完成后一次性输出
- **解决**: 不用 `conda run`，直接用 Python 绝对路径 + `PYTHONUNBUFFERED=1` + `python -u`

### 问题 3: DDP 子进程输出不可见

- **现象**: 即使用 `python -u`，DDP 子进程的输出不会出现在主进程终端
- **根因**: PyTorch Lightning DDP 策略 fork 子进程，子进程有独立的 stdout
- **影响**: 只能通过 Hydra 日志文件 (`exp/ke/.../log.txt`, `train_ddp_process_*.log`) 查看子进程输出
- **解决**: 检查 `exp/ke/drivoR_nav1_eval/<run_time>/` 下的日志

### 问题 4: `ModuleNotFoundError: No module named 'absl'`

```python
File "tensorboard/compat/tensorflow_stub/flags.py", line 25
    from absl.flags import *
ModuleNotFoundError: No module named 'absl'
```

- **根因**: 用户全局 `~/.local/lib/python3.8/site-packages/` 中有 tensorboard，它依赖 absl，但 drivoR conda 环境未安装
- **触发时机**: PyTorch Lightning 初始化 logger 时尝试导入 tensorboard
- **解决**: `pip install absl-py`
- **副作用**: 安装 absl-py 时 pip 升级了 `typing-extensions` (→4.13.2)，这导致了下一个问题

### 问题 5: numpy 被卸载

- **根因**: pip 依赖解析器在安装 absl-py + typing-extensions 时，因版本冲突卸载了 numpy
- **解决**: `pip install numpy==1.24.4`

### 问题 6: 更多缺失包 (级联效应)

安装过程中逐步发现缺失:
1. `sympy` — torch 2.1.0 导入时需要
2. `scipy` — navsim 需要
3. `pytorch-lightning` — 被 pip 依赖解析移除
4. `selenium` — navsim 声明依赖

**教训**: DrivoR 的 `setup.py` 声明的依赖版本与实际兼容版本有差异，pip 安装新包时容易触发级联卸载。

### 问题 7: NCCL Watchdog 超时 (最严重)

```
[Rank 2] Watchdog caught collective operation timeout: 
WorkNCCL(SeqNum=13, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=1800000) 
ran for 1800242 milliseconds before timing out.
```

- **根因**: 服务器上 GPU 2 被其他用户进程占用 ~21GB 显存，导致该 GPU 上的 DDP worker 处理速度远慢于其他 worker。DDP 要求所有 rank 同步，超过 30 分钟 (1800s) 未同步就超时。
- **GPU 占用情况**:
  ```
  GPU 0: 3280 MiB (基本空闲)
  GPU 1: 4237 MiB (少量占用)
  GPU 2: 21068 MiB (大量占用 ← 问题 GPU)
  GPU 3: 4241 MiB (少量占用)
  ```
- **影响**: 3 GPU DDP 在推理 57% (73/127 batches, ~3 小时) 后崩溃，所有工作丢失
- **解决方案**:
  1. **推荐**: 使用单 GPU (`trainer.params.devices=1 trainer.params.strategy=auto`)
  2. 只在所有 GPU 完全空闲时使用多 GPU
  3. 增大 `distributed_timeout_seconds` 配置 (默认 7200s)
  4. 用 `CUDA_VISIBLE_DEVICES` 排除繁忙 GPU

### 问题 8: Hydra 初始化极慢 (~10 分钟)

- **根因**: `navtest.yaml` scene filter 文件有 285,662 字符（包含 12,146 个 token 和 136 个 log name 的完整列表），Hydra/OmegaConf 解析这个巨大 YAML 非常耗时
- **影响**: 每次运行开始前都有 ~10 分钟的"空白期"，容易误以为进程卡死
- **缓解**: 这是正常行为，不需要修复。耐心等待即可。

---

## 关键经验总结

1. **不要用 `conda run`** 运行长时间进程 — 输出缓冲会让你完全看不到进度和错误
2. **直接用 Python 绝对路径** + `PYTHONUNBUFFERED=1` 运行
3. **用 `nohup` + 日志文件** 防止终端断开影响
4. **安装新 pip 包后必须验证** numpy, torch 等核心包是否仍在
5. **多 GPU DDP 前检查 GPU 占用** — `nvidia-smi` 确认所有目标 GPU 空闲
6. **单 GPU 更可靠** — 除非确认 GPU 全部空闲，否则用单 GPU
7. **Hydra 初始化慢是正常的** — navtest filter 很大，等 10 分钟
8. **DDP 子进程日志** 在 `exp/ke/<experiment_name>/<run_time>/train_ddp_process_*.log`

---

## 评估流程的关键文件

| 文件 | 作用 |
|------|------|
| `navsim/planning/script/run_pdm_score_multi_gpu.py` | 多 GPU 评估入口 (也支持单 GPU) |
| `navsim/planning/script/run_pdm_score.py` | 单 GPU 评估入口 (无 DDP, 使用 Ray worker) |
| `navsim/planning/script/config/pdm_scoring/default_run_pdm_score_gpu.yaml` | 多 GPU 评估的 Hydra 配置 |
| `navsim/planning/script/config/pdm_scoring/ddp.yaml` | DDP trainer 配置 (batch_size, num_workers 等) |
| `navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml` | PDM 评分参数 (权重, simulator, scorer) |
| `navsim/planning/script/config/common/agent/drivoR.yaml` | DrivoR agent 完整配置 |
| `navsim/planning/script/config/common/train_test_split/navtest.yaml` | navtest 分割定义 (data_split: test) |
| `navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml` | navtest scene filter (12,146 tokens) |
| `navsim/evaluate/pdm_score.py` | PDM 评分核心函数 |
| `navsim/common/dataloader.py` | SceneLoader + MetricCacheLoader |
| `navsim/planning/training/agent_lightning_module.py` | PL Module (predict_step) |
