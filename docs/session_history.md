# DrivoR 工作会话记录

> 最后更新: 2026-04-02

---

## 会话 1: 代码审查与评估管线提案

**日期**: 2026-04-01
**目标**: 深入审查 DrivoR 仓库，提出增强评估管线方案
**产出**: `docs/evaluation_pipeline_proposal.md`

### 完成的工作

1. 完整审查了 DrivoR 仓库代码结构
2. 分析了模型架构 (DINOv2 + Transformer decoder)
3. 理解了 NAVSIM 评估管线 (PDMS 计算流程)
4. 撰写了详细的增强评估管线提案 (686 行文档)

### 关键发现

- DrivoR 基于 NAVSIM 1.1.0，支持 Nav1 (PDMS) 和 Nav2 (EPDMS) 评估
- 模型使用 DINOv2 ViT-S 作为视觉 backbone，参数约 22M
- 评估支持单 GPU 和多 GPU (DDP) 两种模式
- 配置管理使用 Hydra，结构化配置 + YAML override

---

## 会话 2: NAVSIM-v1 PDMS 评估执行

**日期**: 2026-04-01 ~ 2026-04-02
**目标**: 在 NAVSIM-v1 navtest 分割上运行 DrivoR 的 PDMS 评估
**状态**: ✅ 已完成（navtest PDMS）

### 步骤与执行

#### Step 1: 创建 conda 环境 ✅

```bash
conda create -n drivoR python=3.8 -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e /home/wenzhe/wm_ws/DrivoR/nuplan-devkit
pip install -e /home/wenzhe/wm_ws/DrivoR
```

#### Step 2: 下载权重 ✅

- DINOv2 backbone: `weights/vit_small_patch14_reg4_dinov2.lvd142m/` (HuggingFace)
- DrivoR Nav1: `weights/drivor_Nav1_25epochs.pth` (GitHub Releases, 292MB)

#### Step 3: 配置环境变量 ✅

- `OPENSCENE_DATA_ROOT` → WoTE 数据集
- `NUPLAN_MAPS_ROOT` → WoTE 地图
- `NAVSIM_DEVKIT_ROOT` → DrivoR 仓库
- `NAVSIM_EXP_ROOT` → DrivoR 实验目录

#### Step 4: 验证 metric cache 兼容性 ✅

- WoTE 的 metric cache (69,711 条) 与 DrivoR 完全兼容
- 通过符号链接挂载: `exp/metric_cache → /wm_ws/WoTE/exp/metric_cache`

#### Step 5: 运行评估 ✅

经历了 **9 次尝试**，详见 `docs/troubleshooting_log.md`。最终采用 **单 GPU**（`+trainer.params.devices=1`），避免 DDP NCCL 超时。

**最终结果（navtest）**
- 平均 PDMS: **0.937857**
- 成功场景: 12146，失败: 0
- 结果 CSV: `exp/ke/drivoR_nav1_eval/04.01_17.16/2026.04.02.02.35.44.csv`
- 日志: `exp/ke/drivoR_nav1_eval/04.01_17.16/run_pdm_score_multi_gpu.log`
- 分数摘要: `exp/pdms_navsim_v1_scores.txt`

最终使用的命令（示例）:
```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
NUPLAN_MAP_VERSION="nuplan-maps-v1.0" \
NUPLAN_MAPS_ROOT="/home/wenzhe/wm_ws/WoTE/dataset/maps" \
OPENSCENE_DATA_ROOT="/home/wenzhe/wm_ws/WoTE/dataset" \
NAVSIM_DEVKIT_ROOT="/home/wenzhe/wm_ws/DrivoR" \
NAVSIM_EXP_ROOT="/home/wenzhe/wm_ws/DrivoR/exp" \
SUBSCORE_PATH="/home/wenzhe/wm_ws/DrivoR/exp" \
nohup /home/wenzhe/miniconda/envs/drivoR/bin/python -u \
    /home/wenzhe/wm_ws/DrivoR/navsim/planning/script/run_pdm_score_multi_gpu.py \
    train_test_split=navtest \
    agent=drivoR \
    agent.checkpoint_path="/home/wenzhe/wm_ws/DrivoR/weights/drivor_Nav1_25epochs.pth" \
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
    > /home/wenzhe/wm_ws/DrivoR/exp/eval_output.log 2>&1 &
```

#### Step 6: 结果分析 ✅

- PDMS 已汇总于 `exp/pdms_navsim_v1_scores.txt` 与上述 CSV。

---

## 会话 3: NAVSIM v1 BEV 可视化（plan vs GT）

**日期**: 2026-04-02  
**目标**: 使用 NAVSIM 自带 `plot_bev_with_agent` 对比开环规划轨迹与人类 GT；先筛失败/弱分 token，再可选成功抽样。

### 产出

- **脚本**: `scripts/visualization/bev_plan_vs_gt_navtest.py`（`--mode failure|success`，`--n_worst`，`--eval_like`，`--also_cameras` 等）
- **示例输出**: `exp/viz_navtest_plan_vs_gt/failures_n20/failures/`（20 张 BEV PNG + `manifest.json`）
- **可移植 Skill**: `docs/skills/navsim-bev-plan-vs-gt-visualization/SKILL.md`（已创建：含 YAML `name`/`description`、GT vs Plan 与 PDMS 语义、环境变量、`bev_plan_vs_gt_navtest.py` 示例命令、移植清单；整目录可复制到 `~/.cursor/skills/navsim-bev-plan-vs-gt-visualization/` 供他项目使用）

### 语义说明

- **GT**: `Scene.get_future_trajectory()`（人类未来轨迹）
- **Plan**: `AbstractAgent.compute_trajectory()` 一次前向（与 PDMS 中经多 proposal + 仿真器选优后的轨迹不一定相同）

### 与 NAVSIM v2 / EPDMS 的说明（方案 A）

- 上游 `autonomousvision/navsim` 可跑 EPDMS；本仓库曾尝试时遇 **Ray OOM / GCS** 等问题，EPDMS 未稳定完成；状态见 `exp/epdms_navsim_v2_scores.txt`。

---

## 待办事项

- [x] navtest PDMS 评估完成并记录结果
- [x] NAVSIM v1 BEV 失败 case 批量出图（脚本 + 20 张示例）
- [ ] 可选: 在 navtest_mini 上做快速验证
- [ ] 可选: 稳定复现 NAVSIM v2 EPDMS（降 Ray 并行 / 内存策略后重跑）
- [ ] 可选: 尝试 SimScale 数据增强权重

---

## 文档索引

| 文档 | 路径 | 内容 |
|------|------|------|
| 本文件 | `docs/session_history.md` | 工作会话时间线与进展 |
| 环境搭建与评估指南 | `docs/setup_and_eval_guide.md` | 完整的复现指南 |
| 故障排查记录 | `docs/troubleshooting_log.md` | 所有遇到的问题和解决方案 |
| 架构分析笔记 | `docs/architecture_notes.md` | DrivoR 代码结构和技术分析 |
| 评估管线提案 | `docs/evaluation_pipeline_proposal.md` | 增强评估管线方案 |
| 评估脚本 | `scripts/evaluation/run_drivor_nav1_pdms.sh` | 自动化评估脚本 |
| BEV plan vs GT | `scripts/visualization/bev_plan_vs_gt_navtest.py` | 可视化脚本 |
| Skill（可移植） | `docs/skills/navsim-bev-plan-vs-gt-visualization/SKILL.md` | 他项目复用说明 |

---

## 关键配置快速参考

```bash
# === 环境 ===
conda activate drivoR
export NAVSIM_DEVKIT_ROOT=/home/wenzhe/wm_ws/DrivoR
export NAVSIM_EXP_ROOT=/home/wenzhe/wm_ws/DrivoR/exp
export OPENSCENE_DATA_ROOT=/home/wenzhe/wm_ws/WoTE/dataset
export NUPLAN_MAPS_ROOT=/home/wenzhe/wm_ws/WoTE/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT

# === 权重 ===
# DINOv2:   weights/vit_small_patch14_reg4_dinov2.lvd142m/
# DrivoR:   weights/drivor_Nav1_25epochs.pth

# === 数据 ===
# Logs:     /wm_ws/WoTE/dataset/navsim_logs/{test,trainval}/
# Sensors:  /wm_ws/WoTE/dataset/sensor_blobs/{test,trainval}/
# Maps:     /wm_ws/WoTE/dataset/maps/
# Cache:    exp/metric_cache → /wm_ws/WoTE/exp/metric_cache
```
