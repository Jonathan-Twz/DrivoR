---
name: navsim-bev-plan-vs-gt-visualization
description: >-
  Produces NAVSIM v1 birds-eye-view plots comparing the agent open-loop plan to human GT using
  navsim.visualization.plot_bev_with_agent, SceneLoader, and PDMS CSV token selection. Use when
  the user wants qualitative BEV visualization, failure-case inspection, plan-vs-GT overlays,
  or porting the same workflow to another NAVSIM-based driving repo (DrivoR, WoTE fork, upstream navsim).
---

# NAVSIM v1：BEV 对比「规划 vs GT」

## 何时使用本 Skill

- 需要在 **NAVSIM v1**（OpenScene pickle + `navtest` 等 split）上 **定性** 看模型轨迹与人类轨迹差异。
- 已从 **PDMS 评估**得到逐场景 CSV（`token`, `score`, 子指标, `valid`），想按 **失败/弱分** 或 **高分抽样** 选 token 出图。
- 要把同一流程 **复制到另一个仓库**（保持 API：`plot_bev_with_agent`、`SceneLoader`、`AbstractAgent.compute_trajectory`）。

## 核心概念（必须区分）

| 名称 | 含义 |
|------|------|
| **GT（human）** | `Scene.get_future_trajectory()`，人类标注的未来 ego 轨迹（局部坐标）。 |
| **Plan（agent）** | `AbstractAgent.compute_trajectory(AgentInput)` 的 **单次前向**输出，与 `forward` → `trajectory` 一致。 |
| **与 PDMS 的关系** | PDMS 可能对 **多 proposal** 做仿真与选优；BEV 上这条 plan **不一定**等于 PDMS 最终选用的那条。解释图表时要说清是「开环规划 vs GT」。 |

颜色约定（默认 `navsim/visualization/config.py` 中 `TRAJECTORY_CONFIG`）：human 与 agent 使用不同配色，便于区分。

## 依赖环境

- 与 PDMS 评估相同：`OPENSCENE_DATA_ROOT`、`NUPLAN_MAPS_ROOT`、`NUPLAN_MAP_VERSION`。
- 数据布局：`navsim_logs/{split}`、`sensor_blobs/{split}`（如 navtest → `data_split: test`）。
- Python 环境需已安装该仓库的 `navsim` 包及 agent 依赖（PyTorch、timm、等）。

## 实现要点（移植到他项目时照抄）

1. **SceneFilter**  
   与评估 **同一** `train_test_split`（例如 `navtest` → `scene_filter/navtest`），避免 CSV 里有 token 但 loader 筛不到。

2. **SensorConfig**  
   必须与 agent 推理一致（`agent.get_sensor_config()`），否则特征与评估不一致。

3. **实例化 Agent**  
   从 `agent/drivoR.yaml`（或等价配置）用 Hydra `compose` + `instantiate`；若 YAML 含 `${trainer...}` 等仅训练用的插值，用 override 消掉（例如 `scheduler_args.num_epochs=1`、`batch_size=64`），并设置 `checkpoint_path`。

4. **选 token（阶段一：失败优先）**  
   - `valid == False` 优先（若有）。  
   - 再按 `score` **升序**取 bottom N 或 bottom 百分位。  
   - 可选：对单列（如 `ego_progress`）取最差 K 条以区分 failure mode。  
   - **与 `SceneLoader.tokens` 求交集**后再渲染；无法加载的 token 记入日志。

5. **阶段二：成功对照（可选）**  
   `score >=` 阈值后随机抽 M 条，输出到单独子目录（如 `success_sample/`）。

6. **绘图**  
   - `scene = scene_loader.get_scene_from_token(token)`  
   - `fig, ax = plot_bev_with_agent(scene, agent)`  
   - `matplotlib` 无显示器时用 `Agg` 后端，`savefig` 后 `plt.close(fig)`。

7. **可选**  
   - `plot_cameras_frame_with_annotations`：相机网格 + 中心 BEV。  
   - 仅地图/框：`plot_bev_frame`。  
   - LiDAR：在 `BEV_PLOT_CONFIG["layers"]` 中加入 `"lidar"`（更重）。

## DrivoR 仓库中的现成脚本（参考实现）

若当前项目为 **DrivoR**，可直接使用：

```bash
cd <DRIVOR_REPO>
export OPENSCENE_DATA_ROOT=... NUPLAN_MAPS_ROOT=... NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=...

python scripts/visualization/bev_plan_vs_gt_navtest.py \
  --csv exp/ke/drivoR_nav1_eval/<run>/<timestamp>.csv \
  --out_dir exp/viz_navtest_plan_vs_gt/<run_name> \
  --checkpoint weights/drivor_Nav1_25epochs.pth \
  --n_worst 20 \
  --eval_like
```

- `--eval_like`：与 `scripts/evaluation/run_drivor_nav1_pdms.sh` 中的 agent 超参对齐。  
- `--mode success --n_success 10 --min_score 0.99`：高分抽样。  
- `--also_cameras`：额外导出相机拼图。  
- `--dry_run`：只写 `manifest.json`。

## 性能提示

- 若 `SceneFilter.tokens` 设为仅目标 token，loader 仍会遍历 split 下各 log pickle 以查找匹配；大批量时首次扫描较慢，属预期。  
- 逐场景推理 + 绘图：每张图可能数分钟，批量时建议后台 `nohup` 与日志重定向。

## 移植清单（带到新项目）

- [ ] 复制本 Skill 所在目录，或复制 `SKILL.md` 到 `.cursor/skills/navsim-bev-plan-vs-gt-visualization/`。  
- [ ] 将脚本中的 Hydra 路径改为新仓库的 `navsim/planning/script/config/common/...`。  
- [ ] 将 `_compose_drivor_agent` 替换为新项目的 agent 类与 YAML。  
- [ ] 确认 `plot_bev_with_agent` 与 `SceneLoader` API 与当前 `navsim` 版本一致（v1 fork vs upstream v2 有差异时不要混用）。

## 相关文件（DrivoR）

- 脚本：`scripts/visualization/bev_plan_vs_gt_navtest.py`  
- 会话记录：`docs/session_history.md`  
- PDMS 分数摘要：`exp/pdms_navsim_v1_scores.txt`
