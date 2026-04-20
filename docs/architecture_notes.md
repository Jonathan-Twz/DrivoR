# DrivoR 架构与代码分析笔记

> 日期: 2026-04-01
> 基于对 DrivoR (valeoai/DrivoR) 仓库的详细代码审查。

---

## 1. 项目概览

DrivoR 是一个端到端自动驾驶模型，发表于论文 "DrivoR: Driving with Robust Trajectory Scoring"。核心思想：利用 DINOv2 视觉特征 + Transformer 解码器生成多条轨迹提案，并通过学习到的评分网络选择最优轨迹。

**关键创新**:
- 轨迹提案生成 + 评分选择架构
- 利用 DINOv2 预训练视觉特征
- 基于 NAVSIM 框架的闭环评估

---

## 2. 模型架构

### 2.1 整体流程

```
多视角相机图像
      ↓
DINOv2 ViT-S/14 (frozen backbone)
      ↓
特征提取 + Token 化
      ↓
Transformer Decoder (d_model=256, d_ffn=1024)
  - Cross-attention to visual tokens
  - Self-attention among trajectory tokens
      ↓
轨迹提案 (proposal_num=64 条候选轨迹)
      ↓
Refiner + 评分网络
      ↓
选择最优轨迹 → 输出 8 步 waypoints (4 秒 @ 2Hz)
```

### 2.2 关键组件

| 组件 | 文件 | 说明 |
|------|------|------|
| DrivoRAgent | `navsim/agents/drivoR/drivor_agent.py` | Agent 入口，实现 `AbstractAgent` 接口 |
| DrivoRConfig | `navsim/agents/drivoR/drivor_config.py` | 模型超参数 (Hydra structured config) |
| 图像 Backbone | `navsim/agents/drivoR/image_backbone.py` | DINOv2 ViT-S 封装，支持 `focus_front_cam` |
| Transformer | `navsim/agents/drivoR/` | 解码器层，轨迹 token 生成 |
| 权重路径 | `weights/vit_small_patch14_reg4_dinov2.lvd142m/` | DINOv2 backbone |

### 2.3 Hydra 配置结构

```yaml
# navsim/planning/script/config/common/agent/drivoR.yaml
_target_: navsim.agents.drivoR.drivor_agent.DrivoRAgent
checkpoint_path: ${oc.env:NAVSIM_DEVKIT_ROOT}/weights/drivor_Nav1_25epochs.pth
config:
  image_backbone:
    backbone_name: vit_small_patch14_reg4_dinov2.lvd142m
    backbone_pretrained: true
    pretrained_cfg_overlay:
      file: ${oc.env:NAVSIM_DEVKIT_ROOT}/weights/vit_small_patch14_reg4_dinov2.lvd142m/pytorch_model.bin
    focus_front_cam: false
  proposal_num: 64
  tf_d_model: 256
  tf_d_ffn: 1024
  one_token_per_traj: true
  refiner_num_heads: 1
  ref_num: 4
  # 子分数权重
  noc: 1      # No at-fault Collisions
  dac: 1      # Drivable Area Compliance
  ddc: 0.0    # Driving Direction Compliance
  ttc: 5      # Time to Collision
  ep: 5       # Ego Progress
  comfort: 2  # Comfort
```

---

## 3. NAVSIM 评估管线

### 3.1 代码流 (`run_pdm_score_multi_gpu.py`)

```python
@hydra.main(config_path=..., config_name="default_run_pdm_score_gpu")
def main(cfg: DictConfig) -> None:
    # 1. 加载场景 (SceneLoader) — 从 pkl 文件读取
    scene_loader = SceneLoader(
        sensor_blobs_path=...,
        data_path=...,
        scene_filter=cfg.scene_filter
    )
    scenes = scene_loader.get_scene_ids()

    # 2. 加载 metric cache (MetricCacheLoader)
    metric_cache_loader = MetricCacheLoader(
        cache_path=cfg.metric_cache_path
    )

    # 3. 实例化 Agent (DrivoRAgent)
    agent = instantiate(cfg.agent)

    # 4. 构建 PyTorch Lightning Module + DataLoader
    lightning_module = AgentLightningModule(agent=agent)
    dataloader = DataLoader(dataset, ...)

    # 5. GPU 推理 (trainer.predict → DDP)
    trainer = pl.Trainer(**cfg.trainer.params)
    predictions = trainer.predict(lightning_module, dataloader)

    # 6. 保存预测轨迹 (pkl)
    save_to(predictions, cfg.output_path)

    # 7. CPU PDM 评分 (worker_map → 并行)
    results = worker_map(
        pdm_score_worker,
        predictions,
        metric_caches,
        scoring_cfg
    )

    # 8. 生成 CSV 报告
    generate_report(results, cfg.output_csv)
```

### 3.2 PDM Score (PDMS) 计算

PDM Score 由 6 个子指标复合而成：

| 子指标 | 缩写 | 含义 | 范围 |
|--------|------|------|------|
| No at-fault Collisions | NC | 无过错碰撞 | 0/1 (布尔) |
| Drivable Area Compliance | DAC | 可行驶区域合规 | 0/1 (布尔) |
| Time to Collision | TTC | 碰撞时间裕度 | [0, 1] |
| Ego Progress | EP | 自车行驶进度 | [0, 1] |
| Comfort | C | 驾驶舒适度 | [0, 1] |
| Driving Direction Compliance | DDC | 行驶方向合规 | [0, 1] |

**公式**:
```
PDMS = NC × DAC × weighted_avg(TTC, EP, C, DDC)
     = NC × DAC × (w_ttc*TTC + w_ep*EP + w_c*C + w_ddc*DDC) / (w_ttc + w_ep + w_c + w_ddc)
```

DrivoR 默认权重: `ttc=5, ep=5, comfort=2, ddc=0`
即: `PDMS = NC × DAC × (5×TTC + 5×EP + 2×C) / 12`

### 3.3 数据集分割

| 分割 | 使用 | 场景数 | 数据子集 |
|------|------|--------|----------|
| navtrain | 训练 | ~100k | trainval |
| navtest | 评估 PDMS | 12,146 | test |
| navtest_mini | 快速测试 | ~400 | test |

---

## 4. DrivoR 与 WoTE 的关系

DrivoR 和 WoTE (World-model-based Trajectory Evaluator) 都基于 NAVSIM 框架，但它们是**独立的 agent 实现**。

### 4.1 共享部分

| 资源 | 共享？ | 说明 |
|------|--------|------|
| NAVSIM 数据集 (logs, sensor_blobs) | ✅ | 完全相同的数据 |
| nuPlan 地图 | ✅ | 完全相同 |
| Metric Cache | ✅ | 格式相同 (lzma pkl)，可复用 |
| NAVSIM devkit 代码 | ⚠️ | 版本可能不同，DrivoR 使用 navsim 1.1.0 |
| Agent 模型 | ❌ | 完全不同的架构和权重 |
| Conda 环境 | ❌ | 应独立安装，避免依赖冲突 |

### 4.2 路径映射

```
WoTE 数据路径           →  DrivoR 使用的环境变量
/wm_ws/WoTE/dataset/    →  OPENSCENE_DATA_ROOT
/wm_ws/WoTE/dataset/maps →  NUPLAN_MAPS_ROOT
/wm_ws/WoTE/exp/metric_cache  →  NAVSIM_EXP_ROOT/metric_cache (通过符号链接)

DrivoR 自有路径
/wm_ws/DrivoR/          →  NAVSIM_DEVKIT_ROOT (Hydra 配置根)
/wm_ws/DrivoR/exp/      →  NAVSIM_EXP_ROOT, SUBSCORE_PATH
```

---

## 5. 代码要点

### 5.1 数据加载

- `SceneLoader` (`navsim/common/dataloader.py`): 从 `${OPENSCENE_DATA_ROOT}/navsim_logs/<split>/` 加载 pkl 文件
- `MetricCacheLoader`: 从 `${NAVSIM_EXP_ROOT}/metric_cache/` 加载，期望 lzma 压缩的 `MetricCache` 对象 + `metadata.csv`
- 场景过滤: 由 `scene_filter/*.yaml` 中的 `log_names` 和 `tokens` 列表控制

### 5.2 推理流程

```python
# AgentLightningModule.predict_step()
def predict_step(self, batch, batch_idx):
    # batch: AgentInput (images, ego_state, route, etc.)
    # 调用 agent.forward() 获取轨迹
    trajectory = self.agent.forward(batch)
    return {"token": batch.token, "trajectory": trajectory}
```

### 5.3 Agent 接口

所有 agent 必须实现 `AbstractAgent`:
- `name() -> str`: agent 名称
- `initialize()`: 加载权重等
- `get_sensor_config() -> SensorConfig`: 需要哪些传感器
- `forward(agent_input: AgentInput) -> AbstractTrajectory`: 推理

### 5.4 多 GPU 策略

- `run_pdm_score_multi_gpu.py`: 使用 PL DDP 进行推理，然后在 rank 0 上用 CPU 评分
- `run_pdm_score.py`: 单 GPU 推理 + Ray 并行 CPU 评分
- DDP 推理后需要 `all_gather` 收集所有 rank 的预测结果

---

## 6. 参考论文

- **DrivoR**: "DrivoR: Driving with Robust Trajectory Scoring" (Valeo AI)
  - GitHub: https://github.com/valeoai/DrivoR
  - 在 NAVSIM v1 navtest 上报告的 PDMS: ~87.0
- **NAVSIM**: "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking"
  - Benchmark: https://github.com/autonomousvision/navsim
  - 排行榜: https://huggingface.co/spaces/AGC2024-P/NAVSIM-v1
- **DINOv2**: "DINOv2: Learning Robust Visual Features without Supervision"
  - 使用 ViT-S/14 (vit_small_patch14_reg4_dinov2.lvd142m)
