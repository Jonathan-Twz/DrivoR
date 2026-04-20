# DrivoR 增强评估流程提案

> **状态**: 提案草案  
> **适用范围**: DrivoR (CVPR 2026) 端到端驾驶模型  
> **基线**: NAVSIM PDM Score (PDMS / EPDMS)  
> **前置假设**: 环境变量已配置, NAVSIM 数据集与 devkit 已下载

---

## 1. 现有评估流程分析

### 1.1 现有流程概览

当前 DrivoR 的评估基于 NAVSIM 的 **PDM Score (Predictive Driver Model Score)** 框架, 流程分为三个阶段:

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│ Metric Cache │ ──> │  Agent Inference  │ ──> │  PDM Scoring   │
│ (离线预计算)  │     │  (GPU 批量推理)    │     │ (CPU 多线程)    │
└──────────────┘     └──────────────────┘     └────────────────┘
```

**阶段 1 - Metric Cache** (`run_metric_caching.py`):
- 为每个场景 token 预计算: ego state, PDM observation (周围目标物的时序几何), centerline, route lane IDs, drivable area map
- 存储为 lzma 压缩的 pickle 文件
- 仅需计算一次, 可被所有 agent 复用

**阶段 2 - Agent Inference** (`run_pdm_score_multi_gpu.py`):
- 加载训练好的 DrivoR 模型权重
- 通过 PyTorch Lightning Trainer 的 `predict` 接口进行多 GPU 推理
- 输出: 每个场景 token 对应一条 `Trajectory` (8 个 pose, 0.5s 间隔, 4s 时域)
- 轨迹选取: 64 条 proposal 中, 由 learned scorer 预测的 PDM sub-score 加权选择最优

**阶段 3 - PDM Scoring** (`pdm_score.py`):
- 将 ego 坐标系下的轨迹转换为全局坐标
- 通过 LQR + 自行车运动学模型进行 4s 非反应式仿真 (0.1s 间隔, 40 步)
- 计算 6 项子指标并聚合:

| 子指标 | 权重类型 | 取值范围 |
|--------|---------|---------|
| No at-fault Collisions (NC) | 乘法因子 | {0, 0.5, 1} |
| Drivable Area Compliance (DAC) | 乘法因子 | {0, 1} |
| Time to Collision (TTC) | 加权 5 | {0, 1} |
| Ego Progress (EP) | 加权 5 | [0, 1] |
| Comfort (C) | 加权 2 | {0, 1} |
| Driving Direction Compliance (DDC) | 加权 0 | {0, 0.5, 1} |

最终分数: `PDMS = NC × DAC × (5×TTC + 5×EP + 2×C) / 12`

### 1.2 现有流程的局限性

| 编号 | 局限性 | 影响 |
|------|--------|------|
| L1 | **非反应式仿真**: 背景智能体沿录制轨迹行驶, 不对 ego 行为做出反应 | 无法评估交互式场景下的决策质量 |
| L2 | **单轨迹评估**: 仅评估 scorer 选出的最优轨迹, 忽视了 proposal 集合的整体质量 | 遮盖了 proposal generator 与 scorer 各自的贡献 |
| L3 | **场景均等加权**: 所有场景不论难度均等贡献最终平均分 | 简单场景主导分数, 无法反映 corner case 表现 |
| L4 | **无时序一致性评估**: 相邻帧之间的决策连贯性未被度量 | 可能出现轨迹抖动但仍获高分的情况 |
| L5 | **Scorer 校准未评估**: learned scorer 与 ground truth PDM score 的对齐度不参与最终评分 | 无法独立诊断感知/规划/选择各模块的性能 |
| L6 | **缺少效率指标**: 推理延迟、吞吐量未纳入评估 | 无法评估实时部署可行性 |
| L7 | **缺少结构化失败分析**: 仅有 aggregate score, 无法快速定位失败模式 | 调试和改进缺少方向性 |
| L8 | **固定时域**: 仅评估 4s 未来, 不考虑更长时域的规划能力 | 高速场景下 4s 可能不足以反映安全余量 |

---

## 2. 新评估流程提案

### 2.1 设计原则

1. **向后兼容**: 新流程保留 PDMS 作为核心指标, 确保与 NAVSIM leaderboard 可比
2. **模块化解耦**: 每个评估维度可独立运行, 按需组合
3. **可复现性**: 所有评估结果可通过单一配置文件完整复现
4. **最小代码侵入**: 尽量不修改现有 agent 代码, 通过封装和扩展实现

### 2.2 整体架构

```
                         ┌─────────────────────────────┐
                         │   Unified Evaluation Runner  │
                         │    (run_full_evaluation.py)  │
                         └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
    ┌─────────▼─────────┐   ┌──────────▼──────────┐   ┌─────────▼─────────┐
    │  Layer 1: Core    │   │  Layer 2: Diagnostic │   │  Layer 3: System  │
    │  PDMS + EPDMS     │   │  Extended Metrics    │   │  Deployment       │
    │  (现有 + 增强)     │   │  (新增模块)           │   │  (新增模块)        │
    └───────────────────┘   └──────────────────────┘   └───────────────────┘
```

### 2.3 Layer 1: 核心评估 (Core Evaluation)

保留现有 PDMS/EPDMS, 并增加以下增强:

#### 2.3.1 场景难度分层评估 (Stratified Scenario Evaluation)

**动机**: 当前评估仅报告全局平均分, 简单场景 (如直道匀速) 在数据集中占多数, 会稀释模型在困难场景中的表现差异.

**方案**:
- 基于预计算的 metric cache, 自动将场景划分为难度等级
- 分类维度:
  - **交通密度**: 周围 agent 数量 (从 `PDMObservation` 提取)
  - **路线复杂度**: 转弯角度、车道变换次数 (从 `centerline` 曲率计算)
  - **速度区间**: 初始 ego 速度 (从 `EgoState` 提取)
  - **环境类型**: 交叉口 / 直道 / 匝道 (从 `drivable_area_map` 的图层类型推断)

**输出**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Stratified PDMS Results                                         │
├──────────────┬────────┬────────┬────────┬────────┬─────────────┤
│ 难度等级      │ 场景数  │ PDMS   │ NC     │ DAC    │ EP          │
├──────────────┼────────┼────────┼────────┼────────┼─────────────┤
│ Easy         │ 2,340  │ 96.2   │ 0.99   │ 0.99   │ 0.92        │
│ Medium       │ 1,850  │ 91.5   │ 0.95   │ 0.97   │ 0.88        │
│ Hard         │ 810    │ 82.3   │ 0.88   │ 0.91   │ 0.79        │
│ Extreme      │ 200    │ 71.0   │ 0.75   │ 0.83   │ 0.68        │
│ ───────────  │ ────── │ ────── │ ────── │ ────── │ ──────────  │
│ Overall      │ 5,200  │ 93.1   │ 0.96   │ 0.97   │ 0.89        │
└──────────────┴────────┴────────┴────────┴────────┴─────────────┘
```

**实现路径**:
```python
class ScenarioDifficultyClassifier:
    """基于 metric cache 对场景进行难度分类."""

    def classify(self, metric_cache: MetricCache) -> str:
        traffic_density = len(metric_cache.observation.unique_objects)
        centerline_curvature = self._compute_curvature(metric_cache.centerline)
        ego_speed = np.hypot(
            metric_cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            metric_cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
        )
        is_intersection = self._check_intersection(metric_cache.drivable_area_map)

        score = (
            0.3 * min(traffic_density / 20.0, 1.0)
            + 0.3 * min(centerline_curvature / 0.1, 1.0)
            + 0.2 * min(ego_speed / 15.0, 1.0)
            + 0.2 * float(is_intersection)
        )

        if score < 0.25: return "easy"
        elif score < 0.50: return "medium"
        elif score < 0.75: return "hard"
        else: return "extreme"
```

#### 2.3.2 DDC 权重恢复选项

当前 PDMS 中 DDC 权重为 0 (即被忽略). 提供配置选项允许在非竞赛场景下启用 DDC 评估:

```yaml
# config/evaluation/full_evaluation.yaml
scorer:
  config:
    driving_direction_weight: 5.0  # 恢复 DDC 权重
```

---

### 2.4 Layer 2: 诊断指标 (Diagnostic Metrics)

#### 2.4.1 Proposal 质量分析 (Proposal Quality Analysis)

**动机**: DrivoR 生成 64 条 proposal 轨迹, 再由 learned scorer 选择最优. 当前评估仅衡量最终选择的质量, 无法区分是 "生成的 proposal 集合质量差" 还是 "scorer 选错了".

**指标定义**:

| 指标 | 定义 | 公式 |
|------|------|------|
| **Oracle Score** | 64条 proposal 中真实 PDM 最高分 | `max_k PDMS(proposal_k)` |
| **Coverage Rate** | proposal 覆盖的可行驾驶空间比例 | 以 GT 轨迹为参考, 计算至少有一条 proposal 的 L2 距离 < 阈值的比例 |
| **Selection Accuracy** | scorer 选中的 proposal 是否为真实最优 | `1[argmax(pred_score) == argmax(real_score)]` |
| **Score Gap** | 最终选择与 oracle 之间的分数差距 | `Oracle_Score - Selected_Score` |
| **Top-K Hit Rate** | 真实最优 proposal 是否在 scorer 预测的 Top-K 中 | `1[argmax(real_score) ∈ TopK(pred_score)]` |
| **Diversity** | proposal 集合的空间多样性 | `mean(min_j≠i ||traj_i - traj_j||)` |

**实现**: 在现有 `validation_step` 基础上扩展 (注意: 当前 `agent_lightning_module.py` 已有部分相关逻辑, 可直接复用):

```python
class ProposalQualityAnalyzer:
    """分析 proposal 集合质量与 scorer 校准."""

    def analyze(self, proposals, pred_scores, metric_cache, simulator, scorer):
        # 对所有 64 条 proposal 计算真实 PDM score
        real_scores = []
        for proposal in proposals:
            result = pdm_score(metric_cache, proposal, ...)
            real_scores.append(result.score)

        real_scores = np.array(real_scores)
        oracle_score = real_scores.max()
        selected_idx = pred_scores.argmax()
        selected_score = real_scores[selected_idx]

        return {
            "oracle_score": oracle_score,
            "selected_score": selected_score,
            "score_gap": oracle_score - selected_score,
            "selection_accuracy": int(selected_idx == real_scores.argmax()),
            "top5_hit_rate": int(real_scores.argmax() in pred_scores.argsort()[-5:]),
        }
```

#### 2.4.2 Scorer 校准分析 (Scorer Calibration)

**动机**: DrivoR 的 scorer 通过 6 个独立的 MLP 头预测各子指标的 logit, 这些 logit 经 sigmoid 后应近似真实的二值/连续子分数. 评估 scorer 的校准程度可以帮助诊断模型的薄弱环节.

**指标**:

| 指标 | 定义 |
|------|------|
| **Sub-score BCE** | 每个子指标的 Binary Cross-Entropy (pred logit vs. ground truth) |
| **Reliability Diagram** | 预测概率 vs. 实际频率的校准曲线 |
| **ECE (Expected Calibration Error)** | 校准误差的期望值 |
| **Per-metric Rank Correlation** | 预测 ranking 与真实 ranking 的 Spearman 相关系数 |

**实现**:
```python
class ScorerCalibrationAnalyzer:
    """评估 learned scorer 的校准程度."""

    SUB_METRICS = [
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "comfort",
        "driving_direction_compliance",
    ]

    def compute_calibration(self, pred_logits, gt_scores):
        results = {}
        for i, name in enumerate(self.SUB_METRICS):
            pred = torch.sigmoid(pred_logits[name]).cpu().numpy()
            gt = gt_scores[:, i].cpu().numpy()

            bce = -np.mean(gt * np.log(pred + 1e-7) + (1 - gt) * np.log(1 - pred + 1e-7))
            ece = self._expected_calibration_error(pred, gt, n_bins=10)
            rank_corr = spearmanr(pred, gt).correlation

            results[name] = {
                "bce": bce,
                "ece": ece,
                "rank_correlation": rank_corr,
            }
        return results

    @staticmethod
    def _expected_calibration_error(pred_probs, gt_labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (pred_probs >= bin_boundaries[i]) & (pred_probs < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            avg_confidence = pred_probs[mask].mean()
            avg_accuracy = gt_labels[mask].mean()
            ece += mask.sum() * abs(avg_confidence - avg_accuracy)
        return ece / len(pred_probs)
```

#### 2.4.3 时序一致性评估 (Temporal Consistency)

**动机**: 端到端模型可能在逐帧运行时产生时序不一致的轨迹, 表现为 "抖动" — 相邻帧的规划轨迹在重叠时域内存在较大偏差. 这在 PDMS 中完全不可见.

**指标**:

| 指标 | 定义 |
|------|------|
| **Trajectory Jitter** | 相邻帧规划轨迹在时间重叠区域的 L2 偏差 |
| **Heading Consistency** | 相邻帧预测航向角变化的标准差 |
| **Decision Stability** | 在连续 K 帧中 scorer 选择的 proposal 索引切换频率 |

**实现要点**:
- 需要修改数据加载以支持滑窗式连续帧采样 (当前 SceneLoader 独立采样每个 token)
- 在连续的场景 token 之间, 将前一帧的 4s 轨迹与后一帧的 3.5s (偏移 0.5s) 轨迹对齐, 计算重叠区间的 L2 距离

```python
class TemporalConsistencyEvaluator:
    """评估跨帧的规划一致性."""

    def evaluate_pair(self, traj_t, traj_t_plus_1, dt=0.5):
        """
        traj_t:       shape (8, 3), 时刻 t 的规划轨迹 (0.5s 间隔)
        traj_t_plus_1: shape (8, 3), 时刻 t+0.5s 的规划轨迹
        """
        # traj_t 的 pose[1:] 对应 traj_t_plus_1 的 pose[0:7] (在全局坐标下)
        overlap_t = traj_t[1:]  # 7 个点
        overlap_t1 = traj_t_plus_1[:7]  # 7 个点

        jitter = np.linalg.norm(overlap_t[:, :2] - overlap_t1[:, :2], axis=-1)
        heading_diff = np.abs(overlap_t[:, 2] - overlap_t1[:, 2])

        return {
            "mean_jitter": jitter.mean(),
            "max_jitter": jitter.max(),
            "heading_std": heading_diff.std(),
        }
```

#### 2.4.4 失败模式分类 (Failure Taxonomy)

**动机**: 当 PDMS < 阈值时, 需要快速定位失败原因, 而非仅知道分数低.

**分类体系**:

```
Failure Taxonomy
├── Safety Failures (NC=0 or NC=0.5)
│   ├── Rear-end collision (ego hits stopped/slow vehicle ahead)
│   ├── Lateral collision (lane change into occupied lane)
│   ├── Intersection collision (failing to yield)
│   └── Pedestrian/cyclist collision
├── Compliance Failures (DAC=0)
│   ├── Off-road departure
│   ├── Mounting curb
│   └── Entering non-drivable zone
├── Progress Failures (EP < 0.5)
│   ├── Unnecessary stop (no obstacle ahead)
│   ├── Excessive deceleration
│   └── Route deviation
├── Comfort Failures (C=0)
│   ├── Hard braking (longitudinal jerk)
│   ├── Sharp steering (lateral jerk)
│   └── Combined discomfort
└── Scorer Failures (scorer选错导致的性能损失)
    ├── Missed best proposal (score_gap > 0.1)
    └── Catastrophic selection (selected NC=0, oracle NC=1)
```

**实现**: 基于 PDMScorer 已有的内部状态 (`_multi_metrics`, `_weighted_metrics`, `_collision_time_idcs` 等) 构建:

```python
class FailureTaxonomy:
    """结构化的失败模式分类."""

    def classify(self, pdm_result, scorer_internals, oracle_score=None):
        failures = []

        if pdm_result.no_at_fault_collisions < 1.0:
            collision_type = self._identify_collision_type(scorer_internals)
            failures.append(("safety", collision_type))

        if pdm_result.drivable_area_compliance < 1.0:
            departure_type = self._identify_departure_type(scorer_internals)
            failures.append(("compliance", departure_type))

        if pdm_result.ego_progress < 0.5:
            progress_type = self._identify_progress_failure(scorer_internals)
            failures.append(("progress", progress_type))

        if pdm_result.comfort < 1.0:
            comfort_type = self._identify_comfort_failure(scorer_internals)
            failures.append(("comfort", comfort_type))

        if oracle_score and (oracle_score - pdm_result.score) > 0.1:
            failures.append(("scorer", "missed_best_proposal"))

        return failures
```

---

### 2.5 Layer 3: 系统级评估 (System-Level Evaluation)

#### 2.5.1 推理效率基准 (Inference Efficiency Benchmark)

**指标**:

| 指标 | 单位 | 说明 |
|------|------|------|
| **Latency (p50/p95/p99)** | ms | 单帧推理延迟分布 |
| **Throughput** | FPS | 每秒处理的场景数 |
| **GPU Memory** | GB | 峰值显存占用 |
| **FLOPs** | GFLOPs | 前向传播计算量 |

**实现**: 通过包装推理函数, 使用 `torch.cuda.Event` 精确测量:

```python
class LatencyProfiler:
    """GPU 推理延迟分析器."""

    def __init__(self, warmup_steps=50, measure_steps=200):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps

    def profile(self, model, dataloader, device):
        model.eval()
        latencies = []

        for i, batch in enumerate(dataloader):
            if i >= self.warmup_steps + self.measure_steps:
                break

            features = {k: v.to(device) for k, v in batch[0].items()
                       if isinstance(v, torch.Tensor)}

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                model(features)
            end.record()
            torch.cuda.synchronize()

            if i >= self.warmup_steps:
                latencies.append(start.elapsed_time(end))

        latencies = np.array(latencies)
        return {
            "latency_p50_ms": np.percentile(latencies, 50),
            "latency_p95_ms": np.percentile(latencies, 95),
            "latency_p99_ms": np.percentile(latencies, 99),
            "throughput_fps": 1000.0 / latencies.mean(),
            "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
```

#### 2.5.2 鲁棒性评估 (Robustness Evaluation)

**动机**: 真实部署中传感器输入存在噪声和退化. 评估模型对输入扰动的鲁棒性.

**扰动类型**:

| 扰动 | 方法 | 参数 |
|------|------|------|
| 相机遮挡 | 随机将一个相机图像置为全黑 | 遮挡率: 1/4 相机 |
| 高斯噪声 | 向图像添加高斯噪声 | σ ∈ {0.01, 0.05, 0.1} |
| 运动模糊 | 应用方向性模糊核 | 核大小: 15px |
| 时间戳偏移 | 使用延迟一帧的相机图像 | 延迟: 0.5s |
| Ego 状态噪声 | 向速度/加速度添加高斯噪声 | σ_v=0.5m/s, σ_a=0.3m/s² |

**输出**: 扰动条件下的 PDMS 退化曲线:

```
扰动类型          PDMS (clean)  PDMS (perturbed)  Δ PDMS
─────────────────────────────────────────────────────────
无                93.1          93.1               0.0
相机遮挡 (1/4)    93.1          91.8              -1.3
高斯噪声 σ=0.05  93.1          92.5              -0.6
运动模糊          93.1          90.7              -2.4
时间戳延迟 0.5s   93.1          88.3              -4.8
```

---

### 2.6 统一评估配置

所有评估维度通过统一 Hydra 配置管理:

```yaml
# navsim/planning/script/config/evaluation/full_evaluation.yaml
defaults:
  - default_common
  - default_evaluation
  - default_scoring_parameters
  - agent: drivoR
  - _self_

# === Layer 1: Core ===
core:
  enabled: true
  stratified: true  # 启用场景分层
  difficulty_bins: ["easy", "medium", "hard", "extreme"]
  include_ddc: false  # 是否启用 DDC (默认与 NAVSIM 一致)

# === Layer 2: Diagnostics ===
diagnostics:
  proposal_quality:
    enabled: true
    compute_oracle: true
    top_k_values: [1, 3, 5, 10]

  scorer_calibration:
    enabled: true
    n_bins: 15
    per_metric: true

  temporal_consistency:
    enabled: false  # 需要连续帧数据支持
    window_size: 5

  failure_taxonomy:
    enabled: true
    score_gap_threshold: 0.1

# === Layer 3: System ===
system:
  latency:
    enabled: true
    warmup_steps: 50
    measure_steps: 200

  robustness:
    enabled: false  # 可选, 计算密集
    perturbations: ["camera_occlusion", "gaussian_noise", "motion_blur"]
    noise_levels: [0.01, 0.05, 0.1]
```

---

## 3. 实现计划

### 3.1 新增文件结构

```
navsim/evaluate/
├── pdm_score.py                   # [已有] 核心 PDM Score 计算
├── full_evaluation.py             # [新增] 统一评估入口
├── stratified_evaluator.py        # [新增] 场景分层评估
├── proposal_quality.py            # [新增] Proposal 质量分析
├── scorer_calibration.py          # [新增] Scorer 校准分析
├── temporal_consistency.py        # [新增] 时序一致性评估
├── failure_taxonomy.py            # [新增] 失败模式分类
├── latency_profiler.py            # [新增] 推理延迟基准
├── robustness_evaluator.py        # [新增] 鲁棒性评估
└── report_generator.py            # [新增] 评估报告生成
```

### 3.2 新增脚本

```
navsim/planning/script/
├── run_full_evaluation.py         # [新增] 统一评估入口脚本
└── config/evaluation/
    └── full_evaluation.yaml       # [新增] 统一评估配置
```

### 3.3 开发优先级

| 优先级 | 模块 | 工作量 | 依赖 |
|--------|------|--------|------|
| P0 | 统一评估入口 + 报告生成 | 2 天 | 无 |
| P0 | 场景分层评估 | 1 天 | Metric Cache |
| P0 | Proposal 质量分析 | 1 天 | Agent 推理 |
| P1 | Scorer 校准分析 | 1 天 | Agent 推理 |
| P1 | 失败模式分类 | 2 天 | PDM Scorer 内部状态 |
| P1 | 推理效率基准 | 0.5 天 | Agent 推理 |
| P2 | 时序一致性评估 | 2 天 | 连续帧数据加载 |
| P2 | 鲁棒性评估 | 3 天 | Feature Builder 扰动 |

---

## 4. 运行指南

### 4.1 前置条件

假设环境变量已在另一个仓库中配置:
```bash
# 以下变量应已设置:
# NUPLAN_MAP_VERSION, NUPLAN_MAPS_ROOT, NAVSIM_EXP_ROOT,
# NAVSIM_DEVKIT_ROOT, OPENSCENE_DATA_ROOT
```

### 4.2 Step 1: Metric Cache (如尚未生成)

```bash
# 为 navtest 分割缓存评估所需的 metric cache
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

### 4.3 Step 2: 运行完整评估

```bash
CHECKPOINT=path/to/Nav1_25epochs.pth

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_full_evaluation.py \
    train_test_split=navtest \
    agent=drivoR \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.proposal_num=64 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.ref_num=4 \
    core.stratified=true \
    diagnostics.proposal_quality.enabled=true \
    diagnostics.scorer_calibration.enabled=true \
    diagnostics.failure_taxonomy.enabled=true \
    system.latency.enabled=true
```

### 4.4 Step 3: 查看评估报告

评估完成后, 报告生成在 `$NAVSIM_EXP_ROOT/evaluation_reports/` 目录下:

```
evaluation_reports/
├── 2026.03.31.14.30.00/
│   ├── summary.csv              # 总览: PDMS + 所有增强指标
│   ├── stratified_results.csv   # 分层评估详情
│   ├── proposal_quality.csv     # Proposal 质量分析
│   ├── scorer_calibration.json  # Scorer 校准结果
│   ├── failure_analysis.csv     # 失败模式统计
│   ├── latency_profile.json     # 推理延迟统计
│   └── report.html              # 可视化 HTML 报告
```

---

## 5. 评估报告样例

### 5.1 Summary Report

```
╔══════════════════════════════════════════════════════════════════╗
║                DrivoR Full Evaluation Report                    ║
║                Model: Nav1_25epochs.pth                         ║
║                Split: navtest (5,200 scenarios)                 ║
║                Date: 2026-03-31 14:30:00                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─ Core Metrics ─────────────────────────────────────────────┐ ║
║  │ PDMS (Overall):          93.1                               │ ║
║  │ NC:  0.963  DAC: 0.971  TTC: 0.945  EP: 0.892  C: 0.978   │ ║
║  └─────────────────────────────────────────────────────────────┘ ║
║                                                                  ║
║  ┌─ Stratified Results ───────────────────────────────────────┐ ║
║  │ Easy:     96.2 (2,340 scenes)                               │ ║
║  │ Medium:   91.5 (1,850 scenes)                               │ ║
║  │ Hard:     82.3 (810 scenes)                                 │ ║
║  │ Extreme:  71.0 (200 scenes)                                 │ ║
║  └─────────────────────────────────────────────────────────────┘ ║
║                                                                  ║
║  ┌─ Proposal Quality ────────────────────────────────────────┐  ║
║  │ Oracle Score:            95.8 (+2.7 above selected)        │  ║
║  │ Selection Accuracy:      34.2%                             │  ║
║  │ Top-5 Hit Rate:          78.6%                             │  ║
║  │ Proposal Diversity:      2.34m (mean min-distance)         │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─ Scorer Calibration ──────────────────────────────────────┐  ║
║  │ Overall ECE:             0.047                             │  ║
║  │ NC Rank Correlation:     0.82                              │  ║
║  │ DAC Rank Correlation:    0.79                              │  ║
║  │ EP Rank Correlation:     0.68                              │  ║
║  │ TTC Rank Correlation:    0.71                              │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─ Failure Analysis (PDMS < 50, N=127) ─────────────────────┐ ║
║  │ Safety failures:         48.0%  (61 scenes)                │  ║
║  │   └─ Intersection:       27.6%  (35 scenes)                │  ║
║  │   └─ Rear-end:           14.2%  (18 scenes)                │  ║
║  │   └─ Lateral:             6.3%  (8 scenes)                 │  ║
║  │ Compliance failures:     22.0%  (28 scenes)                │  ║
║  │ Progress failures:       18.9%  (24 scenes)                │  ║
║  │ Scorer failures:         11.0%  (14 scenes)                │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─ System Performance ──────────────────────────────────────┐  ║
║  │ Latency p50 / p95:       42ms / 58ms                       │  ║
║  │ Throughput:               23.8 FPS                         │  ║
║  │ GPU Memory:               4.2 GB                           │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 6. 与现有 Pipeline 的兼容性

| 方面 | 兼容性 |
|------|--------|
| Metric Cache 格式 | 完全兼容, 直接复用现有缓存 |
| Agent 接口 | 无需修改, 通过 `AbstractAgent` 接口对接 |
| Hydra 配置 | 扩展现有配置结构, 不修改已有 yaml |
| NAVSIM Leaderboard | PDMS/EPDMS 核心指标计算逻辑不变 |
| 多 GPU 推理 | 复用 `AgentLightningModule.predict_step` |
| 输出格式 | 保留 CSV 输出, 额外生成 JSON + HTML 报告 |

---

## 7. 未来扩展方向

1. **反应式仿真集成**: 对接 nuPlan 的闭环仿真器, 评估交互式场景
2. **多模型对比工具**: 在同一评估框架下同时评估多个模型, 生成对比报告
3. **在线评估看板**: 基于 nuBoard 或 Streamlit 构建实时评估可视化
4. **置信度标定**: 利用 scorer 的 sigmoid 输出构建校准的不确定性估计
5. **长尾场景挖掘**: 基于失败分析自动识别训练数据中缺失的场景类型
