---
name: drivor-navsim-v2-eval
description: >-
  Set up and run NAVSIM v2 (navhard_two_stage) EPDMS evaluation for DrivoR,
  including metric cache, BEV feature paths, navsim repo agent patches, and Hydra
  overrides. Use when evaluating Nav2 checkpoints, BEV scorer on v2, or debugging
  v2 eval setup.
disable-model-invocation: true
---

# DrivoR on NAVSIM v2 (navhard_two_stage)

## When To Use

- User asks for **NAVSIM v2 / EPDMS / navhard_two_stage** evaluation.
- Setting up v2 benchmark after cloning `autonomousvision/navsim`.
- Evaluating **Nav2** checkpoint (no BEV) or **BEV scorer** checkpoint on v2.
- Debugging: Hydra `OverrideParseException` on checkpoint path, missing BEV files, eval stuck before `Processing stage one`.

## Critical: v2 Uses the Official `navsim` Repo, Not DrivoR Alone

| Task | Repo / entry |
|------|----------------|
| NAVSIM **v1** PDMS (navtest) | `DrivoR/scripts/evaluation/run_drivor_bev_evaluation.sh` → `run_pdm_score_multi_gpu.py` |
| NAVSIM **v2** EPDMS (navhard_two_stage) | **`/mnt/ws-frb/users/jingyuso/wenzhet/navsim`** → `run_pdm_score.py` |

Copy **only** `navsim/agents/drivoR/` + `navsim/planning/script/config/common/agent/drivoR.yaml` into the official navsim tree. Do not replace the whole DrivoR repo with navsim.

## Paths (wenzhet layout)

| Item | Path |
|------|------|
| Official navsim devkit | `/mnt/ws-frb/users/jingyuso/wenzhet/navsim` |
| v2 data | `navsim_dataset/navhard_two_stage/` (`sensor_blobs`, `synthetic_scene_pickles`, `openscene_meta_datas`) |
| v2 metric cache | `navsim/exp/navhard_two_stage_metric_cache` |
| v1 BEV exports (navtest) | `navsim_bev_feature/exports_pretrained/test/` |
| v2 BEV exports | `navsim_bev_feature/exports_pretrained_navsim_v2/navhard_two_stage/` |
| v2 BEV root + stage-1 fallback | `exports_pretrained_navsim_v2/test` → symlink to `../exports_pretrained/test` |
| Eval conda env | `navsim` (`python 3.9`, torch 2.0.1) |
| Weights symlink in navsim | `navsim/weights` → `DrivoR/weights` |

## One-Time Environment

```bash
# Clone (sibling of DrivoR)
cd /mnt/ws-frb/users/jingyuso/wenzhet
git clone https://github.com/autonomousvision/navsim.git

# Conda + editable install
conda activate navsim   # create from navsim/environment.yml if missing
cd navsim && pip install -e . && pip install einops

# Env vars
source /mnt/ws-frb/users/jingyuso/wenzhet/navsim/setup_env.sh

# Symlink weights + v1 test BEV for stage-one fallback
ln -sfn /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/weights /mnt/ws-frb/users/jingyuso/wenzhet/navsim/weights
ln -sfn /mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained/test \
  /mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained_navsim_v2/test
```

## Metric Cache (one-time)

```bash
conda activate navsim
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim
source setup_env.sh
bash scripts/evaluation/run_metric_caching_navhard.sh
# → navsim/exp/navhard_two_stage_metric_cache (77 log dirs, exit 0)
```

## Agent Patches in `navsim` Copy (required for BEV on v2)

Official `AbstractAgent` requires `trajectory_sampling`; DrivoR fork did not pass it. Evaluator calls `compute_trajectory(agent_input, scene)` when `requires_scene=True`.

**`navsim/agents/drivoR/drivor_agent.py`**

- `super().__init__(trajectory_sampling=config.trajectory_sampling, requires_scene=bool(config.get("use_bev_feature", False))`
- `compute_trajectory(self, agent_input, scene=None)` passes `scene.scene_metadata.initial_token` and `log_name` into `DrivoRFeatureBuilder.compute_features`.

**`navsim/agents/drivoR/drivor_features.py`**

- When `bev_data_split=navhard_two_stage`, try `navhard_two_stage/{log}/{token}_decoder_neck.pt` first, then fallback `test/{log}/...` for stage-one original frames.

## Verify BEV Features Before BEV Eval

```bash
# Manifest: bevfusion/logs/manifest_navsim_v2_navhard_two_stage.json
# Features: exports_pretrained_navsim_v2/navhard_two_stage/{log}/{token}_decoder_neck.pt
# Expect shape (256, 128, 128) float32
```

Checked pattern: manifest token paths resolve to existing `.pt` files; random samples finite and correct shape.

## Hydra Overrides (always for v2 eval)

| Override | Why |
|----------|-----|
| `agent.loss=null` | Avoid loading `train_metric_cache` in agent `__init__` (eval-only) |
| `agent.scheduler_args.num_epochs=1` | Breaks `${trainer.params.max_epochs}` interpolation |
| `agent.batch_size=1` | Breaks `${dataloader.params.batch_size}` interpolation |
| `"agent.checkpoint_path='...best-epoch=12-step=17277.ckpt'"` | **Quote** path — unquoted `=` in filename breaks Hydra |
| `worker=sequential` | Default `ray_distributed_no_torch` uses CPU Ray; DrivoR needs GPU in `compute_trajectory` |
| v2 metric weights | `noc=10 dac=13 ddc=6 ttc=14 ep=15 comfort=2` (not v1 navtest weights) |
| `+trainer.params.inference_mode=false` | Not needed for `run_pdm_score.py` (unlike v1 `run_pdm_score_multi_gpu.py`) |

## Baseline Nav2 Eval (no BEV)

```bash
conda activate navsim
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim
source setup_env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,4   # golduck: A100 only; skip DGX Display index 3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CHECKPOINT=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/weights/checkpoints/drivor_Nav2_10epochs.pth \
EXPERIMENT=drivoR_nav2_full \
bash scripts/evaluation/run_drivoR_pdm_score_v2.sh
```

**Reference result** (5912 scenarios, sequential, GPU): **EPDMS combined ≈ 0.483** (matches README Nav2 ~48.3). CSV under `navsim/exp/drivoR_nav2_full/<timestamp>/`.

## BEV Scorer Eval on v2

Same as baseline, plus:

```bash
agent.config.use_bev_feature=true \
agent.config.bev_feature_type=decoder_neck \
agent.config.bev_channels=256 \
agent.config.bev_features_root=/mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained_navsim_v2 \
agent.config.bev_data_split=navhard_two_stage \
agent.config.scorer_bev.lora_rank=16 \
agent.config.long_trajectory_additional_poses=2
```

Default BEV scorer ckpt (user workspace):

`/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/ke/golduck-4gpu-16batch-8worker-01gate-16lora/05.28_23.15/checkpoints/best-epoch=12-step=17277.ckpt`

Launcher template: `navsim/_run_full_bev_nav2.sh` (background: `nohup bash _run_full_bev_nav2.sh > _full_bev_nav2.log 2>&1 &`).

## v1 vs v2 Split Choice

| Benchmark | Split | BEV `bev_data_split` | Eval script location |
|-----------|-------|----------------------|---------------------|
| NAVSIM v1 PDMS | `navtest` | `test` | DrivoR `run_drivor_bev_evaluation.sh` |
| NAVSIM v2 EPDMS | `navhard_two_stage` | `navhard_two_stage` (+ `test` fallback) | navsim `run_drivoR_pdm_score_v2.sh` |

Precomputed BEV for v2 lives under `exports_pretrained_navsim_v2/`, **not** `exports_pretrained/`.

## GPU Notes

- **golduck**: `CUDA_VISIBLE_DEVICES=0,1,2,4`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` (nvidia-smi index 3 = DGX Display, 4 GB).
- **guppy**: `CUDA_VISIBLE_DEVICES=1` (single A6000 for sequential eval).
- Sequential v2 full run ~1.5–2 h for 5912 scenarios (Nav2 reference); multi-GPU `run_pdm_score.py` uses Ray CPU workers by default — poor for GPU-heavy DrivoR.

## Smoke Test (2 scenes)

```bash
train_test_split.scene_filter.max_scenes=2 worker=sequential ...
```

Inference can pass (1 stage-one + 10 stage-two tokens), but **final aggregation may fail** on tiny subsets (`Missing token in score_df`) — use full split for official score.

## Read Results

```python
import pandas as pd
df = pd.read_csv("navsim/exp/<experiment>/<timestamp>.csv")
summ = df[df["token"].astype(str).str.startswith("extended_pdm_score")]
print(summ[["token","valid","score"]])
# extended_pdm_score_combined → headline EPDMS
```

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `OverrideParseException` on checkpoint | Unquoted `agent.checkpoint_path=...epoch=12...` |
| `FileNotFoundError: train_metric_cache` | Missing `agent.loss=null` |
| `TypeError: trajectory_sampling` missing | Agent not patched in navsim copy |
| BEV always zero | `compute_trajectory` not passing scene; or wrong `bev_features_root` |
| Stuck after "Starting pdm scoring" | Sequential worker building SceneLoader + first heavy sample; check CPU/GPU with `ps`, log mtime |
| LoRA size mismatch | v1 eval script issue — use matching `scorer_bev.lora_rank` |

## Do Not

- Run v2 eval from DrivoR-only tree without navsim clone and patches.
- Point `bev_features_root` at `exports_pretrained` only — v2 synthetic features are under `exports_pretrained_navsim_v2`.
- Use `max_scenes=1` for final benchmark score.
