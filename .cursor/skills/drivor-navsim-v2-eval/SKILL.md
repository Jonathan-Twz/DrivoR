---
name: drivor-navsim-v2-eval
description: >-
  Set up and run NAVSIM v2 (navhard_two_stage) EPDMS evaluation for DrivoR,
  including metric cache, BEV feature paths, navsim repo agent patches,
  decoder-BEV vs scorer-BEV scripts, and Hydra overrides. Use when evaluating
  Nav2 checkpoints, BEV scorer/decoder on v2, or debugging v2 eval setup.
disable-model-invocation: true
---

# DrivoR on NAVSIM v2 (navhard_two_stage)

## When To Use

- User asks for **NAVSIM v2 / EPDMS / navhard_two_stage** evaluation.
- Setting up v2 benchmark after cloning `autonomousvision/navsim`.
- Evaluating **Nav2**, **BEV scorer**, or **BEV decoder** checkpoint on v2.
- Debugging: Hydra `OverrideParseException`, missing BEV files, `scene_aggregator` import, `StateIndex` IndexError, eval killed mid-run.

## Critical: v2 Uses the Official `navsim` Repo, Not DrivoR Alone

| Task | Repo / entry |
|------|----------------|
| NAVSIM **v1** PDMS (navtest) | `DrivoR/scripts/evaluation/run_drivor_bev_*_evaluation.sh` → `run_pdm_score_multi_gpu.py` |
| NAVSIM **v2** EPDMS (navhard_two_stage) | **`/mnt/ws-frb/users/jingyuso/wenzhet/navsim`** → `run_pdm_score.py` |

Copy/sync **only** needed agent files into the official navsim tree. Do not replace the whole DrivoR repo with navsim.

## Paths (wenzhet layout)

| Item | Path |
|------|------|
| Official navsim devkit | `/mnt/ws-frb/users/jingyuso/wenzhet/navsim` |
| v2 data | `navsim_dataset/navhard_two_stage/` |
| v2 metric cache | `navsim/exp/navhard_two_stage_metric_cache` |
| v1 BEV exports (navtest / stage-1 fallback) | `navsim_bev_feature/exports_pretrained/test/` |
| v2 BEV exports | `navsim_bev_feature/exports_pretrained_navsim_v2/navhard_two_stage/` |
| v2 BEV root + stage-1 fallback | `exports_pretrained_navsim_v2/test` → symlink to `../exports_pretrained/test` |
| **Conda env (train + eval)** | **`drivoR-share`** (Python 3.9) |
| Weights symlink in navsim | `navsim/weights` → `DrivoR/weights` |

## Environment

Use **`drivoR-share`** for both training and v2 evaluation (Python 3.9).

```bash
conda activate drivoR-share
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim && source setup_env.sh
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"   # v2 scripts set this automatically
```

**Why py3.9**: py3.8 breaks `nuplan-devkit` `StateIndex.STATE_SE2` (`@classmethod @property` returns bound method, not slice) → `IndexError` in v2 scoring. `drivoR-share` is now py3.9.

**PYTHONPATH shadowing**: `drivoR-share` egg-link registers DrivoR's older `navsim` package (lacks `scene_aggregator`). v2 scripts must prepend the official navsim repo:

```bash
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT:-/mnt/ws-frb/users/jingyuso/wenzhet/navsim}:${PYTHONPATH:-}"
```

Verify:

```bash
conda activate drivoR-share
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim && source setup_env.sh
PYTHONPATH=$NAVSIM_DEVKIT_ROOT python -c "
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
assert type(StateIndex.STATE_SE2) is slice
from navsim.planning.simulation.planner.pdm_planner.scoring import scene_aggregator
print('OK')
"
```

## Metric Cache (one-time)

```bash
conda activate drivoR-share
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim && source setup_env.sh
bash scripts/evaluation/run_metric_caching_navhard.sh
# → navsim/exp/navhard_two_stage_metric_cache
```

## Agent Patches in `navsim` Copy

Required for all BEV v2 evals:

- `drivor_agent.py`: `requires_scene`, `trajectory_sampling`, `compute_trajectory(..., scene=...)`.
- `drivor_features.py`: BEV path fallback `navhard_two_stage/...` → `test/...` for stage-one frames.

**Decoder-BEV additionally needs** (sync from DrivoR repo):

- `layers/bev_decoder_blocks.py` (new)
- `drivor_model.py`: `use_bev_in_scorer`, `use_bev_in_decoder`, `BevAwareTrajectoryDecoder`
- `drivoR.yaml`: `decoder_bev: {init_gate, lora_rank, lora_dropout}`
- `drivor_agent.py`: whitelist for `trajectory_decoder.layers.*.cross_attn_bev*` and `*_lora.*`

## Eval Scripts

| Model | Script |
|-------|--------|
| Nav2 baseline | `navsim/scripts/evaluation/run_drivoR_pdm_score_v2.sh` |
| Scorer-BEV | same + `use_bev_in_scorer=true`, `scorer_bev.lora_rank=16` |
| **Decoder-BEV** | `navsim/scripts/evaluation/run_drivoR_pdm_score_v2_decoder_bev.sh` |

Decoder-BEV defaults: `use_bev_in_scorer=false`, `use_bev_in_decoder=true`, `decoder_bev.lora_rank=16`.

Reference ckpt: `DrivoR/exp/ke/Jun12-golduck-4gpu-lora16-bev-decoder/06.12_01.12/checkpoints/last.ckpt`

## Hydra Overrides (always for v2 eval)

| Override | Why |
|----------|-----|
| `agent.loss=null` | Avoid loading `train_metric_cache` in agent `__init__` |
| `agent.scheduler_args.num_epochs=1` | Breaks `${trainer.params.max_epochs}` interpolation |
| `agent.batch_size=1` | Breaks `${dataloader.params.batch_size}` interpolation |
| `"agent.checkpoint_path='...last.ckpt'"` | Quote path — unquoted `=` breaks Hydra |
| `worker=sequential` | Ray CPU workers can't run GPU-heavy DrivoR inference |
| v2 metric weights | `noc=10 dac=13 ddc=6 ttc=14 ep=15 comfort=2` |

## Run Full v2 Eval (Decoder-BEV)

```bash
conda activate drivoR-share
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim && source setup_env.sh
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0   # sequential uses cuda:0 only

# Long run: detach from terminal session
LOG=exp/v2_full_decoder_bev_$(date +%m%d_%H%M).log
setsid nohup bash scripts/evaluation/run_drivoR_pdm_score_v2_decoder_bev.sh \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
```

Monitor: `tail -f "$LOG"` or Hydra log under `navsim/exp/drivoR_nav2_decoder_bev_lora16_last/<timestamp>/run_pdm_score.log`.

## v1 vs v2 / stage1 vs stage2

| Term | Meaning |
|------|---------|
| **v1 / PDMS** | Open-loop, `navtest`, single-frame planning score |
| **v2 / EPDMS** | Closed-loop reactive, `navhard_two_stage`, harder benchmark |
| **stage_one** | Score on original / initial-frame scenarios |
| **stage_two** | Score on synthetic reactive follow-up scenarios (~5462 sub-scenarios) |
| **combined** | Official headline EPDMS (pseudo closed-loop weighted) |

One v2 run produces all three aggregate rows in the CSV.

## Reference Results (Jun 2026, decoder-BEV LoRA16 last.ckpt)

| Metric | Score |
|--------|-------|
| v2 stage_one EPDMS | 0.8413 |
| v2 stage_two EPDMS | 0.5850 |
| v2 combined EPDMS | **0.4966** |
| Nav2 baseline combined | ≈ 0.483 |

Bottleneck: stage_two `lane_keeping` ≈ 0.519.

CSV: `navsim/exp/drivoR_nav2_decoder_bev_lora16_last/2026.06.15.00.29.11/2026.06.15.03.14.05.csv`

## Read Results

```python
import pandas as pd
df = pd.read_csv("navsim/exp/<experiment>/<timestamp>.csv")
summ = df[df["token"].astype(str).str.startswith("extended_pdm_score")]
print(summ[["token", "score"]])
# extended_pdm_score_stage_one / stage_two / combined
```

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `ModuleNotFoundError: scene_aggregator` | PYTHONPATH not set; DrivoR egg-link shadows navsim repo |
| `IndexError` in `pdm_array_representation` | Python < 3.9; upgrade `drivoR-share` to py3.9 |
| `Missing token in score_df` on tiny `max_scenes` | Aggregation needs full token chain; not a model bug |
| Process dies mid-run, no traceback | Cursor terminal session killed; use `setsid nohup` |
| LoRA size mismatch | `decoder_bev.lora_rank` or `scorer_bev.lora_rank` doesn't match ckpt |
| v1 script can't find `run_pdm_score_multi_gpu.py` | `NAVSIM_DEVKIT_ROOT` still points to navsim; reset to DrivoR |

## Do Not

- Run v2 eval from DrivoR-only tree without navsim clone and patches.
- Use `max_scenes` subset for official benchmark score.
- Point `bev_features_root` at `exports_pretrained` only — v2 synthetic features are under `exports_pretrained_navsim_v2`.
