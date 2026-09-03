---
name: drivor-training-testing-pipeline
description: >-
  Run and reproduce DrivoR baseline, BEV decoder/scorer LoRA fine-tuning,
  BEV residual proposal refiner training, W&B smoke checks, checkpoint
  selection, NAVSIM v1 PDMS evaluation, and NAVSIM v2 EPDMS evaluation in the
  wenzhet workspace. Use when launching DrivoR training, evaluating checkpoints,
  comparing finetune vs baseline, or handing the pipeline to another agent.
disable-model-invocation: true
---

# DrivoR Training And Testing Pipeline

## Workspace Contract

Run from this layout:

| Item | Path |
|---|---|
| Workspace | `/mnt/ws-frb/users/jingyuso/wenzhet` |
| DrivoR repo | `/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR` |
| Official NAVSIM v2 repo | `/mnt/ws-frb/users/jingyuso/wenzhet/navsim` |
| Dataset root | `/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset` |
| BEV features v1 | `/mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained` |
| BEV features v2 | `/mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained_navsim_v2` |
| Pretrained baseline | `DrivoR/weights/checkpoints/drivor_Nav1_25epochs.pth` |
| Conda env | `/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share` |

Use Python from `drivoR-share` for training and both v1/v2 evaluation. On module-based hosts, this setup is safe:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
conda activate drivoR-share
module load Ninja/1.11.1-GCCcore-12.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load GCC/12.2.0

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset/maps
export OPENSCENE_DATA_ROOT=/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset
export NAVSIM_EXP_ROOT=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp
export NAVSIM_DEVKIT_ROOT=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
```

On golduck, `CUDA_VISIBLE_DEVICES=0,1,2,3` maps to the four A100 compute GPUs under CUDA FASTEST_FIRST. If physical `nvidia-smi` indices are required instead, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` first and then use `CUDA_VISIBLE_DEVICES=0,1,2,4`. Never use `0,1,2,4` under the default ordering: the fourth logical device becomes the 4 GB display GPU.

## Core Training Launchers

All run logs go under `DrivoR/exp/ke/<experiment>/<uid>/launcher.log`; checkpoints go under `.../checkpoints/`.

### W&B Run Naming Convention

Use W&B run names in the form `<experiment>/<MM.DD_HH.MM>` by setting the launcher `experiment_name` argument and `EXPERIMENT_UID="$(date +%m.%d_%H.%M)"`. Use a date-prefixed experiment name plus the key intervention, for example `Jul14-future-bev-decoder-scorer-nocache/07.14_16.00`.

If a run differs from the matched comparison settings, note the difference in the experiment name, e.g. `batch8`, `2gpu`, `cache`, `nocache`, `workers4`, or `gate0.1`. For matched scorer-decoder co-tune settings, omit redundant hardware/batch suffixes.

### Baseline, No BEV

Matches the NAVSIM-v1 baseline hyperparameters and trains from scratch with no BEV feature path:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=drivor-baseline \
EXPERIMENT_UID="$(date +%m.%d_%H.%M)" \
bash scripts/training/run_drivor_nav1_baseline.sh \
  Jun26-drivor-nav1-baseline-no-bev-25epochs \
  25
```

Defaults: 4 GPUs, batch 16, workers 16, LR `0.0002`, AdamW, seed 2, `long_trajectory_additional_poses=2`.

### Pretrained BEV Fine-Tune, Generic Launcher

Use `scripts/training/run_drivor_bev_phase1.sh` for frozen-pretrained BEV fine-tuning. It loads `agent.checkpoint_path`, freezes original DrivoR, and trains only enabled BEV modules. Default baseline checkpoint is `weights/checkpoints/drivor_Nav1_25epochs.pth`.

Common defaults: 4 GPUs, batch 16, workers 8, cached data `exp/navsim_cache_nommcv_full`, LR `1e-4`, `ddp_find_unused_parameters_true`, W&B online, token filter from BEV feature files.

Decoder-only LoRA fine-tune:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_BEV_IN_SCORER=false \
USE_BEV_IN_DECODER=true \
USE_BEV_RESIDUAL_PROPOSAL_REFINER=false \
DECODER_BEV_INIT_GATE=0.0 \
DECODER_BEV_LORA_RANK=16 \
WANDB_PROJECT=drivor-bev-decoder \
EXPERIMENT_UID="$(date +%m.%d_%H.%M)" \
setsid nohup bash scripts/training/run_drivor_bev_phase1.sh \
  weights/checkpoints/drivor_Nav1_25epochs.pth \
  Jun16-golduck-4gpu-lora16-0initgate-bev-decoder \
  30 \
  > exp/launch_bev_decoder_$(date +%m%d_%H%M).log 2>&1 &
```

Scorer-only LoRA fine-tune:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_BEV_IN_SCORER=true \
USE_BEV_IN_DECODER=false \
USE_BEV_RESIDUAL_PROPOSAL_REFINER=false \
SCORER_BEV_INIT_GATE=0.0 \
SCORER_BEV_LORA_RANK=16 \
WANDB_PROJECT=drivor-bev-scorer \
EXPERIMENT_UID="$(date +%m.%d_%H.%M)" \
setsid nohup bash scripts/training/run_drivor_bev_phase1.sh \
  weights/checkpoints/drivor_Nav1_25epochs.pth \
  golduck-4gpu-16batch-8worker-0initgate-16lora \
  30 \
  > exp/launch_bev_scorer_$(date +%m%d_%H%M).log 2>&1 &
```

Decoder + scorer LoRA fine-tune:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_BEV_IN_SCORER=true \
USE_BEV_IN_DECODER=true \
USE_BEV_RESIDUAL_PROPOSAL_REFINER=false \
SCORER_BEV_INIT_GATE=0.0 \
SCORER_BEV_LORA_RANK=16 \
DECODER_BEV_INIT_GATE=0.0 \
DECODER_BEV_LORA_RANK=16 \
WANDB_PROJECT=drivor-bev-decoder-scorer \
EXPERIMENT_UID="$(date +%m.%d_%H.%M)" \
setsid nohup bash scripts/training/run_drivor_bev_phase1.sh \
  weights/checkpoints/drivor_Nav1_25epochs.pth \
  Jul08-golduck-4gpu-lora16-0initgate-bev-decoder-scorer-finetune-30epochs \
  30 \
  > exp/launch_bev_decoder_scorer_$(date +%m%d_%H%M).log 2>&1 &
```

### BEV Residual Proposal Refiner

Wrapper for post-decoder/pre-scorer residual proposal refinement. Original DrivoR is frozen; trainable modules are `bev_tokenizer.*` and `bev_residual_proposal_refiner.*`.

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RESIDUAL_REFINER_INIT_ALPHA=0.0 \
RESIDUAL_REFINER_NUM_LAYERS=1 \
WANDB_PROJECT=drivor-bev-residual-proposal-refiner \
EXPERIMENT_UID="$(date +%m.%d_%H.%M)" \
setsid nohup bash scripts/training/run_drivor_bev_residual_proposal_refiner.sh \
  weights/checkpoints/drivor_Nav1_25epochs.pth \
  Jun22-golduck-4gpu-0initalpha-bev-residual-proposal-refiner \
  30 \
  > exp/launch_bev_residual_$(date +%m%d_%H%M).log 2>&1 &
```

### BEV Decoder From Scratch

No pretrained checkpoint and no freeze. This launcher matches baseline hyperparameters while enabling decoder BEV.

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DECODER_BEV_INIT_GATE=0.0 \
DECODER_BEV_LORA_RANK=8 \
WANDB_PROJECT=drivor-bev-decoder \
EXPERIMENT_UID="$(date +%m.%d_%H.%M)" \
setsid nohup bash scripts/training/run_drivor_bev_decoder_from_scratch.sh \
  Jul05-golduck-4gpu-lora8-0initgate-bev-decoder-from-scratch-20epochs \
  20 \
  > exp/launch_bev_decoder_scratch_$(date +%m%d_%H%M).log 2>&1 &
```

Set the second arg to `30` for 30 epochs. The launcher defaults to LoRA rank 8 and `DRIVOR_EPOCH_PDMS_EVAL=0`; do not re-enable epoch-end PDMS unless explicitly requested.

## W&B Smoke Before New Training Changes

For changed training code, run a tiny real-data online W&B smoke before a full job:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
USE_WANDB=1 \
WANDB_MODE=online \
WANDB_CONSOLE=wrap \
NUM_GPUS=1 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
PREFETCH_FACTOR=1 \
LIMIT_TRAIN_BATCHES=8 \
LIMIT_VAL_BATCHES=3 \
CACHE_PATH="$PWD/exp/navsim_cache_nommcv_full" \
USE_CACHE_WITHOUT_DATASET=true \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/training/run_drivor_bev_phase1.sh \
  weights/checkpoints/drivor_Nav1_25epochs.pth \
  debug-wandb-bev-smoke \
  2
```

Expected W&B metrics: `train/loss_step`, `train/loss_epoch`, `train/trajectory_loss`, `val/score_epoch`. See `drivor-wandb-smoke` for deeper logging debugging.

## Monitor Training

```bash
tail -f DrivoR/exp/ke/<experiment>/<uid>/launcher.log
tail -f DrivoR/exp/ke/<experiment>/<uid>/run_training_full.log
find DrivoR/exp/ke/<experiment>/<uid>/checkpoints -maxdepth 1 -type f -name '*.ckpt' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

Checkpoint policy is in `navsim/agents/drivoR/drivor_agent.py`: `save_top_k=5`, monitor `val/score_epoch`, mode `max`, plus `last.ckpt`. Select best checkpoint by reading the checkpoint callback metadata when possible, or by filename/metadata:

```bash
python - <<'PY'
from pathlib import Path
import torch
d = Path("DrivoR/exp/ke/<experiment>/<uid>/checkpoints")
for p in sorted(d.glob("*.ckpt")):
    ck = torch.load(p, map_location="cpu")
    cb = ck.get("callbacks", {})
    best = [(k, v) for k, v in cb.items() if "ModelCheckpoint" in k and isinstance(v, dict)]
    score = None
    for _, v in best:
        if "best_model_path" in v:
            score = v.get("best_model_score")
            print("best_model_path", v.get("best_model_path"), "best_model_score", score)
            raise SystemExit
print("No callback best metadata found; inspect checkpoint filenames and val/score_epoch in W&B.")
PY
```

## NAVSIM v1 PDMS Evaluation

Run v1 from the DrivoR tree with `NAVSIM_DEVKIT_ROOT=DrivoR`. It uses navtest and writes a CSV under `DrivoR/exp/ke/<experiment_name>/<timestamp>/`.

Baseline:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python navsim/planning/script/run_pdm_score_multi_gpu.py \
  train_test_split=navtest \
  agent=drivoR \
  agent.checkpoint_path=weights/checkpoints/drivor_Nav1_25epochs.pth \
  experiment_name=drivoR_nav1_baseline_pretrained \
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
  +trainer.params.inference_mode=false
```

`scripts/evaluation/run_drivor_evaluation.sh` is convenient for the default pretrained baseline, but it currently overwrites `CKPT_PATH` and `experiment_name`; use the direct command above when the checkpoint or eval name matters.

Decoder-only or decoder+scorer BEV:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
PYTHONPATH=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR:/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/nuplan-devkit \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CKPT_PATH=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/ke/<experiment>/<uid>/checkpoints/<best>.ckpt \
EXPERIMENT_NAME=drivoR_nav1_<short_run_name> \
DECODER_BEV_LORA_RANK=16 \
bash scripts/evaluation/run_drivor_bev_decoder_evaluation.sh \
  agent.config.use_bev_in_scorer=false \
  agent.config.scorer_bev.lora_rank=16 \
  agent.config.scorer_bev.init_gate=0.0 \
  agent.config.decoder_bev.init_gate=0.0
```

For decoder+scorer checkpoints, change `agent.config.use_bev_in_scorer=true` and keep both LoRA ranks at the training values.

Proposal-conditioned world refiner:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR-idea0002-01
CKPT_PATH=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/ke/<experiment>/<uid>/checkpoints/<best>.ckpt \
EXPERIMENT_NAME=drivoR_nav1_<short_run_name> \
bash scripts/evaluation/run_drivor_proposal_world_evaluation.sh
```

This launcher uses `CUDA_DEVICE_ORDER=PCI_BUS_ID` with physical A100 indices
`0,1,2,4`, local DINO weights, `bev_data_split=test`, and the architecture
defaults used by idea 0002-01: 2 Transformer layers, 4 heads, FFN 512, one
rollout step, and proposal chunk size 8. Checkpoint state-dict loading should
report zero missing and zero unexpected keys.

Scorer-only BEV:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python navsim/planning/script/run_pdm_score_multi_gpu.py \
  train_test_split=navtest \
  agent=drivoR \
  "agent.checkpoint_path='/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/ke/<experiment>/<uid>/checkpoints/<best>.ckpt'" \
  "experiment_name='drivoR_nav1_<short_run_name>'" \
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
  agent.config.use_bev_feature=true \
  agent.config.use_bev_in_scorer=true \
  agent.config.use_bev_in_decoder=false \
  agent.config.use_ray_score=true \
  agent.config.bev_feature_type=decoder_neck \
  agent.config.bev_channels=256 \
  agent.config.scorer_bev.lora_rank=16 \
  agent.config.bev_features_root=/mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained \
  agent.config.bev_data_split=test \
  agent.config.long_trajectory_additional_poses=2 \
  +trainer.params.inference_mode=false
```

`scripts/evaluation/run_drivor_bev_evaluation.sh` currently hard-codes a historical scorer checkpoint and eval name; do not rely on environment `CKPT_PATH` with that script unless it has been updated.

Residual proposal refiner:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CKPT_PATH=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/ke/<experiment>/<uid>/checkpoints/<best>.ckpt \
EXPERIMENT_NAME=drivoR_nav1_bev_residual_<short_run_name> \
bash scripts/evaluation/run_drivor_bev_residual_proposal_refiner_evaluation.sh
```

Parse v1 CSV:

```bash
python - <<'PY'
import csv
p = "DrivoR/exp/ke/<eval_experiment>/<timestamp>/<results>.csv"
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
avg = rows[-1]
print(avg["token"], avg["score"])
for k, v in avg.items():
    if k and k not in {"token", "valid"}:
        print(k, v)
PY
```

## NAVSIM v2 EPDMS Evaluation

Run v2 from the official `navsim/` checkout, not from DrivoR. v2 uses `navhard_two_stage`, `worker=sequential`, one GPU effectively, and produces `extended_pdm_score_stage_one`, `extended_pdm_score_stage_two`, and `extended_pdm_score_combined` in one CSV. Full v2 typically takes about 2.5 hours for 5912 scenarios.

Setup:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim
conda activate drivoR-share
source setup_env.sh
export PATH=/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin:$PATH
export DRIVOR_ROOT=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
export PYTHONPATH=/mnt/ws-frb/users/jingyuso/wenzhet/navsim:/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/nuplan-devkit:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

Metric cache, one-time:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim
source setup_env.sh
bash scripts/evaluation/run_metric_caching_navhard.sh
```

Decoder-only or decoder+scorer BEV:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim
source setup_env.sh
export PATH=/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin:$PATH
export PYTHONPATH=/mnt/ws-frb/users/jingyuso/wenzhet/navsim:/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/nuplan-devkit:${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CHECKPOINT=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/ke/<experiment>/<uid>/checkpoints/<best>.ckpt \
EXPERIMENT=drivoR_nav2_<short_run_name> \
DECODER_BEV_LORA_RANK=16 \
setsid nohup bash scripts/evaluation/run_drivoR_pdm_score_v2_decoder_bev.sh \
  agent.config.use_bev_in_scorer=false \
  agent.config.scorer_bev.lora_rank=16 \
  agent.config.scorer_bev.init_gate=0.0 \
  agent.config.decoder_bev.init_gate=0.0 \
  > exp/v2_eval_<short_run_name>_$(date +%m%d_%H%M).log 2>&1 &
```

For decoder+scorer checkpoints, pass `agent.config.use_bev_in_scorer=true`.

Residual proposal refiner:

```bash
cd /mnt/ws-frb/users/jingyuso/wenzhet/navsim
source setup_env.sh
export PATH=/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin:$PATH
export DRIVOR_ROOT=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR
CHECKPOINT=/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/ke/<experiment>/<uid>/checkpoints/<best>.ckpt \
EXPERIMENT=drivoR_nav2_bev_residual_<short_run_name> \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
setsid nohup bash scripts/evaluation/run_drivoR_pdm_score_v2_residual_refiner.sh \
  > exp/v2_eval_residual_$(date +%m%d_%H%M).log 2>&1 &
```

Monitor v2:

```bash
tail -f navsim/exp/<eval_experiment>/<timestamp>/run_pdm_score.log
```

Parse v2 CSV:

```bash
python - <<'PY'
import csv
p = "navsim/exp/<eval_experiment>/<timestamp>/<results>.csv"
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
for r in rows[-3:]:
    print(r["token"], r["score"])
PY
```

## Evaluation Rules That Prevent Bad Runs

- Structure-changing eval params must match training: `decoder_bev.lora_rank`, `scorer_bev.lora_rank`, `tf_d_model`, `tf_d_ffn`, `ref_num`, `proposal_num`, `bev_channels`.
- `init_gate` does not affect tensor shape; still pass the training value for clarity.
- Quote checkpoint Hydra overrides containing `=`: `"agent.checkpoint_path='...best-epoch=0-step=1329.ckpt'"`.
- v1 must use `NAVSIM_DEVKIT_ROOT=DrivoR`; v2 must use `NAVSIM_DEVKIT_ROOT=navsim`.
- v2 must prepend official `navsim` to `PYTHONPATH`; otherwise DrivoR's egg-link can shadow `scene_aggregator`.
- Do not use v2 `max_scenes` as an official score; partial chains can break aggregation.
- Do not re-enable epoch-end PDMS during training unless requested; it is slow and previously failed under training.
- Do not skip `agent.initialize()` in training; pretrained fine-tune relies on it to load the baseline before freezing.

## Recent Reference Results

For the canonical local evaluation ledger, including checkpoint paths, network setup, LoRA/gate settings, v1 PDMS sub-scores, v2 EPDMS stage scores, and result CSV paths, read:

`DrivoR/docs/evaluation-ledger.md`

Jul08 decoder+scorer LoRA16 fine-tune best checkpoint:

`DrivoR/exp/ke/Jul08-golduck-4gpu-lora16-0initgate-bev-decoder-scorer-finetune-30epochs/07.08_23.51/checkpoints/best-epoch=0-step=1329.ckpt`

| Benchmark | Result |
|---|---:|
| NAVSIM v1 PDMS | `0.9360065084481483` |
| NAVSIM v2 stage_one EPDMS | `0.847325237374825` |
| NAVSIM v2 stage_two EPDMS | `0.5474358201575438` |
| NAVSIM v2 combined EPDMS | `0.46783133411282796` |

Older decoder-only LoRA16 reference: v1 PDMS about `0.9322`; v2 combined EPDMS about `0.4966`.

Residual proposal refiner epoch 29 reference: v1 PDMS `0.931695`; v2 combined EPDMS `0.478820`.

Proposal-conditioned world refiner best epoch 3 reference: v1 PDMS `0.934903`
(12,146 valid / 0 failed), versus pretrained baseline `0.936905`. NAVSIM v2 is
pending. Full provenance and submetrics are in `docs/evaluation-ledger.md` and
`docs/experiments/idea0002_01_fast_validation.md`.

For detailed v1 BEV eval troubleshooting use `drivor-bev-eval`. For v2-specific setup and failures use `drivor-navsim-v2-eval`. For W&B-specific smoke/debug use `drivor-wandb-smoke`.
