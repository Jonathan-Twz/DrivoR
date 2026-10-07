#!/usr/bin/env bash
# Historical filename: run with bash inside an existing allocation, NOT sbatch.
set -euo pipefail
export DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHON_BIN="${PYTHON_BIN:-/home/wenzhet/.conda/envs/drivor-blackwell/bin/python}"
export BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$(dirname "$DRIVOR_ROOT")/navsim_bev_feature/exports_pretrained}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export NUM_GPUS=1 BATCH_SIZE="${BATCH_SIZE:-4}" NUM_WORKERS="${NUM_WORKERS:-2}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}" USE_WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-drivor-bev-decoder}" WANDB_ENTITY="${WANDB_ENTITY:-jonathan-twz}"
export INCLUDE_VAL_LOGS_IN_TRAIN=false USE_RUNTIME_OPTIMIZER_SCHEDULE=true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CHECKPOINT_EVERY_N_TRAIN_STEPS=500
# Read raw inputs until a matching BEV dataset cache has been built.
export USE_CACHE_WITHOUT_DATASET="${USE_CACHE_WITHOUT_DATASET:-false}"
export CACHE_PATH="${CACHE_PATH:-null}"
if [[ ! -d "$BEV_FEATURES_ROOT/trainval" ]]; then
    echo "ERROR: Export/copy trainval BEV features before training: $BEV_FEATURES_ROOT/trainval" >&2
    exit 1
fi
if [[ ! -f "$NAVSIM_EXP_ROOT/train_metric_cache/metadata/verification.json" ]]; then
    echo "ERROR: Finish and verify the training metric cache first." >&2
    exit 1
fi
cd "$DRIVOR_ROOT"
if [[ "${RUN_MODE:-train}" == smoke ]]; then
    export MAX_EPOCHS="${SMOKE_MAX_EPOCHS:-2}" LIMIT_TRAIN_BATCHES="${SMOKE_TRAIN_BATCHES:-8}" LIMIT_VAL_BATCHES="${SMOKE_VAL_BATCHES:-2}"
    export SCENE_FILTER_MAX_SCENES="${SMOKE_MAX_SCENES:-4096}"
fi
bash scripts/training/run_in_allocation.sh bash scripts/training/run_drivor_bev_phase1.sh \
    "${BASELINE_CKPT:-$DRIVOR_ROOT/weights/checkpoints/drivor_Nav1_25epochs.pth}" \
    "${EXPERIMENT:-greatlakes-bev-decoder-1gpu-batch4-nocache}" "${MAX_EPOCHS:-30}"
