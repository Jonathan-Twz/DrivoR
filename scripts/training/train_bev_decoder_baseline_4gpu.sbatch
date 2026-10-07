#!/usr/bin/env bash
# Historical filename: run with bash inside an existing allocation, NOT sbatch.
set -euo pipefail
cd /nfs/turbo/coe-xiaonanh/wenzhet/DrivoR
export DRIVOR_ROOT="$PWD"
export PYTHON_BIN="${PYTHON_BIN:-/home/wenzhet/.conda/envs/drivor-blackwell/bin/python}"
export NAVSIM_EXP_ROOT="$DRIVOR_ROOT/exp"
export BEV_FEATURES_ROOT=/nfs/turbo/coe-xiaonanh/wenzhet/navsim_bev_feature/exports_pretrained
export NUM_GPUS=4 BATCH_SIZE=16 ACCUMULATE_GRAD_BATCHES=1 NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR=1
export SCORING_WORKERS="${SCORING_WORKERS:-3}" SCORING_WORKER_THREADS="${SCORING_WORKER_THREADS:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export USE_BEV_IN_DECODER=true USE_BEV_IN_SCORER=false
export SCORER_BEV_INIT_GATE=0.1 SCORER_BEV_LORA_RANK=16
export DECODER_BEV_LORA_RANK=16 DECODER_BEV_INIT_GATE=0.1 BASE_LR=1e-4
export EGO_MOTION_DROPOUT_PROB=0.0 EGO_MOTION_DROPOUT_WARMUP_EPOCHS=0
export CHECKPOINT_EVERY_N_TRAIN_STEPS=500
export USE_CACHE_WITHOUT_DATASET=false CACHE_PATH=null USE_RAY_SCORE=false
export INCLUDE_VAL_LOGS_IN_TRAIN=false USE_RUNTIME_OPTIMIZER_SCHEDULE=true
export WANDB_MODE=online USE_WANDB=1 WANDB_PROJECT=drivor-bev-decoder WANDB_ENTITY=jonathan-twz
export EXPERIMENT="${EXPERIMENT:-bev-decoder-ego00-4gpu-b16-job${ALLOCATION_ID:-${SLURM_JOB_ID:-unset}}}"
export EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"
export BEV_TOKEN_FILTER_FILE="${BEV_TOKEN_FILTER_FILE:-$NAVSIM_EXP_ROOT/bev_feature_tokens/jun12_reference_tokens.txt}"
export MAX_EPOCHS="${MAX_EPOCHS:-30}" LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}" LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
export SCENE_FILTER_MAX_SCENES="${SCENE_FILTER_MAX_SCENES:-}" FORCE_CACHE_COMPUTATION=false TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-null}"
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [[ "${RUN_MODE:-train}" == smoke ]]; then
    export MAX_EPOCHS="${SMOKE_MAX_EPOCHS:-2}" LIMIT_TRAIN_BATCHES="${SMOKE_TRAIN_BATCHES:-8}" LIMIT_VAL_BATCHES="${SMOKE_VAL_BATCHES:-2}"
    export SCENE_FILTER_MAX_SCENES="${SMOKE_MAX_SCENES:-4096}"
fi
bash scripts/training/run_in_allocation.sh bash scripts/training/run_drivor_bev_phase1.sh \
    "$DRIVOR_ROOT/weights/checkpoints/drivor_Nav1_25epochs.pth" "$EXPERIMENT" "$MAX_EPOCHS"
