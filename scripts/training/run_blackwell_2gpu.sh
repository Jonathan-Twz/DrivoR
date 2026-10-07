#!/usr/bin/env bash
set -euo pipefail
cd /nfs/turbo/coe-xiaonanh/wenzhet/DrivoR
export DRIVOR_ROOT="$PWD"
export PYTHON_BIN="${PYTHON_BIN:-/home/wenzhet/.conda/envs/drivor-blackwell/bin/python}"
export NAVSIM_EXP_ROOT="$DRIVOR_ROOT/exp"
export BEV_FEATURES_ROOT=/nfs/turbo/coe-xiaonanh/wenzhet/navsim_bev_feature/exports_pretrained
export NUM_GPUS=2 BATCH_SIZE="${BATCH_SIZE:-32}" ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}" NUM_WORKERS="${NUM_WORKERS:-6}" PREFETCH_FACTOR=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# NCCL 2.26.2 Simple/LL128 kernels fail on these Blackwell GPUs; LL passes
# the standalone collective probe. Keep this workaround local to this launcher.
export NCCL_PROTO=LL
export USE_BEV_IN_DECODER=true USE_BEV_IN_SCORER=false
export SCORER_BEV_INIT_GATE=0.1 SCORER_BEV_LORA_RANK=16
export DECODER_BEV_LORA_RANK=16 DECODER_BEV_INIT_GATE=0.1 BASE_LR=1e-4
export EGO_MOTION_DROPOUT_PROB="${EGO_MOTION_DROPOUT_PROB:-0.10}" EGO_MOTION_DROPOUT_WARMUP_EPOCHS=1
export CHECKPOINT_EVERY_N_TRAIN_STEPS=500
export USE_CACHE_WITHOUT_DATASET=false CACHE_PATH=null USE_RAY_SCORE=false
export INCLUDE_VAL_LOGS_IN_TRAIN=false USE_RUNTIME_OPTIMIZER_SCHEDULE=true
export WANDB_MODE=online USE_WANDB=1 WANDB_PROJECT=drivor-bev-decoder WANDB_ENTITY=jonathan-twz
export EXPERIMENT="${EXPERIMENT:-bev-decoder-ego${EGO_MOTION_DROPOUT_PROB}-2PRO6000-b${BATCH_SIZE}-job${ALLOCATION_ID:-${SLURM_JOB_ID:-unset}}-${RUN_MODE:-train}}"
export EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"
export BEV_TOKEN_FILTER_FILE="${BEV_TOKEN_FILTER_FILE:-$NAVSIM_EXP_ROOT/bev_feature_tokens/jun12_reference_tokens.txt}"
export MAX_EPOCHS="${MAX_EPOCHS:-30}" LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}" LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
export SCENE_FILTER_MAX_SCENES="${SCENE_FILTER_MAX_SCENES:-}" FORCE_CACHE_COMPUTATION=false TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-null}"
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
if [[ "${RUN_MODE:-train}" == "smoke" ]]; then
    export MAX_EPOCHS="${SMOKE_MAX_EPOCHS:-1}" LIMIT_TRAIN_BATCHES="${SMOKE_TRAIN_BATCHES:-2}" LIMIT_VAL_BATCHES="${SMOKE_VAL_BATCHES:-2}"
    export SCENE_FILTER_MAX_SCENES="${SMOKE_MAX_SCENES:-512}"
fi
# Preserve trusted local checkpoint loading behavior with PyTorch >= 2.6.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
bash scripts/training/run_in_allocation.sh bash scripts/training/run_drivor_bev_phase1.sh \
    "$DRIVOR_ROOT/weights/checkpoints/drivor_Nav1_25epochs.pth" "$EXPERIMENT" "$MAX_EPOCHS"
