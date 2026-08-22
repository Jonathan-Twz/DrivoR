#!/bin/bash
# Full 8-GPU training for the proposal-conditioned BEV world refiner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FULL_DATASET=1
export VARIANT=proposal_world
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export MAX_EPOCHS="${MAX_EPOCHS:-30}"
export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
# Scheduler steps once per optimizer update, so account for two-way accumulation.
export SCHEDULER_DATASET_SIZE="${SCHEDULER_DATASET_SIZE:-62741}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-20}"
export INIT_REFINE_GATE="${INIT_REFINE_GATE:-0.01}"
export INIT_SCORE_GATE="${INIT_SCORE_GATE:-0.01}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-drivor-world-model}"
export USE_WANDB=1
export EXPERIMENT="${EXPERIMENT:-Aug18-idea0002-01-proposal-world-full-8gpu-gates001}"

exec "$SCRIPT_DIR/run_drivor_idea0002_01_fast.sh"
