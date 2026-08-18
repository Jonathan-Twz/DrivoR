#!/bin/bash
# Wait for one A100, run a smoke check, then execute the matched experiment pair.
# This process does not reserve GPU resources while waiting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR}"
TARGET_GPU="${TARGET_GPU:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-8000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-15}"
READY_CHECKS_REQUIRED="${READY_CHECKS_REQUIRED:-3}"
QUEUE_ROOT="${QUEUE_ROOT:-$REFERENCE_ROOT/exp/idea0002_fast}"
QUEUE_LOG="${QUEUE_LOG:-$QUEUE_ROOT/queue_idea0002_01.log}"

mkdir -p "$QUEUE_ROOT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "=== idea_0002_01 paired-run queue ==="
echo "Started       : $(date -Is)"
echo "Worktree      : $DRIVOR_ROOT"
echo "Branch        : $(git -C "$DRIVOR_ROOT" branch --show-current)"
echo "Commit        : $(git -C "$DRIVOR_ROOT" rev-parse HEAD)"
echo "Target GPU    : $TARGET_GPU"
echo "Ready rule    : memory<=$MAX_MEMORY_USED_MB MiB, utilization<=$MAX_UTILIZATION%, $READY_CHECKS_REQUIRED checks"
echo "Queue log     : $QUEUE_LOG"

ready_checks=0
while (( ready_checks < READY_CHECKS_REQUIRED )); do
  read -r memory_used utilization < <(
    nvidia-smi \
      --id="$TARGET_GPU" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ','
  )
  if (( memory_used <= MAX_MEMORY_USED_MB && utilization <= MAX_UTILIZATION )); then
    ready_checks=$((ready_checks + 1))
  else
    ready_checks=0
  fi
  echo "$(date -Is) gpu=$TARGET_GPU memory=${memory_used}MiB util=${utilization}% ready=$ready_checks/$READY_CHECKS_REQUIRED"
  if (( ready_checks < READY_CHECKS_REQUIRED )); then
    sleep "$POLL_SECONDS"
  fi
done

echo "$(date -Is) GPU ready; starting smoke test"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" \
NUM_GPUS=1 \
VARIANT=proposal_world \
EXPERIMENT=Aug18-idea0002-01-proposal-world-smoke \
TOKEN_COUNT=128 \
MAX_EPOCHS=1 \
LIMIT_TRAIN_BATCHES=2 \
LIMIT_VAL_BATCHES=2 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
ACCUMULATE_GRAD_BATCHES=1 \
USE_WANDB=0 \
bash "$SCRIPT_DIR/run_drivor_idea0002_01_fast.sh"

echo "$(date -Is) smoke passed; benchmarking both modules on A100"
PYTHONPATH="$DRIVOR_ROOT" \
CUDA_VISIBLE_DEVICES="$TARGET_GPU" \
/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python \
  "$SCRIPT_DIR/benchmark_proposal_world_refiner.py" \
  --variant static_bev_refiner --device cuda --warmup 10 --iterations 50 \
  > "$QUEUE_ROOT/static_bev_refiner_a100.json"

PYTHONPATH="$DRIVOR_ROOT" \
CUDA_VISIBLE_DEVICES="$TARGET_GPU" \
/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python \
  "$SCRIPT_DIR/benchmark_proposal_world_refiner.py" \
  --variant proposal_world --device cuda --warmup 10 --iterations 50 \
  > "$QUEUE_ROOT/proposal_world_a100.json"

echo "$(date -Is) starting matched static-BEV control"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" \
NUM_GPUS=1 \
VARIANT=static_bev_refiner \
EXPERIMENT=Aug18-idea0002-01-static-bev-refiner-fast \
WANDB_MODE=online \
bash "$SCRIPT_DIR/run_drivor_idea0002_01_fast.sh"

echo "$(date -Is) starting matched proposal-world experiment"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" \
NUM_GPUS=1 \
VARIANT=proposal_world \
EXPERIMENT=Aug18-idea0002-01-proposal-world-fast \
WANDB_MODE=online \
bash "$SCRIPT_DIR/run_drivor_idea0002_01_fast.sh"

echo "$(date -Is) paired validation complete"
