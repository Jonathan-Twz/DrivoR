#!/usr/bin/env bash
# Launch a step, never a new allocation. Use bash, not sbatch.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ALLOCATION_ID="${ALLOCATION_ID:-${SLURM_JOB_ID:-}}"
: "${ALLOCATION_ID:?Set ALLOCATION_ID to your existing seven-day GPU job}"
: "${NUM_GPUS:?Set NUM_GPUS}"
[[ "$ALLOCATION_ID" =~ ^[0-9]+$ ]] || { echo 'ALLOCATION_ID must be a numeric job ID' >&2; exit 1; }
if [[ "${RUN_MODE:-train}" == smoke ]]; then
    EXPECTED_RUN_HOURS="${EXPECTED_RUN_HOURS:-0.25}"
fi
: "${EXPECTED_RUN_HOURS:?Set an estimated experiment duration in hours; checked against allocation time left}"
PYTHON_BIN="${PYTHON_BIN:-/home/wenzhet/.conda/envs/drivor-blackwell/bin/python}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    lock_dir="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}/allocation_locks"
    mkdir -p "$lock_dir"
    exec 9>"$lock_dir/$ALLOCATION_ID.lock"
    flock -n 9 || { echo "Another launcher owns allocation $ALLOCATION_ID" >&2; exit 1; }
fi
checked="$("$PYTHON_BIN" "$SCRIPT_DIR/check_allocation.py" \
    --job "$ALLOCATION_ID" --gpus "$NUM_GPUS" --cpus "${CPUS_PER_TASK:-0}" \
    --expected-hours "$EXPECTED_RUN_HOURS" --reserve-minutes "${TIME_RESERVE_MINUTES:-5}" \
    --account "${EXPECTED_ACCOUNT:-}")"
read -r cpus minutes <<< "$checked"
step=(srun --jobid="$ALLOCATION_ID" --overlap --nodes=1 --ntasks="$NUM_GPUS"
      --ntasks-per-node="$NUM_GPUS" --cpus-per-task="$cpus" --gpus="$NUM_GPUS"
      --gpu-bind=none --kill-on-bad-exit=1 --time="$minutes"
      --job-name="${STEP_NAME:-drivor-train}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'Validated command (not executed): '
    printf '%q ' "${step[@]}" "$@"
    printf '\n'
    exit 0
fi
test "$#" -gt 0
test -f "${BASELINE_CKPT:-$DRIVOR_ROOT/weights/checkpoints/drivor_Nav1_25epochs.pth}"
test -f "${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}/train_metric_cache/metadata/verification.json"
test -d "${BEV_FEATURES_ROOT:-$(dirname "$DRIVOR_ROOT")/navsim_bev_feature/exports_pretrained}/trainval"
if [[ "${WANDB_MODE:-online}" == online && "${USE_WANDB:-1}" == 1 ]]; then
    "$PYTHON_BIN" -c 'import wandb; print("W&B authenticated:", wandb.Api(timeout=30).viewer.username)'
fi
cd "$DRIVOR_ROOT"
exec "${step[@]}" "$@"
