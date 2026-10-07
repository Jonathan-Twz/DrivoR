#!/usr/bin/env bash
# Submit with --dependency=afterok:<cache-array-job-id>.
#SBATCH --job-name=drivor-cache-verify
#SBATCH --account=xiaonanh0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=exp/cache_slurm/verify-%j.log
set -euo pipefail
DRIVOR_ROOT="${DRIVOR_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$DRIVOR_ROOT"
exec "${PYTHON_BIN:-/home/wenzhet/.conda/envs/drivor/bin/python}" scripts/training/finalize_metric_cache.py
