#!/usr/bin/env bash
# Submit from DrivoR after creating exp/cache_slurm.
# Each task writes separate metadata; combine and validate after all tasks finish.
#SBATCH --job-name=drivor-metrics
#SBATCH --account=xiaonanh0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --array=0-31%8
#SBATCH --output=exp/cache_slurm/%A_%a.log
set -euo pipefail
DRIVOR_ROOT="${DRIVOR_ROOT:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:-$(dirname "$DRIVOR_ROOT")/navsim_dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="$DRIVOR_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/train_metric_cache}"
export CACHE_SHARDS="${CACHE_SHARDS:-32}"
export SCENE_FILTER_TOKEN_FILE="${SCENE_FILTER_TOKEN_FILE:-$DRIVOR_ROOT/exp/bev_feature_tokens/trainval_decoder_neck_tokens_full.txt}"
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
cd "$DRIVOR_ROOT"
exec "${PYTHON_BIN:-/home/wenzhet/.conda/envs/drivor/bin/python}" scripts/training/cache_metrics_shard.py
