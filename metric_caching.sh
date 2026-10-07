#!/usr/bin/env bash
# CPU-only training metric cache; no camera, LiDAR, or BEV tensors required.
set -euo pipefail
DRIVOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$DRIVOR_ROOT")/navsim_dataset}"
PYTHON_BIN="${PYTHON_BIN:-/home/wenzhet/.conda/envs/drivor/bin/python}"
export NAVSIM_DEVKIT_ROOT="$DRIVOR_ROOT"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
cd "$DRIVOR_ROOT"
exec "$PYTHON_BIN" "$DRIVOR_ROOT/navsim/planning/script/run_train_metric_caching.py" \
    train_test_split="${TRAIN_TEST_SPLIT:-navtrain}" \
    '~train_test_split_synthetic' \
    worker=sequential \
    cache.cache_path="${CACHE_PATH:-$NAVSIM_EXP_ROOT/train_metric_cache}" "$@"
