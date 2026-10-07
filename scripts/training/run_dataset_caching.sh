#!/usr/bin/env bash
# Cache BEV training inputs only after exporting BEV tensors.
set -euo pipefail
DRIVOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$DRIVOR_ROOT")/navsim_dataset}"
PYTHON_BIN="${PYTHON_BIN:-/home/wenzhet/.conda/envs/drivor/bin/python}"
export NAVSIM_DEVKIT_ROOT="$DRIVOR_ROOT"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$(dirname "$DRIVOR_ROOT")/navsim_bev_feature/exports_pretrained}"
BEV_DATA_SPLIT="${BEV_DATA_SPLIT:-trainval}"
if [[ ! -d "$BEV_FEATURES_ROOT/$BEV_DATA_SPLIT" ]]; then
    echo "ERROR: BEV features missing: $BEV_FEATURES_ROOT/$BEV_DATA_SPLIT" >&2
    exit 1
fi
cd "$DRIVOR_ROOT"
exec "$PYTHON_BIN" navsim/planning/script/run_dataset_caching.py \
    train_test_split=navtrain split=trainval agent=drivoR worker=sequential \
    experiment_name="${EXPERIMENT:-cache_navsim_same_as_training}" \
    cache_path="${CACHE_PATH:-$NAVSIM_EXP_ROOT/navsim_cache_nommcv_same_as_training}" \
    force_cache_computation="${FORCE_CACHE_COMPUTATION:-false}" \
    agent.config.long_trajectory_additional_poses=2 \
    agent.config.use_bev_feature=true \
    agent.config.bev_feature_type=decoder_neck agent.config.bev_channels=256 \
    agent.config.bev_features_root="$BEV_FEATURES_ROOT" \
    agent.config.bev_data_split="$BEV_DATA_SPLIT" "$@"
