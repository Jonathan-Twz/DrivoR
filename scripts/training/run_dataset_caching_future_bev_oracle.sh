#!/bin/bash
# Cache DrivoR training samples for the current+future BEV oracle experiment.
#
# This matches scripts/training/run_drivor_future_bev_oracle.sh for the fields
# that affect cached feature/target schema: current BEV features, privileged
# future BEV targets, and the long trajectory target.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$DRIVOR_ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/navsim_dataset}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python}"
BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$WORKSPACE_ROOT/navsim_bev_feature/exports_pretrained}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor-matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

EXPERIMENT="${1:-cache_future_bev_oracle_decoder_scorer}"
EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-trainval}"
WORKER="${WORKER:-sequential}"
SCENE_FILTER_MAX_SCENES="${SCENE_FILTER_MAX_SCENES:-}"

CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/navsim_cache_future_bev_oracle_decoder_scorer}"
FORCE_CACHE_COMPUTATION="${FORCE_CACHE_COMPUTATION:-false}"

BEV_DATA_SPLIT="${BEV_DATA_SPLIT:-trainval}"
BEV_FEATURE_TYPE="${BEV_FEATURE_TYPE:-decoder_neck}"
FUTURE_BEV_NUM_STEPS="${FUTURE_BEV_NUM_STEPS:-4}"
FUTURE_BEV_STRIDE="${FUTURE_BEV_STRIDE:-1}"
USE_FUTURE_BEV_IN_DECODER="${USE_FUTURE_BEV_IN_DECODER:-true}"
USE_FUTURE_BEV_IN_SCORER="${USE_FUTURE_BEV_IN_SCORER:-true}"

BEV_TOKEN_FILTER_DEFAULT="$NAVSIM_EXP_ROOT/bev_feature_tokens/${BEV_DATA_SPLIT}_${BEV_FEATURE_TYPE}_tokens_full.txt"
BEV_TOKEN_FILTER_FILE="${BEV_TOKEN_FILTER_FILE:-$BEV_TOKEN_FILTER_DEFAULT}"

LAUNCHER_LOG_DIR="$NAVSIM_EXP_ROOT/ke/$EXPERIMENT/$EXPERIMENT_UID"
LAUNCHER_LOG_FILE="$LAUNCHER_LOG_DIR/launcher.log"
mkdir -p "$LAUNCHER_LOG_DIR"
exec > >(tee -a "$LAUNCHER_LOG_FILE") 2>&1

trap 'rc=$?; echo ""; echo "=== Dataset cache launcher finished ==="; echo "Exit code  : $rc"; echo "Finish time: $(date -Is)"; exit $rc' EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$BEV_TOKEN_FILTER_FILE" ]]; then
  echo "Generating BEV token filter from $BEV_FEATURES_ROOT/$BEV_DATA_SPLIT ..."
  mkdir -p "$(dirname "$BEV_TOKEN_FILTER_FILE")"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
bev_root = Path("$BEV_FEATURES_ROOT") / "$BEV_DATA_SPLIT"
out_path = Path("$BEV_TOKEN_FILTER_FILE")
suffix = "_$BEV_FEATURE_TYPE.pt"
if not bev_root.exists():
    raise FileNotFoundError(f"BEV feature split directory does not exist: {bev_root}")
tokens = sorted({path.name[:-len(suffix)] for path in bev_root.rglob(f"*{suffix}")})
if not tokens:
    raise RuntimeError(f"No BEV feature files matching *{suffix} under {bev_root}")
out_path.write_text("\n".join(tokens) + "\n")
print(f"Wrote {len(tokens)} BEV tokens to {out_path}")
PY
fi

HYDRA_OVERRIDES=(
  train_test_split="$TRAIN_TEST_SPLIT"
  split="$SPLIT"
  agent=drivoR
  worker="$WORKER"
  experiment_name="$EXPERIMENT"
  cache_path="$CACHE_PATH"
  use_cache_without_dataset=false
  force_cache_computation="$FORCE_CACHE_COMPUTATION"
  +scene_filter_token_file="$BEV_TOKEN_FILTER_FILE"
  agent.config.use_bev_feature=true
  agent.config.use_privileged_future_bev=true
  agent.config.validate_with_privileged_future_bev=true
  agent.config.future_bev_num_steps="$FUTURE_BEV_NUM_STEPS"
  agent.config.future_bev_stride="$FUTURE_BEV_STRIDE"
  agent.config.use_future_bev_in_decoder="$USE_FUTURE_BEV_IN_DECODER"
  agent.config.use_future_bev_in_scorer="$USE_FUTURE_BEV_IN_SCORER"
  agent.config.use_bev_in_decoder=true
  agent.config.use_bev_in_scorer=true
  agent.config.bev_feature_type="$BEV_FEATURE_TYPE"
  agent.config.bev_channels=256
  agent.config.bev_features_root="$BEV_FEATURES_ROOT"
  agent.config.bev_data_split="$BEV_DATA_SPLIT"
  agent.config.long_trajectory_additional_poses=2
  agent.config.use_ray_score=false
)

if [[ -n "$SCENE_FILTER_MAX_SCENES" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="$SCENE_FILTER_MAX_SCENES")
fi

echo "=== DrivoR Future BEV Oracle Dataset Cache ==="
echo "Launcher log : $LAUNCHER_LOG_FILE"
echo "DrivoR root  : $DRIVOR_ROOT"
echo "Data root    : $DATA_ROOT"
echo "BEV root     : $BEV_FEATURES_ROOT/$BEV_DATA_SPLIT"
echo "Token filter : $BEV_TOKEN_FILTER_FILE"
echo "Cache path   : $CACHE_PATH"
echo "Experiment   : $EXPERIMENT / $EXPERIMENT_UID"
echo "Future BEV   : steps=$FUTURE_BEV_NUM_STEPS stride=$FUTURE_BEV_STRIDE decoder=$USE_FUTURE_BEV_IN_DECODER scorer=$USE_FUTURE_BEV_IN_SCORER"
echo "Current BEV  : decoder=true scorer=true"
echo "Worker/force : worker=$WORKER force=$FORCE_CACHE_COMPUTATION max_scenes=${SCENE_FILTER_MAX_SCENES:-all}"
echo "=============================================="

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py" \
  "${HYDRA_OVERRIDES[@]}"
