# #!/bin/bash

EXPERIMENT="${1:-cache_navsim_full_same_as_training}"
EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"
# ─── Launcher-level logging: duplicate ALL stdout+stderr to launcher.log ───
LAUNCHER_LOG_DIR="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/ke/${EXPERIMENT}/${EXPERIMENT_UID}"
LAUNCHER_LOG_FILE="${LAUNCHER_LOG_DIR}/launcher.log"
mkdir -p "$LAUNCHER_LOG_DIR"
exec > >(tee -a "$LAUNCHER_LOG_FILE") 2>&1

on_exit() {
  local rc=$?
  echo ""
  echo "=== Launcher finished ==="
  echo "Exit code  : $rc"
  echo "Finish time: $(date -Is)"
  exit $rc
}
trap on_exit EXIT

## Running on goldeen now
## /exp/navsim_cache_nommcv_full

# OPENSCENE_DATA_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset" \
# NUPLAN_MAPS_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset/maps" \
# NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NAVSIM_EXP_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp" \
# NAVSIM_DEVKIT_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR" \
# PYTHONUNBUFFERED=1 \
# /mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python -u \
# "/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/navsim/planning/script/run_dataset_caching.py" \
# train_test_split=navtrain \
# split=trainval \
# agent=drivoR \
# worker=sequential \
# experiment_name=cache_navsim_full \
# cache_path="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/navsim_cache_nommcv_full" \
# force_cache_computation=true \
# # agent.config.long_trajectory_additional_poses=2 \
# # worker.threads_per_node=32 \
# # dataloader.params.batch_size=1 

## Same as training script
## /exp/navsim_cache_nommcv_same_as_training

OPENSCENE_DATA_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset" \
NUPLAN_MAPS_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset/maps" \
NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NAVSIM_EXP_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp" \
NAVSIM_DEVKIT_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR" \
PYTHONUNBUFFERED=1 \
/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python -u \
"/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/navsim/planning/script/run_dataset_caching.py" \
train_test_split=navtrain \
split=trainval \
agent=drivoR \
worker=sequential \
experiment_name=cache_navsim_full \
cache_path="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp/navsim_cache_nommcv_same_as_training" \
force_cache_computation=true \
agent.config.long_trajectory_additional_poses=2 \
# worker.threads_per_node=32 \
# dataloader.params.batch_size=1 \
agent.config.use_bev_feature=true \
agent.config.bev_feature_type=decoder_neck \
agent.config.bev_channels=256 \
agent.config.bev_features_root="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained" \
agent.config.bev_data_split=trainval 