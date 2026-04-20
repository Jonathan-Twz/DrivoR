#!/bin/bash
# DrivoR Nav1 baseline training (short run: 3 epochs for verification)
# Matches original paper hyperparameters (4x GPU, batch_size=16/GPU)
# Data: /media/hdd2/wenzhe (trainval), Maps: WoTE path

# set -euo pipefail

export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/wenzhe/wm_ws/WoTE/dataset/maps"
export OPENSCENE_DATA_ROOT="/media/hdd2/wenzhe"
export NAVSIM_DEVKIT_ROOT="/home/wenzhe/wm_ws/DrivoR"
export NAVSIM_EXP_ROOT="/home/wenzhe/wm_ws/DrivoR/exp"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

EXPERIMENT="${1:-baseline_drivoR_Nav1_3epochs}"
MAX_EPOCHS="${2:-3}"
NUM_GPUS=4

echo "=== DrivoR Nav1 Baseline Training ==="
echo "Experiment: $EXPERIMENT"
echo "Max epochs: $MAX_EPOCHS"
echo "GPUs: $NUM_GPUS"
echo "Data root: $OPENSCENE_DATA_ROOT"
echo "======================================"

PYTHONUNBUFFERED=1 \
/home/wenzhe/miniconda/envs/drivoR/bin/python -u \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent=drivoR \
  experiment_name="$EXPERIMENT" \
  train_test_split=navtrain \
  split=trainval \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  trainer.params.devices="$NUM_GPUS" \
  trainer.params.strategy=ddp \
  dataloader.params.prefetch_factor=1 \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers=8 \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr=0.0002 \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.loss.prev_weight=0.0 \
  agent.config.long_trajectory_additional_poses=2 \
  seed=2
