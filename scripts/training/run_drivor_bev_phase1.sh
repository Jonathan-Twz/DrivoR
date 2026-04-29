#!/bin/bash
# Phase-1 fine-tuning of DrivoR with BEV injection into the scorer only.
#
# Loads a pre-trained DrivoR checkpoint via `agent.checkpoint_path`
# (strict=False in initialize()), freezes the main network and only trains the
# BEV tokenizer, side-LoRA adapters and the new cross_attn_bev sublayers inside
# scorer_attention.
#
# Usage:
#   bash scripts/training/run_drivor_bev_phase1.sh <baseline_ckpt> [experiment_name] [max_epochs]
#
# Assumes:
#   - dataset lives at /home/wenzhe/wm_ws/WoTE/dataset
#   - BEV features live at /media/hdd/wenzhe/bev_features/{trainval,test}/<log>/<token>_<suffix>.pt

set -euo pipefail

export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/wenzhe/wm_ws/WoTE/dataset/maps"
export OPENSCENE_DATA_ROOT="/home/wenzhe/wm_ws/WoTE/dataset"
# IMPORTANT: force the DrivoR devkit so we run DrivoR's run_training_full.py, not WoTE's.
export NAVSIM_DEVKIT_ROOT="/home/wenzhe/wm_ws/DrivoR"
export NAVSIM_EXP_ROOT="/home/wenzhe/wm_ws/DrivoR/exp"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

BASELINE_CKPT="${1:?Usage: $0 <baseline_checkpoint.(ckpt|pth)> [experiment_name] [max_epochs]}"
EXPERIMENT="${2:-training_drivor_bev_scorer_phase1}"
MAX_EPOCHS="${3:-2}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_LR="${BASE_LR:-1e-4}"

echo "=== DrivoR BEV Scorer Phase-1 Fine-tuning ==="
echo "Baseline ckpt: $BASELINE_CKPT"
echo "Experiment   : $EXPERIMENT"
echo "Max epochs   : $MAX_EPOCHS"
echo "GPUs         : $NUM_GPUS"
echo "Batch size   : $BATCH_SIZE"
echo "Base LR      : $BASE_LR"
echo "============================================="

PYTHONUNBUFFERED=1 \
/home/wenzhe/miniconda/envs/drivoR/bin/python -u \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent=drivoR \
  experiment_name="$EXPERIMENT" \
  train_test_split=navtrain \
  split=trainval \
  cache_path=null \
  use_cache_without_dataset=false \
  force_cache_computation=false \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  +trainer.params.devices="$NUM_GPUS" \
  trainer.params.strategy=ddp \
  dataloader.params.prefetch_factor=1 \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers=8 \
  agent.checkpoint_path="$BASELINE_CKPT" \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="$BASE_LR" \
  agent.config.use_bev_feature=true \
  agent.config.freeze_pretrained_except_bev_scorer=true \
  agent.config.bev_feature_type=decoder_neck \
  agent.config.bev_channels=256 \
  agent.config.bev_features_root=/media/hdd/wenzhe/bev_features \
  agent.config.bev_data_split=trainval \
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
