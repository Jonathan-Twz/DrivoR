#!/usr/bin/env bash
# NAVSIM v1 PDMS evaluation for the post-decoder BEV residual proposal refiner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DRIVOR_ROOT"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DRIVOR_ROOT/../navsim_dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DRIVOR_ROOT/../navsim_dataset}"
export BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$DRIVOR_ROOT/../navsim_bev_feature/exports_pretrained}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to a residual-proposal-refiner checkpoint}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-drivoR_nav1-bev-residual-proposal-refiner}"
RESIDUAL_REFINER_NUM_LAYERS="${RESIDUAL_REFINER_NUM_LAYERS:-1}"
RESIDUAL_REFINER_NUM_HEADS="${RESIDUAL_REFINER_NUM_HEADS:-1}"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  train_test_split=navtest \
  agent=drivoR \
  "agent.checkpoint_path='${CKPT_PATH}'" \
  "experiment_name='${EXPERIMENT_NAME}'" \
  agent.config.proposal_num=64 \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.config.use_bev_feature=true \
  agent.config.use_bev_in_scorer=false \
  agent.config.use_bev_in_decoder=false \
  agent.config.use_bev_residual_proposal_refiner=true \
  agent.config.bev_residual_proposal_refiner.num_layers="$RESIDUAL_REFINER_NUM_LAYERS" \
  agent.config.bev_residual_proposal_refiner.num_heads="$RESIDUAL_REFINER_NUM_HEADS" \
  agent.config.bev_feature_type=decoder_neck \
  agent.config.bev_channels=256 \
  agent.config.bev_features_root="$BEV_FEATURES_ROOT" \
  agent.config.bev_data_split=test \
  agent.config.long_trajectory_additional_poses=2 \
  +trainer.params.inference_mode=false \
  "$@"
