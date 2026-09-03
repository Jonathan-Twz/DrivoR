#!/usr/bin/env bash
# NAVSIM v1 PDMS evaluation for the proposal-conditioned BEV world refiner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REFERENCE_ROOT="${REFERENCE_ROOT:-/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/ws-frb/users/jingyuso/wenzhet}"
USER_ROOT="${USER_ROOT:-/mnt/ws-frb/users/jingyuso}"
PYTHON_BIN="${PYTHON_BIN:-$USER_ROOT/miniconda3/envs/drivoR-share/bin/python}"

cd "$DRIVOR_ROOT"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$WORKSPACE_ROOT/navsim_dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REFERENCE_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="$DRIVOR_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$WORKSPACE_ROOT/navsim_dataset}"
export BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$WORKSPACE_ROOT/navsim_bev_feature/exports_pretrained}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"

CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to a proposal-world checkpoint}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-drivoR_nav1-idea0002-01-proposal-world}"
WORLD_LAYERS="${WORLD_LAYERS:-2}"
WORLD_HEADS="${WORLD_HEADS:-4}"
WORLD_FFN_DIM="${WORLD_FFN_DIM:-512}"
WORLD_ROLLOUT_STEPS="${WORLD_ROLLOUT_STEPS:-1}"
PROPOSAL_CHUNK_SIZE="${PROPOSAL_CHUNK_SIZE:-8}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  train_test_split=navtest \
  agent=drivoR \
  "agent.checkpoint_path='${CKPT_PATH}'" \
  "experiment_name='${EXPERIMENT_NAME}'" \
  agent.config.proposal_num=64 \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.model_weights="$REFERENCE_ROOT/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors" \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.lidar_backbone.model_weights="$REFERENCE_ROOT/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors" \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.config.noc=1 \
  agent.config.dac=1 \
  agent.config.ddc=0.0 \
  agent.config.ttc=5 \
  agent.config.ep=5 \
  agent.config.comfort=2 \
  agent.config.use_bev_feature=true \
  agent.config.use_bev_in_scorer=false \
  agent.config.use_bev_in_decoder=false \
  agent.config.use_bev_residual_proposal_refiner=false \
  agent.config.use_proposal_world_refiner=true \
  agent.config.use_privileged_future_bev=false \
  agent.config.use_ray_score=true \
  agent.config.bev_feature_type=decoder_neck \
  agent.config.bev_channels=256 \
  agent.config.bev_features_root="$BEV_FEATURES_ROOT" \
  agent.config.bev_data_split=test \
  agent.config.proposal_world_refiner.num_layers="$WORLD_LAYERS" \
  agent.config.proposal_world_refiner.num_heads="$WORLD_HEADS" \
  agent.config.proposal_world_refiner.ffn_dim="$WORLD_FFN_DIM" \
  agent.config.proposal_world_refiner.rollout_steps="$WORLD_ROLLOUT_STEPS" \
  agent.config.proposal_world_refiner.proposal_chunk_size="$PROPOSAL_CHUNK_SIZE" \
  agent.config.proposal_world_refiner.init_refine_gate=0.01 \
  agent.config.proposal_world_refiner.init_score_gate=0.01 \
  agent.config.long_trajectory_additional_poses=2 \
  +trainer.params.inference_mode=false \
  "$@"
