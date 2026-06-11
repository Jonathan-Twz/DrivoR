#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DRIVOR_ROOT"

# ENV variables
export DRIVOR_ROOT
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset/maps"
export NAVSIM_EXP_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}"
export OPENSCENE_DATA_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset"
export BEV_FEATURES_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# NCCL variables, sync over time
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# export CKPT_PATH="${NAVSIM_EXP_ROOT}/ke/3gpu-30epoch-32batch-16worker/05.08_01.23/checkpoints/last.ckpt"
# export CKPT_PATH="${NAVSIM_EXP_ROOT}/ke/guppy-2gpu-16batch-8worker-cache-full/05.20_00.11/checkpoints/last.ckpt"
# export CKPT_PATH="${NAVSIM_EXP_ROOT}/ke/golduck-4gpu-16batch-8worker-cache-full/05.26_22.04/checkpoints/last.ckpt"

# export CKPT_PATH="${NAVSIM_EXP_ROOT}/ke/golduck-4gpu-16batch-8worker-cache-full/05.26_22.04/checkpoints/best-epoch=13-step=18606.ckpt"
# export EXPERIMENT_NAME="drivoR_nav1-best-epoch=13-step=18606"

export CKPT_PATH="${NAVSIM_EXP_ROOT}/ke/golduck-4gpu-16batch-8worker-01gate-16lora/05.28_23.15/checkpoints/best-epoch=12-step=17277.ckpt"
export EXPERIMENT_NAME="drivoR_nav1-best-epoch=12-step=17277-01gate-16lora"

export SUBSCORE_PATH=$NAVSIM_EXP_ROOT

# BEV scorer structure params — MUST match the values used when training the checkpoint,
# otherwise loading state_dict fails with LoRA shape mismatch.
SCORER_BEV_LORA_RANK="${SCORER_BEV_LORA_RANK:-16}"
# SCORER_BEV_INIT_GATE="${SCORER_BEV_INIT_GATE:-0.1}"

# Hydra treats '=' inside override values as syntax; quote the value for Hydra (bash quotes alone are not enough).
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
        agent.config.noc=1 \
        agent.config.dac=1 \
        agent.config.ddc=0.0 \
        agent.config.ttc=5 \
        agent.config.ep=5 \
        agent.config.comfort=2 \
        agent.config.use_bev_feature=true \
        agent.config.use_ray_score=true \
        agent.config.bev_feature_type=decoder_neck \
        agent.config.bev_channels=256 \
        agent.config.scorer_bev.lora_rank="${SCORER_BEV_LORA_RANK}" \
        agent.config.bev_features_root="${BEV_FEATURES_ROOT}" \
        agent.config.bev_data_split=test \
        agent.config.long_trajectory_additional_poses=2 \
        +trainer.params.inference_mode=false
