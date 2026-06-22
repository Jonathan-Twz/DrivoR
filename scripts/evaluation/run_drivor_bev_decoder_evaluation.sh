#!/usr/bin/env bash
# NAVSIM v1 (navtest) PDMS evaluation for the DECODER-side BEV DrivoR agent.
#
# Difference vs run_drivor_bev_evaluation.sh (scorer-side BEV):
#   use_bev_in_scorer=false, use_bev_in_decoder=true, decoder_bev.lora_rank must
#   match training (16).  The scorer stays the original TransformerDecoderScorer.
#
# Runs in the DrivoR tree directly (decoder-BEV code already lives here).
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
# golduck: do NOT set CUDA_DEVICE_ORDER=PCI_BUS_ID -> FASTEST_FIRST makes 0,1,2,3 the four A100s.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# NCCL variables
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# Trained decoder-BEV (LoRA rank 16) checkpoint, 30 epochs.
export CKPT_PATH="${CKPT_PATH:-${NAVSIM_EXP_ROOT}/ke/Jun12-golduck-4gpu-lora16-bev-decoder/06.12_01.12/checkpoints/last.ckpt}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-drivoR_nav1-decoder-bev-lora16-last}"

export SUBSCORE_PATH=$NAVSIM_EXP_ROOT

# Decoder-BEV structure params — MUST match training, else state_dict LoRA shape mismatch.
DECODER_BEV_LORA_RANK="${DECODER_BEV_LORA_RANK:-16}"

# Hydra treats '=' inside override values as syntax; quote the value for Hydra.
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
        agent.config.use_bev_in_scorer=false \
        agent.config.use_bev_in_decoder=true \
        agent.config.use_ray_score=true \
        agent.config.bev_feature_type=decoder_neck \
        agent.config.bev_channels=256 \
        agent.config.decoder_bev.lora_rank="${DECODER_BEV_LORA_RANK}" \
        agent.config.bev_features_root="${BEV_FEATURES_ROOT}" \
        agent.config.bev_data_split=test \
        agent.config.long_trajectory_additional_poses=2 \
        +trainer.params.inference_mode=false \
        "$@"
