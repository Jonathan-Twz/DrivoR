# ENV variables
export DRIVOR_ROOT="."
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset/maps"
export NAVSIM_EXP_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}"
export OPENSCENE_DATA_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset"
export BEV_FEATURES_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_bev_feature/exports_pretrained"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
# NCCL variables, sync over time
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

export CKPT_PATH="${DRIVOR_ROOT}/weights/checkpoints/drivor_Nav1_25epochs.pth"

export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py \
        train_test_split=navtest \
        agent=drivoR \
        agent.checkpoint_path=$CKPT_PATH \
        experiment_name=drivoR_nav1 \
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
        +trainer.params.inference_mode=false