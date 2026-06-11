#!/bin/bash
# Intended for main branch

export HYDRA_FULL_ERROR=1 \
export DRIVOR_ROOT="./" \
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" \
export NUPLAN_MAPS_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset/maps" \
export NAVSIM_EXP_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp" \
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}" \
export OPENSCENE_DATA_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset" \
EXPERIMENT=training_drivoR_Nav1_traj_long_25epochs
AGENT=drivoR
NUM_GPUS=${NUM_GPUS:-1}
TRAINER_STRATEGY=${TRAINER_STRATEGY:-auto}

/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python \
 ./navsim/planning/script/run_training_full.py  \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
    +trainer.params.devices=$NUM_GPUS \
    trainer.params.strategy=$TRAINER_STRATEGY \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=$NUM_GPUS \
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