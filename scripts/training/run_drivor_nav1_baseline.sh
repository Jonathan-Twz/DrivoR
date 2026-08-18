#!/usr/bin/env bash
# Train the original DrivoR NAVSIM-v1 baseline without BEV fine-tuning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$DRIVOR_ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/navsim_dataset}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor-matplotlib-$USER}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"
mkdir -p "$MPLCONFIGDIR"

EXPERIMENT="${1:-Jun26-drivor-nav1-baseline-no-bev-25epochs}"
EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"
MAX_EPOCHS="${2:-25}"
WANDB_PROJECT="${WANDB_PROJECT:-drivor-baseline}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}/${EXPERIMENT_UID}}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
BASE_LR="${BASE_LR:-0.0002}"
TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"

LOG_DIR="$NAVSIM_EXP_ROOT/ke/$EXPERIMENT/$EXPERIMENT_UID"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/launcher.log") 2>&1

echo "=== DrivoR NAVSIM-v1 baseline training ==="
echo "Experiment    : $EXPERIMENT"
echo "Experiment UID: $EXPERIMENT_UID"
echo "W&B project   : $WANDB_PROJECT"
echo "W&B run       : $WANDB_RUN_NAME"
echo "Epochs        : $MAX_EPOCHS"
echo "GPUs          : $NUM_GPUS"
echo "Batch size    : $BATCH_SIZE"
echo "LR            : $BASE_LR"
echo "No BEV        : true"
echo "=========================================="

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent=drivoR \
  experiment_name="$EXPERIMENT" \
  experiment_uid="$EXPERIMENT_UID" \
  train_test_split=navtrain \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  +trainer.params.devices="$NUM_GPUS" \
  trainer.params.strategy="$TRAINER_STRATEGY" \
  dataloader.params.prefetch_factor="$PREFETCH_FACTOR" \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers="$NUM_WORKERS" \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="$BASE_LR" \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.config.use_bev_feature=false \
  agent.config.use_bev_in_scorer=false \
  agent.config.use_bev_in_decoder=false \
  agent.config.use_bev_residual_proposal_refiner=false \
  agent.config.freeze_pretrained_except_bev_scorer=false \
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
  seed=2 \
  +trainer.params.logger._target_=pytorch_lightning.loggers.WandbLogger \
  +trainer.params.logger.project="$WANDB_PROJECT" \
  +trainer.params.logger.name="$WANDB_RUN_NAME" \
  +trainer.params.logger.save_dir="$LOG_DIR" \
  +trainer.params.logger.id="$EXPERIMENT_UID" \
  +trainer.params.logger.offline=false
