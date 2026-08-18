#!/usr/bin/env bash
# Train DrivoR with BEV injection in the trajectory decoder from scratch.
#
# This matches the NAVSIM-v1 baseline training hyperparameters while enabling
# the BEV trajectory-decoder path. It does not load a DrivoR baseline checkpoint
# and it does not freeze the pretrained DrivoR modules.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$DRIVOR_ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/navsim_dataset}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python}"
BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$WORKSPACE_ROOT/navsim_bev_feature/exports_pretrained}"

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

EXPERIMENT="${1:-Jul05-golduck-4gpu-lora8-0initgate-bev-decoder-from-scratch-20epochs}"
MAX_EPOCHS="${2:-20}"
EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-drivor-bev-decoder}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}/${EXPERIMENT_UID}}"

NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
BASE_LR="${BASE_LR:-0.0002}"
TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"

BEV_DATA_SPLIT="${BEV_DATA_SPLIT:-trainval}"
BEV_FEATURE_TYPE="${BEV_FEATURE_TYPE:-decoder_neck}"
USE_BEV_TOKEN_FILTER="${USE_BEV_TOKEN_FILTER:-0}"
BEV_TOKEN_FILTER_DEFAULT="$NAVSIM_EXP_ROOT/bev_feature_tokens/${BEV_DATA_SPLIT}_${BEV_FEATURE_TYPE}_tokens_full.txt"
BEV_TOKEN_FILTER_FILE="${BEV_TOKEN_FILTER_FILE:-$BEV_TOKEN_FILTER_DEFAULT}"
DECODER_BEV_INIT_GATE="${DECODER_BEV_INIT_GATE:-0.0}"
DECODER_BEV_LORA_RANK="${DECODER_BEV_LORA_RANK:-8}"
DRIVOR_EPOCH_PDMS_EVAL="${DRIVOR_EPOCH_PDMS_EVAL:-0}"
DRIVOR_EPOCH_PDMS_EVERY_N_EPOCHS="${DRIVOR_EPOCH_PDMS_EVERY_N_EPOCHS:-1}"
DRIVOR_EPOCH_PDMS_TIMEOUT_SEC="${DRIVOR_EPOCH_PDMS_TIMEOUT_SEC:-7200}"

if [[ "$NUM_WORKERS" == "0" && "$PREFETCH_FACTOR" == "1" ]]; then
  PREFETCH_FACTOR=null
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$USE_BEV_TOKEN_FILTER" == "1" && ! -f "$BEV_TOKEN_FILTER_FILE" ]]; then
  echo "Generating BEV token filter from $BEV_FEATURES_ROOT/$BEV_DATA_SPLIT ..."
  mkdir -p "$(dirname "$BEV_TOKEN_FILTER_FILE")"
  "$PYTHON_BIN" - <<PY
from pathlib import Path

bev_root = Path("$BEV_FEATURES_ROOT") / "$BEV_DATA_SPLIT"
out_path = Path("$BEV_TOKEN_FILTER_FILE")
suffix = "_$BEV_FEATURE_TYPE.pt"

if not bev_root.exists():
    raise FileNotFoundError(f"BEV feature split directory does not exist: {bev_root}")

tokens = sorted({
    path.name[:-len(suffix)]
    for path in bev_root.rglob(f"*{suffix}")
})
if not tokens:
    raise RuntimeError(f"No BEV feature files matching *{suffix} under {bev_root}")

out_path.write_text("\\n".join(tokens) + "\\n")
print(f"Wrote {len(tokens)} BEV tokens to {out_path}")
PY
fi

TOKEN_FILTER_OVERRIDE=()
if [[ "$USE_BEV_TOKEN_FILTER" == "1" ]]; then
  TOKEN_FILTER_OVERRIDE+=(+scene_filter_token_file="$BEV_TOKEN_FILTER_FILE")
fi

LOG_DIR="$NAVSIM_EXP_ROOT/ke/$EXPERIMENT/$EXPERIMENT_UID"
export DRIVOR_TRAIN_OUTPUT_DIR="$LOG_DIR"
export DRIVOR_ROOT
export DECODER_BEV_LORA_RANK
export DRIVOR_EPOCH_PDMS_EVAL
export DRIVOR_EPOCH_PDMS_EVERY_N_EPOCHS
export DRIVOR_EPOCH_PDMS_TIMEOUT_SEC
export DRIVOR_EPOCH_PDMS_SCRIPT="${DRIVOR_EPOCH_PDMS_SCRIPT:-$DRIVOR_ROOT/scripts/evaluation/run_drivor_bev_decoder_evaluation.sh}"
export DRIVOR_EPOCH_PDMS_CUDA_VISIBLE_DEVICES="${DRIVOR_EPOCH_PDMS_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
export DRIVOR_EPOCH_PDMS_DECODER_BEV_LORA_RANK="${DRIVOR_EPOCH_PDMS_DECODER_BEV_LORA_RANK:-$DECODER_BEV_LORA_RANK}"
export DRIVOR_EPOCH_PDMS_EXPERIMENT_PREFIX="${DRIVOR_EPOCH_PDMS_EXPERIMENT_PREFIX:-$EXPERIMENT/$EXPERIMENT_UID}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/launcher.log") 2>&1

echo "=== DrivoR BEV decoder from-scratch training ==="
echo "Experiment     : $EXPERIMENT"
echo "Experiment UID : $EXPERIMENT_UID"
echo "W&B project    : $WANDB_PROJECT"
echo "W&B run        : $WANDB_RUN_NAME"
echo "Epochs         : $MAX_EPOCHS"
echo "GPUs           : $NUM_GPUS"
echo "CUDA devices   : $CUDA_VISIBLE_DEVICES"
echo "Batch size     : $BATCH_SIZE"
echo "Base LR        : $BASE_LR"
echo "Workers        : $NUM_WORKERS"
echo "Prefetch       : $PREFETCH_FACTOR"
echo "BEV root       : $BEV_FEATURES_ROOT"
echo "BEV type       : $BEV_FEATURE_TYPE"
if [[ "$USE_BEV_TOKEN_FILTER" == "1" ]]; then
  echo "Token filter   : $BEV_TOKEN_FILTER_FILE"
else
  echo "Token filter   : disabled (baseline navtrain split size)"
fi
echo "Decoder BEV    : init_gate=$DECODER_BEV_INIT_GATE lora_rank=$DECODER_BEV_LORA_RANK"
echo "Epoch PDMS     : enabled=$DRIVOR_EPOCH_PDMS_EVAL every=${DRIVOR_EPOCH_PDMS_EVERY_N_EPOCHS} eval_cuda=${DRIVOR_EPOCH_PDMS_CUDA_VISIBLE_DEVICES}"
echo "Checkpoint     : none"
echo "Freeze         : false"
echo "Log dir        : $LOG_DIR"
echo "================================================="

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent=drivoR \
  experiment_name="$EXPERIMENT" \
  experiment_uid="$EXPERIMENT_UID" \
  train_test_split=navtrain \
  cache_path=null \
  use_cache_without_dataset=false \
  force_cache_computation=false \
  train_ckpt_path=null \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  +trainer.params.devices="$NUM_GPUS" \
  trainer.params.strategy="$TRAINER_STRATEGY" \
  dataloader.params.prefetch_factor="$PREFETCH_FACTOR" \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers="$NUM_WORKERS" \
  agent.checkpoint_path="" \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="$BASE_LR" \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.config.use_bev_feature=true \
  agent.config.use_ray_score=true \
  agent.config.use_bev_in_scorer=false \
  agent.config.use_bev_in_decoder=true \
  agent.config.use_bev_residual_proposal_refiner=false \
  agent.config.freeze_pretrained_except_bev_scorer=false \
  agent.config.bev_feature_type="$BEV_FEATURE_TYPE" \
  agent.config.bev_channels=256 \
  agent.config.bev_features_root="$BEV_FEATURES_ROOT" \
  agent.config.bev_data_split="$BEV_DATA_SPLIT" \
  agent.config.decoder_bev.init_gate="$DECODER_BEV_INIT_GATE" \
  agent.config.decoder_bev.lora_rank="$DECODER_BEV_LORA_RANK" \
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
  "${TOKEN_FILTER_OVERRIDE[@]}" \
  +trainer.params.logger._target_=pytorch_lightning.loggers.WandbLogger \
  +trainer.params.logger.project="$WANDB_PROJECT" \
  +trainer.params.logger.name="$WANDB_RUN_NAME" \
  +trainer.params.logger.save_dir="$LOG_DIR" \
  +trainer.params.logger.id="$EXPERIMENT_UID" \
  +trainer.params.logger.offline=false \
  seed=2
