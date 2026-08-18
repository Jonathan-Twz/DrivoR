#!/bin/bash
# Oracle experiment: train DrivoR with privileged future BEV tokens injected
# into both the trajectory decoder and the scorer.  The future BEV tensors are
# training/validation-only targets; normal inference still uses current inputs.
#
# Usage:
#   bash scripts/training/run_drivor_future_bev_oracle.sh <baseline_ckpt> [experiment_name] [max_epochs]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$DRIVOR_ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/navsim_dataset}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python}"
BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$WORKSPACE_ROOT/navsim_bev_feature/exports_pretrained}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor-matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

DEFAULT_BASELINE_CKPT="$DRIVOR_ROOT/weights/checkpoints/drivor_Nav1_25epochs.pth"
BASELINE_CKPT="${1:-$DEFAULT_BASELINE_CKPT}"
EXPERIMENT="${2:-future-bev-oracle-decoder-scorer}"
MAX_EPOCHS="${3:-30}"
EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"

NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BASE_LR="${BASE_LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp_find_unused_parameters_true}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-trainval}"
SCENE_FILTER_MAX_SCENES="${SCENE_FILTER_MAX_SCENES:-}"

# Default to on-the-fly samples.  Passing a real cache_path makes this repo's
# Dataset eagerly cache every token before training; with future BEV targets
# that is a multi-day preprocessing job.  Set CACHE_PATH explicitly only when
# intentionally building/reusing a small debug cache.
CACHE_PATH="${CACHE_PATH:-null}"
USE_CACHE_WITHOUT_DATASET="${USE_CACHE_WITHOUT_DATASET:-false}"
FORCE_CACHE_COMPUTATION="${FORCE_CACHE_COMPUTATION:-false}"

BEV_DATA_SPLIT="${BEV_DATA_SPLIT:-trainval}"
BEV_FEATURE_TYPE="${BEV_FEATURE_TYPE:-decoder_neck}"
FUTURE_BEV_NUM_STEPS="${FUTURE_BEV_NUM_STEPS:-4}"
FUTURE_BEV_STRIDE="${FUTURE_BEV_STRIDE:-1}"
USE_FUTURE_BEV_IN_DECODER="${USE_FUTURE_BEV_IN_DECODER:-true}"
USE_FUTURE_BEV_IN_SCORER="${USE_FUTURE_BEV_IN_SCORER:-true}"
DECODER_BEV_INIT_GATE="${DECODER_BEV_INIT_GATE:-0.0}"
DECODER_BEV_LORA_RANK="${DECODER_BEV_LORA_RANK:-16}"
SCORER_BEV_INIT_GATE="${SCORER_BEV_INIT_GATE:-0.0}"
SCORER_BEV_LORA_RANK="${SCORER_BEV_LORA_RANK:-16}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-drivor-future-bev-oracle}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$NUM_WORKERS" == "0" && "$PREFETCH_FACTOR" == "1" ]]; then
  PREFETCH_FACTOR=null
fi

BEV_TOKEN_FILTER_DEFAULT="$NAVSIM_EXP_ROOT/bev_feature_tokens/${BEV_DATA_SPLIT}_${BEV_FEATURE_TYPE}_tokens_full.txt"
BEV_TOKEN_FILTER_FILE="${BEV_TOKEN_FILTER_FILE:-$BEV_TOKEN_FILTER_DEFAULT}"
if [[ -z "${BEV_TOKEN_FILTER_FILE// }" ]] || [[ "$BEV_TOKEN_FILTER_FILE" == *'/PATH/'* ]] || [[ "$BEV_TOKEN_FILTER_FILE" == *'PATH/TO'* ]]; then
  BEV_TOKEN_FILTER_FILE="$BEV_TOKEN_FILTER_DEFAULT"
fi

LAUNCHER_LOG_DIR="${NAVSIM_EXP_ROOT}/ke/${EXPERIMENT}/${EXPERIMENT_UID}"
LAUNCHER_LOG_FILE="${LAUNCHER_LOG_DIR}/launcher.log"
mkdir -p "$LAUNCHER_LOG_DIR"
exec > >(tee -a "$LAUNCHER_LOG_FILE") 2>&1

trap 'rc=$?; echo ""; echo "=== Launcher finished ==="; echo "Exit code  : $rc"; echo "Finish time: $(date -Is)"; exit $rc' EXIT

echo "=== DrivoR Future BEV Oracle ==="
echo "Launcher log : $LAUNCHER_LOG_FILE"
echo "DrivoR root  : $DRIVOR_ROOT"
echo "Data root    : $DATA_ROOT"
echo "BEV root     : $BEV_FEATURES_ROOT/$BEV_DATA_SPLIT"
echo "Baseline ckpt: $BASELINE_CKPT"
echo "Experiment   : $EXPERIMENT / $EXPERIMENT_UID"
echo "Future BEV   : steps=$FUTURE_BEV_NUM_STEPS stride=$FUTURE_BEV_STRIDE decoder=$USE_FUTURE_BEV_IN_DECODER scorer=$USE_FUTURE_BEV_IN_SCORER"
echo "Current BEV  : decoder=true scorer=true"
echo "BEV gates    : decoder=$DECODER_BEV_INIT_GATE scorer=$SCORER_BEV_INIT_GATE ranks decoder=$DECODER_BEV_LORA_RANK scorer=$SCORER_BEV_LORA_RANK"
echo "W&B          : use=$USE_WANDB mode=$WANDB_MODE project=$WANDB_PROJECT"
echo "GPUs/Batch   : gpus=$NUM_GPUS batch_per_gpu=$BATCH_SIZE workers=$NUM_WORKERS prefetch=$PREFETCH_FACTOR"
echo "Cache        : path=$CACHE_PATH use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET force=$FORCE_CACHE_COMPUTATION"
echo "================================"

HYDRA_OVERRIDES=()
if [[ -n "$SCENE_FILTER_MAX_SCENES" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="$SCENE_FILTER_MAX_SCENES")
fi

if [[ ! -f "$BEV_TOKEN_FILTER_FILE" ]]; then
  echo "Generating BEV token filter from $BEV_FEATURES_ROOT/$BEV_DATA_SPLIT ..."
  mkdir -p "$(dirname "$BEV_TOKEN_FILTER_FILE")"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
bev_root = Path("$BEV_FEATURES_ROOT") / "$BEV_DATA_SPLIT"
out_path = Path("$BEV_TOKEN_FILTER_FILE")
suffix = "_$BEV_FEATURE_TYPE.pt"
if not bev_root.exists():
    raise FileNotFoundError(f"BEV feature split directory does not exist: {bev_root}")
tokens = sorted({path.name[:-len(suffix)] for path in bev_root.rglob(f"*{suffix}")})
if not tokens:
    raise RuntimeError(f"No BEV feature files matching *{suffix} under {bev_root}")
out_path.write_text("\\n".join(tokens) + "\\n")
print(f"Wrote {len(tokens)} BEV tokens to {out_path}")
PY
fi
HYDRA_OVERRIDES+=(+scene_filter_token_file="$BEV_TOKEN_FILTER_FILE")

WANDB_RUN_NAME="${EXPERIMENT}/${EXPERIMENT_UID}"
if [[ "$USE_WANDB" == "1" ]]; then
  HYDRA_OVERRIDES+=(+trainer.params.logger._target_=pytorch_lightning.loggers.WandbLogger)
  HYDRA_OVERRIDES+=(+trainer.params.logger.project="$WANDB_PROJECT")
  HYDRA_OVERRIDES+=(+trainer.params.logger.name="$WANDB_RUN_NAME")
  HYDRA_OVERRIDES+=(+trainer.params.logger.save_dir="$NAVSIM_EXP_ROOT/ke/$EXPERIMENT/$EXPERIMENT_UID")
  HYDRA_OVERRIDES+=(+trainer.params.logger.id="$EXPERIMENT_UID")
  if [[ "$WANDB_MODE" == "offline" ]]; then
    HYDRA_OVERRIDES+=(+trainer.params.logger.offline=true)
  else
    HYDRA_OVERRIDES+=(+trainer.params.logger.offline=false)
  fi
  export WANDB_MODE
  if [[ -n "$WANDB_ENTITY" ]]; then
    HYDRA_OVERRIDES+=(+trainer.params.logger.entity="$WANDB_ENTITY")
  fi
fi

PYTHONUNBUFFERED=1 \
"$PYTHON_BIN" -u \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent=drivoR \
  experiment_name="$EXPERIMENT" \
  experiment_uid="$EXPERIMENT_UID" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  split="$SPLIT" \
  cache_path="$CACHE_PATH" \
  use_cache_without_dataset="$USE_CACHE_WITHOUT_DATASET" \
  force_cache_computation="$FORCE_CACHE_COMPUTATION" \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  +trainer.params.devices="$NUM_GPUS" \
  trainer.params.strategy="$TRAINER_STRATEGY" \
  trainer.params.limit_train_batches="$LIMIT_TRAIN_BATCHES" \
  trainer.params.limit_val_batches="$LIMIT_VAL_BATCHES" \
  dataloader.params.prefetch_factor="$PREFETCH_FACTOR" \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers="$NUM_WORKERS" \
  agent.checkpoint_path="$BASELINE_CKPT" \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="$BASE_LR" \
  agent.config.use_bev_feature=true \
  agent.config.use_privileged_future_bev=true \
  agent.config.validate_with_privileged_future_bev=true \
  agent.config.future_bev_num_steps="$FUTURE_BEV_NUM_STEPS" \
  agent.config.future_bev_stride="$FUTURE_BEV_STRIDE" \
  agent.config.use_future_bev_in_decoder="$USE_FUTURE_BEV_IN_DECODER" \
  agent.config.use_future_bev_in_scorer="$USE_FUTURE_BEV_IN_SCORER" \
  agent.config.use_bev_in_decoder=true \
  agent.config.use_bev_in_scorer=true \
  agent.config.freeze_pretrained_except_bev_scorer=true \
  agent.config.bev_feature_type="$BEV_FEATURE_TYPE" \
  agent.config.bev_channels=256 \
  agent.config.bev_features_root="$BEV_FEATURES_ROOT" \
  agent.config.bev_data_split="$BEV_DATA_SPLIT" \
  agent.config.decoder_bev.init_gate="$DECODER_BEV_INIT_GATE" \
  agent.config.decoder_bev.lora_rank="$DECODER_BEV_LORA_RANK" \
  agent.config.scorer_bev.init_gate="$SCORER_BEV_INIT_GATE" \
  agent.config.scorer_bev.lora_rank="$SCORER_BEV_LORA_RANK" \
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
  "${HYDRA_OVERRIDES[@]}" \
  seed=2
