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
# Defaults are for the guppy workspace; override env vars when running elsewhere.

set -euo pipefail

# Root paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$DRIVOR_ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/navsim_dataset}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python}"
BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$WORKSPACE_ROOT/navsim_bev_feature/exports_pretrained}"

export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT}"
# IMPORTANT: force the DrivoR devkit so we run DrivoR's run_training_full.py, not WoTE's.
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVOR_ROOT/exp}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

# Terminal input
BASELINE_CKPT="${1:?Usage: $0 <baseline_checkpoint.(ckpt|pth)> [experiment_name] [max_epochs]}"
EXPERIMENT="${2:-training_drivor_bev_scorer_phase1}"
MAX_EPOCHS="${3:-10}"

EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"
NUM_GPUS="${NUM_GPUS:-2}"
# 0.46 GB per batch
BATCH_SIZE="${BATCH_SIZE:-16}"
BASE_LR="${BASE_LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
BEV_DATA_SPLIT="${BEV_DATA_SPLIT:-trainval}"
BEV_FEATURE_TYPE="${BEV_FEATURE_TYPE:-decoder_neck}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-trainval}"
# BEV phase-1 freezes most weights; some trainable params may not appear in every step's loss graph.
# Plain "ddp" then fails with: unused parameters in DDP (see PyTorch DDP find_unused_parameters).
# TRAINER_STRATEGY="${TRAINER_STRATEGY:-auto}"
# TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"
TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp_find_unused_parameters_true}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
SCENE_FILTER_MAX_SCENES="${SCENE_FILTER_MAX_SCENES:-}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/navsim_cache_nommcv_full}"
USE_CACHE_WITHOUT_DATASET="${USE_CACHE_WITHOUT_DATASET:-true}"
FORCE_CACHE_COMPUTATION="${FORCE_CACHE_COMPUTATION:-false}"

# W&B logging
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-drivor-bev-scorer}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

BEV_TOKEN_FILTER_DEFAULT="$NAVSIM_EXP_ROOT/bev_feature_tokens/${BEV_DATA_SPLIT}_${BEV_FEATURE_TYPE}_tokens_full.txt"
BEV_TOKEN_FILTER_FILE="${BEV_TOKEN_FILTER_FILE:-$BEV_TOKEN_FILTER_DEFAULT}"

# Doc placeholders (e.g. /PATH/TO/drivoR/...) or empty env break mkdir; fall back to DrivoR/exp/.
if [[ -z "${BEV_TOKEN_FILTER_FILE// }" ]] || [[ "$BEV_TOKEN_FILTER_FILE" == *'/PATH/'* ]] || [[ "$BEV_TOKEN_FILTER_FILE" == *'PATH/TO'* ]]; then
  echo "WARNING: BEV_TOKEN_FILTER_FILE is empty or looks like a placeholder; using $BEV_TOKEN_FILTER_DEFAULT" >&2
  BEV_TOKEN_FILTER_FILE="$BEV_TOKEN_FILTER_DEFAULT"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/drivoR/bin/python or activate conda env drivoR." >&2
  exit 1
fi

if [[ "$NUM_WORKERS" == "0" && "$PREFETCH_FACTOR" == "1" ]]; then
  PREFETCH_FACTOR=null
fi

# ─── Launcher-level logging: duplicate ALL stdout+stderr to launcher.log ───
LAUNCHER_LOG_DIR="${NAVSIM_EXP_ROOT}/ke/${EXPERIMENT}/${EXPERIMENT_UID}"
LAUNCHER_LOG_FILE="${LAUNCHER_LOG_DIR}/launcher.log"
mkdir -p "$LAUNCHER_LOG_DIR"
exec > >(tee -a "$LAUNCHER_LOG_FILE") 2>&1

on_exit() {
  local rc=$?
  echo ""
  echo "=== Launcher finished ==="
  echo "Exit code  : $rc"
  echo "Finish time: $(date -Is)"
  exit $rc
}
trap on_exit EXIT

echo "=== DrivoR BEV Scorer Phase-1 Fine-tuning ==="
echo "Launcher log : $LAUNCHER_LOG_FILE"
echo "Launch time  : $(date -Is)"
echo "Host         : $(hostname)"
echo "DrivoR root : $DRIVOR_ROOT"
echo "Data root   : $DATA_ROOT"
echo "BEV root    : $BEV_FEATURES_ROOT"
echo "BEV type    : $BEV_FEATURE_TYPE"
echo "Python      : $PYTHON_BIN"
echo "Baseline ckpt: $BASELINE_CKPT"
echo "Experiment   : $EXPERIMENT"
echo "Experiment UID: $EXPERIMENT_UID"
echo "Max epochs   : $MAX_EPOCHS"
echo "GPUs         : $NUM_GPUS"
echo "PL strategy  : $TRAINER_STRATEGY"
echo "NCCL P2P/IB  : $NCCL_P2P_DISABLE / $NCCL_IB_DISABLE"
echo "Batch size   : $BATCH_SIZE"
echo "Base LR      : $BASE_LR"
echo "Workers      : $NUM_WORKERS"
echo "Prefetch     : $PREFETCH_FACTOR"
echo "Split        : $TRAIN_TEST_SPLIT / $SPLIT"
echo "Max scenes   : ${SCENE_FILTER_MAX_SCENES:-default}"
echo "Cache path   : $CACHE_PATH"
echo "Use cache    : $USE_CACHE_WITHOUT_DATASET (force_build=$FORCE_CACHE_COMPUTATION)"
echo "Token filter : $BEV_TOKEN_FILTER_FILE"
WANDB_RUN_NAME="${EXPERIMENT}/${EXPERIMENT_UID}"
echo "W&B logger   : $USE_WANDB ($WANDB_MODE) run=$WANDB_RUN_NAME"
echo "============================================="

HYDRA_OVERRIDES=()
if [[ -n "$SCENE_FILTER_MAX_SCENES" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="$SCENE_FILTER_MAX_SCENES")
fi

if [[ ! -f "$BEV_TOKEN_FILTER_FILE" ]]; then
  echo "Generating BEV token filter from $BEV_FEATURES_ROOT/$BEV_DATA_SPLIT ..."
  if ! mkdir -p "$(dirname "$BEV_TOKEN_FILTER_FILE")" 2>/dev/null; then
    echo "WARNING: cannot mkdir '$(dirname "$BEV_TOKEN_FILTER_FILE")' (invalid BEV_TOKEN_FILTER_FILE in env?). Using $BEV_TOKEN_FILTER_DEFAULT" >&2
    BEV_TOKEN_FILTER_FILE="$BEV_TOKEN_FILTER_DEFAULT"
    mkdir -p "$(dirname "$BEV_TOKEN_FILTER_FILE")"
  fi
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

HYDRA_OVERRIDES+=(+scene_filter_token_file="$BEV_TOKEN_FILTER_FILE")

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
  +trainer.params.profiler="simple" \
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
  agent.config.use_ray_score=false \
  agent.config.freeze_pretrained_except_bev_scorer=true \
  agent.config.bev_feature_type="$BEV_FEATURE_TYPE" \
  agent.config.bev_channels=256 \
  agent.config.bev_features_root="$BEV_FEATURES_ROOT" \
  agent.config.bev_data_split="$BEV_DATA_SPLIT" \
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
