#!/bin/bash
# Fast controlled validation of idea_0002_01:
# frozen DrivoR -> proposals -> proposal-conditioned BEV world model
# -> gated trajectory refinement + gated scorer context.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="${DRIVOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/ws-frb/users/jingyuso/wenzhet}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/navsim_dataset}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$DRIVOR_ROOT:$DRIVOR_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA_ROOT}"
export NAVSIM_DEVKIT_ROOT="$DRIVOR_ROOT"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REFERENCE_ROOT/exp}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export WANDB_MODE="${WANDB_MODE:-online}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

VARIANT="${VARIANT:-proposal_world}"
case "$VARIANT" in
  proposal_world)
    USE_PROPOSAL_WORLD_REFINER=true
    USE_STATIC_BEV_REFINER=false
    ;;
  static_bev_refiner)
    USE_PROPOSAL_WORLD_REFINER=false
    USE_STATIC_BEV_REFINER=true
    ;;
  *)
    echo "ERROR: VARIANT must be proposal_world or static_bev_refiner, got: $VARIANT" >&2
    exit 1
    ;;
esac

EXPERIMENT="${EXPERIMENT:-Aug18-idea0002-01-${VARIANT}-fast}"
EXPERIMENT_UID="${EXPERIMENT_UID:-$(date +%m.%d_%H.%M)}"
BASELINE_CKPT="${BASELINE_CKPT:-$REFERENCE_ROOT/weights/checkpoints/drivor_Nav1_25epochs.pth}"
BEV_FEATURES_ROOT="${BEV_FEATURES_ROOT:-$WORKSPACE_ROOT/navsim_bev_feature/exports_pretrained}"
CACHE_PATH="${CACHE_PATH:-$REFERENCE_ROOT/exp/navsim_cache_future_bev_oracle_decoder_scorer}"

TOKEN_SOURCE_FILE="${TOKEN_SOURCE_FILE:-$REFERENCE_ROOT/exp/bev_feature_tokens/trainval_decoder_neck_tokens_full_metric_covered_after_cache.txt}"
TOKEN_COUNT="${TOKEN_COUNT:-2048}"
TOKEN_SEED="${TOKEN_SEED:-2}"
FIXED_TOKEN_FILE="${FIXED_TOKEN_FILE:-$REFERENCE_ROOT/exp/idea0002_fast/fixed_tokens_seed${TOKEN_SEED}_n${TOKEN_COUNT}.txt}"

NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-4}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-64}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
BASE_LR="${BASE_LR:-1e-4}"
WANDB_PROJECT="${WANDB_PROJECT:-drivor-world-model-fast-validation}"
USE_WANDB="${USE_WANDB:-1}"

if [[ "$NUM_WORKERS" == "0" ]]; then
  PREFETCH_FACTOR=null
fi

WORLD_LAYERS="${WORLD_LAYERS:-2}"
WORLD_HEADS="${WORLD_HEADS:-4}"
WORLD_FFN_DIM="${WORLD_FFN_DIM:-512}"
WORLD_ROLLOUT_STEPS="${WORLD_ROLLOUT_STEPS:-1}"
PROPOSAL_CHUNK_SIZE="${PROPOSAL_CHUNK_SIZE:-8}"
INIT_REFINE_GATE="${INIT_REFINE_GATE:-0.0}"
INIT_SCORE_GATE="${INIT_SCORE_GATE:-0.0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_CKPT" ]]; then
  echo "ERROR: baseline checkpoint not found: $BASELINE_CKPT" >&2
  exit 1
fi
if [[ ! -d "$CACHE_PATH" ]]; then
  echo "ERROR: cache not found: $CACHE_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/build_fixed_token_subset.py" \
  --source "$TOKEN_SOURCE_FILE" \
  --output "$FIXED_TOKEN_FILE" \
  --count "$TOKEN_COUNT" \
  --seed "$TOKEN_SEED"

OUTPUT_DIR="$NAVSIM_EXP_ROOT/ke/$EXPERIMENT/$EXPERIMENT_UID"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/launcher.log") 2>&1

if [[ "$NUM_GPUS" == "1" ]]; then
  TRAINER_STRATEGY="auto"
else
  TRAINER_STRATEGY="ddp_find_unused_parameters_true"
fi

echo "=== idea_0002_01 fast validation ==="
echo "Code branch    : $(git -C "$DRIVOR_ROOT" branch --show-current)"
echo "Code commit    : $(git -C "$DRIVOR_ROOT" rev-parse HEAD)"
echo "Experiment     : $EXPERIMENT/$EXPERIMENT_UID"
echo "Variant        : $VARIANT"
echo "Fixed tokens   : $FIXED_TOKEN_FILE ($TOKEN_COUNT, seed=$TOKEN_SEED)"
echo "Training       : epochs=$MAX_EPOCHS batches=$LIMIT_TRAIN_BATCHES val_batches=$LIMIT_VAL_BATCHES"
echo "GPU/batch      : devices=$CUDA_VISIBLE_DEVICES batch=$BATCH_SIZE accumulate=$ACCUMULATE_GRAD_BATCHES"
echo "World model    : layers=$WORLD_LAYERS heads=$WORLD_HEADS rollout=$WORLD_ROLLOUT_STEPS chunk=$PROPOSAL_CHUNK_SIZE"
echo "Gates          : refine=$INIT_REFINE_GATE score=$INIT_SCORE_GATE"
echo "======================================="

LOGGER_OVERRIDES=()
if [[ "$USE_WANDB" == "1" ]]; then
  LOGGER_OVERRIDES+=(+trainer.params.logger._target_=pytorch_lightning.loggers.WandbLogger)
  LOGGER_OVERRIDES+=(+trainer.params.logger.project="$WANDB_PROJECT")
  LOGGER_OVERRIDES+=(+trainer.params.logger.name="$EXPERIMENT/$EXPERIMENT_UID")
  LOGGER_OVERRIDES+=(+trainer.params.logger.save_dir="$OUTPUT_DIR")
  LOGGER_OVERRIDES+=(+trainer.params.logger.offline=false)
else
  LOGGER_OVERRIDES+=(+trainer.params.logger._target_=pytorch_lightning.loggers.CSVLogger)
  LOGGER_OVERRIDES+=(+trainer.params.logger.save_dir="$OUTPUT_DIR")
  LOGGER_OVERRIDES+=(+trainer.params.logger.name=csv_logs)
  LOGGER_OVERRIDES+=(+trainer.params.logger.version=0)
fi

"$PYTHON_BIN" -u "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent=drivoR \
  experiment_name="$EXPERIMENT" \
  experiment_uid="$EXPERIMENT_UID" \
  train_test_split=navtrain \
  split=trainval \
  cache_path="$CACHE_PATH" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  +scene_filter_token_file="$FIXED_TOKEN_FILE" \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  +trainer.params.devices="$NUM_GPUS" \
  trainer.params.strategy="$TRAINER_STRATEGY" \
  trainer.params.precision=bf16-mixed \
  trainer.params.limit_train_batches="$LIMIT_TRAIN_BATCHES" \
  trainer.params.limit_val_batches="$LIMIT_VAL_BATCHES" \
  trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  trainer.params.log_every_n_steps="$LOG_EVERY_N_STEPS" \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers="$NUM_WORKERS" \
  dataloader.params.prefetch_factor="$PREFETCH_FACTOR" \
  agent.checkpoint_path="$BASELINE_CKPT" \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="$BASE_LR" \
  agent.config.use_bev_feature=true \
  agent.config.use_bev_in_decoder=false \
  agent.config.use_bev_in_scorer=false \
  agent.config.use_bev_residual_proposal_refiner="$USE_STATIC_BEV_REFINER" \
  agent.config.use_proposal_world_refiner="$USE_PROPOSAL_WORLD_REFINER" \
  agent.config.use_privileged_future_bev=false \
  agent.config.freeze_pretrained_except_bev_scorer=true \
  agent.config.bev_feature_type=decoder_neck \
  agent.config.bev_channels=256 \
  agent.config.bev_features_root="$BEV_FEATURES_ROOT" \
  agent.config.bev_data_split=trainval \
  agent.config.proposal_world_refiner.num_layers="$WORLD_LAYERS" \
  agent.config.proposal_world_refiner.num_heads="$WORLD_HEADS" \
  agent.config.proposal_world_refiner.ffn_dim="$WORLD_FFN_DIM" \
  agent.config.proposal_world_refiner.rollout_steps="$WORLD_ROLLOUT_STEPS" \
  agent.config.proposal_world_refiner.proposal_chunk_size="$PROPOSAL_CHUNK_SIZE" \
  agent.config.proposal_world_refiner.init_refine_gate="$INIT_REFINE_GATE" \
  agent.config.proposal_world_refiner.init_score_gate="$INIT_SCORE_GATE" \
  agent.config.bev_residual_proposal_refiner.num_layers="$WORLD_LAYERS" \
  agent.config.bev_residual_proposal_refiner.num_heads="$WORLD_HEADS" \
  agent.config.bev_residual_proposal_refiner.init_alpha="$INIT_REFINE_GATE" \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.model_weights="$REFERENCE_ROOT/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors" \
  agent.config.lidar_backbone.model_weights="$REFERENCE_ROOT/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors" \
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
  "${LOGGER_OVERRIDES[@]}" \
  seed=2
