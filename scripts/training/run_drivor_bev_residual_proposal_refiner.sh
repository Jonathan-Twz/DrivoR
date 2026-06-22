#!/usr/bin/env bash
# Phase-1 fine-tuning for the post-decoder BEV residual proposal refiner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export USE_BEV_IN_SCORER="${USE_BEV_IN_SCORER:-false}"
export USE_BEV_IN_DECODER="${USE_BEV_IN_DECODER:-false}"
export USE_BEV_RESIDUAL_PROPOSAL_REFINER="${USE_BEV_RESIDUAL_PROPOSAL_REFINER:-true}"
export RESIDUAL_REFINER_NUM_LAYERS="${RESIDUAL_REFINER_NUM_LAYERS:-1}"
export RESIDUAL_REFINER_NUM_HEADS="${RESIDUAL_REFINER_NUM_HEADS:-1}"
export RESIDUAL_REFINER_INIT_ALPHA="${RESIDUAL_REFINER_INIT_ALPHA:-0.0}"
export RESIDUAL_REFINER_DROPOUT="${RESIDUAL_REFINER_DROPOUT:-0.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-drivor-bev-residual-proposal-refiner}"

BASELINE_CKPT="${1:-$DRIVOR_ROOT/weights/checkpoints/drivor_Nav1_25epochs.pth}"
EXPERIMENT="${2:-Jun22-golduck-4gpu-0initalpha-bev-residual-proposal-refiner}"
MAX_EPOCHS="${3:-30}"

exec bash "$SCRIPT_DIR/run_drivor_bev_phase1.sh" \
  "$BASELINE_CKPT" "$EXPERIMENT" "$MAX_EPOCHS"
