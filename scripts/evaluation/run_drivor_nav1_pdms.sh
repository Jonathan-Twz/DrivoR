#!/bin/bash
# DrivoR NAVSIM-v1 PDMS Evaluation Script
# 使用 WoTE 的数据集路径 + DrivoR 的代码和权重

set -e

# === 环境变量配置 ===
# 数据集路径 (复用 WoTE 已下载的数据)
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/wenzhe/wm_ws/WoTE/dataset/maps"
export OPENSCENE_DATA_ROOT="/home/wenzhe/wm_ws/WoTE/dataset"

# DrivoR 代码和实验路径
export NAVSIM_DEVKIT_ROOT="/home/wenzhe/wm_ws/DrivoR"
export NAVSIM_EXP_ROOT="/home/wenzhe/wm_ws/DrivoR/exp"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

# === 创建必要目录 ===
mkdir -p "$NAVSIM_EXP_ROOT"

# === 评估参数 ===
CHECKPOINT="$NAVSIM_DEVKIT_ROOT/weights/drivor_Nav1_25epochs.pth"
EXPERIMENT="drivoR_nav1_eval"

echo "========================================="
echo "DrivoR NAVSIM-v1 PDMS Evaluation"
echo "========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Data root:  $OPENSCENE_DATA_ROOT"
echo "Maps root:  $NUPLAN_MAPS_ROOT"
echo "Exp root:   $NAVSIM_EXP_ROOT"
echo "========================================="

# === Step 1: 检查 metric cache ===
METRIC_CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache"
if [ ! -d "$METRIC_CACHE_PATH" ] || [ -z "$(ls -A $METRIC_CACHE_PATH 2>/dev/null)" ]; then
    echo "[Step 1/2] Generating metric cache for navtest..."
    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
        train_test_split=navtest \
        cache.cache_path="$METRIC_CACHE_PATH"
else
    echo "[Step 1/2] Metric cache already exists at $METRIC_CACHE_PATH, skipping..."
fi

# === Step 2: 运行 PDMS 评估 ===
echo "[Step 2/2] Running PDMS evaluation on navtest split..."
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    train_test_split=navtest \
    agent=drivoR \
    agent.checkpoint_path="$CHECKPOINT" \
    experiment_name="$EXPERIMENT" \
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
    agent.config.comfort=2

echo "========================================="
echo "Evaluation complete!"
echo "Results saved to: $NAVSIM_EXP_ROOT"
echo "========================================="
