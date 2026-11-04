#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Activate Python venv with ManiFlow & torch
source /ros2_ws/venv/bin/activate

echo "=========================================================="
echo " Starting Relative Action Ablation Matrix (3 runs, 80 ep) "
echo " Dataset: /ros2_ws/data/zarr_datasets/foil_lifting_box_recorded_action.zarr"
echo " Working directory: $(pwd)"
echo "=========================================================="

echo ""
echo "[1/3] Starting: abl_rel_wrist_pc (0019_rel_wrist_pc)..."
bash scripts/train_policy.sh ablation/abl_rel_wrist_pc ablation/abl_rel_wrist_pc 0019_rel_wrist_pc 0 0

echo ""
echo "[2/3] Starting: abl_rel_wrist (0020_rel_wrist_only)..."
bash scripts/train_policy.sh ablation/abl_rel_wrist ablation/abl_rel_wrist 0020_rel_wrist_only 0 0

echo ""
echo "[3/3] Starting: abl_rel_wrist_front (0021_rel_wrist_front)..."
bash scripts/train_policy.sh ablation/abl_rel_wrist_front ablation/abl_rel_wrist_front 0021_rel_wrist_front 0 0

echo ""
echo "=========================================================="
echo " All 3 relative ablation runs completed successfully!     "
echo "=========================================================="
