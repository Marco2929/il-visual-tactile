#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo " Starting Sequential Ablation Studies on Recorded Action"
echo " Dataset: /ros2_ws/data/zarr_datasets/foil_lifting_box_recorded_action.zarr"
echo " Baseline: Run 0015 (Wrist RGB + Point Cloud)"
echo "======================================================================"

# Run 1: Wrist-Only (Point Cloud disabled)
echo -e "\n\033[32m>>> [1/3] Starting Run 0016: Wrist RGB Only (Point Cloud disabled) <<<\033[0m"
bash "${SCRIPT_DIR}/train_policy.sh" maniflow_multimodal_policy foil_lifting_wrist_recorded_action 0016_abl_wrist_only 0 0 \
    +policy.pc_conditioned=false

# Run 2: Wrist + Front RGB (Dual 2D View, Point Cloud disabled)
echo -e "\n\033[32m>>> [2/3] Starting Run 0017: Wrist + Front RGB (Dual-View, no Point Cloud) <<<\033[0m"
bash "${SCRIPT_DIR}/train_policy.sh" maniflow_multimodal_policy foil_lifting_wrist_recorded_action 0017_abl_wrist_front 0 0 \
    +policy.pc_conditioned=false \
    "policy.rgb_keys=['wrist_rgb','front_rgb']" \
    task.dataset.use_front_rgb=true

# Run 3: Full Sensor Suite (Wrist RGB + Front RGB + Point Cloud)
echo -e "\n\033[32m>>> [3/3] Starting Run 0018: Full Suite (Wrist + Front RGB + Point Cloud) <<<\033[0m"
bash "${SCRIPT_DIR}/train_policy.sh" maniflow_multimodal_policy foil_lifting_wrist_recorded_action 0018_abl_all_visual 0 0 \
    "policy.rgb_keys=['wrist_rgb','front_rgb']" \
    task.dataset.use_front_rgb=true

echo "======================================================================"
echo " All ablation trainings completed successfully!"
echo " Checkpoints saved in /ros2_ws/data/outputs/"
echo "======================================================================"

