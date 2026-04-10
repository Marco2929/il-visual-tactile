#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Configuration
WORKSPACE_ROOT="/ros2_ws"
# POLICY_DIR="$WORKSPACE_ROOT/src/imitation_grasping/imitation_grasping/third_party/3D-Diffusion-Policy"
VENV_DIR="$WORKSPACE_ROOT/src/il_main/il/venv"

# 1. Validation
if [ ! -d "$WORKSPACE_ROOT" ]; then
    echo "Error: The directory $WORKSPACE_ROOT does not exist."
    exit 1
fi

# 2. Colcon Build
echo "Starting colcon build process..."
cd "$WORKSPACE_ROOT"

colcon build

# # 3. Environment Setup
# # Check if the deep third_party directory exists before proceeding
# if [ ! -d "$POLICY_DIR" ]; then
#     echo "Error: The policy directory $POLICY_DIR does not exist."
#     exit 1
# fi

# cd "$POLICY_DIR"

echo "Creating virtual environment in $VENV_DIR..."
mkdir -p "$VENV_DIR"
python -m venv "$VENV_DIR" --clear
python -m venv --system-site-packages "$VENV_DIR"
# 4. Activation and Installation
echo "Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"

# Ensure pip is up to date to avoid build isolation issues
pip install --upgrade pip

pip install dex-retargeting pinocchio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

touch /ros2_ws/src/il_main/il/venv/COLCON_IGNORE
export PYTHONPATH=$PYTHONPATH:/ros2_ws/src/il_main/il/venv/lib/python3.12/site-packages


