#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Configuration
WORKSPACE_ROOT="/ros2_ws"
POLICY_DIR="$WORKSPACE_ROOT/third_party/3D-Diffusion-Policy"
POLICY_DIR2="$WORKSPACE_ROOT/third_party/ManiFlow_Policy"
VENV_DIR="$WORKSPACE_ROOT/venv"

# 1. Validation
if [ ! -d "$WORKSPACE_ROOT" ]; then
    echo "Error: The directory $WORKSPACE_ROOT does not exist."
    exit 1
fi

# 2. Colcon Build
echo "Starting colcon build process..."
cd "$WORKSPACE_ROOT"

colcon build

echo "Creating virtual environment in $VENV_DIR..."
mkdir -p "$VENV_DIR"
python -m venv "$VENV_DIR" --clear
python -m venv --system-site-packages "$VENV_DIR"
# 4. Activation and Installation
echo "Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"

# Ensure pip is up to date to avoid build isolation issues
pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

cd "$POLICY_DIR"

pip install -r requirements.txt

# Install the package in editable mode
cd 3D-Diffusion-Policy && pip install -v -e . && cd ..

cd third_party/pytorch3d_simplified && pip install --no-build-isolation -e . && cd ../..

pip install kinpy
pip install rerun-sdk

# necessary to avoid flash-attn warning
pip install ninja
export TORCH_CUDA_ARCH_LIST="8.9"
MAX_JOBS=8 python -m pip -v install flash-attn --no-build-isolation

cd "$POLICY_DIR2"

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"  --no-build-isolation

pip install -r requirements.txt

# Install the package in editable mode
cd ManiFlow && pip install -v -e . && cd ..

touch /ros2_ws/venv/COLCON_IGNORE
export PYTHONPATH=$PYTHONPATH:$VENV_DIR/lib/python3.12/site-packages

echo "Success: Build and installation complete."
