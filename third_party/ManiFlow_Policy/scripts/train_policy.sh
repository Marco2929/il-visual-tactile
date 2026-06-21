#!/bin/bash

# 1. Capture absolute paths immediately before any 'cd' commands
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="/ros2_ws/data/outputs/${exp_name}_seed${seed}"
                             # ls 
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

# Print configuration
echo -e "\033[33mTask: ${task_name}\033[0m"
echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
echo -e "\033[33mTrain: ${train}, Eval: ${eval}\033[0m"

# 2. Export environment variables and explicitly set PYTHONPATH
export PYTHONPATH="${ROOT_DIR}/ManiFlow:${PYTHONPATH}"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 3. Safely navigate to the target directory using absolute paths
cd "${ROOT_DIR}/ManiFlow/maniflow/workspace"

# Execute training
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}

