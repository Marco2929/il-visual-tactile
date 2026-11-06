#!/bin/bash
# ============================================================================
# Agentless Pull-Worker for Colleague/Neighbor PCs WITHOUT openssh-server
# ============================================================================
# Run this on the neighbor PC (which already has SSH client access to master):
#   ssh mm@192.168.56.100 "cat /home/mm/dev/il-visual-tactile/scripts/remote_pull_worker.sh" | bash
# ============================================================================
set -e

MASTER_HOST="${MASTER_HOST:-mm@192.168.56.100}"
MASTER_REPO="/home/mm/dev/il-visual-tactile"
WORKER_NAME="${WORKER_NAME:-$(hostname)}"

echo "=========================================================="
echo " Pull-Worker Starting on: ${WORKER_NAME}"
echo " Master Queue Host      : ${MASTER_HOST}"
echo "=========================================================="

echo ""
echo "[1] Checking local disks on ${WORKER_NAME}:"
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,MODEL
echo ""
df -h -x tmpfs -x devtmpfs -x squashfs -x efivarfs

# Pick writable mountpoint with the largest available space (preferring secondary drives over / if >100GB free)
BEST_MNT=""
BEST_AVAIL=0
while read -r mnt avail; do
    if [ -w "$mnt" ] || [ "$mnt" = "/" ]; then
        if [ "$avail" -gt "$BEST_AVAIL" ]; then
            BEST_AVAIL="$avail"
            BEST_MNT="$mnt"
        fi
    fi
done < <(df -BG --output=target,avail -x tmpfs -x devtmpfs -x squashfs -x efivarfs | tail -n +2 | sed 's/G//g')

if [ "$BEST_MNT" = "/" ] || [ -z "$BEST_MNT" ]; then
    WS_DIR="$HOME/training_ws"
else
    WS_DIR="${BEST_MNT%/}/training_ws_${USER}"
fi

mkdir -p "$WS_DIR/data/zarr_datasets" "$WS_DIR/data/outputs" "$WS_DIR/logs"
echo ""
echo "--> Selected Workspace Directory: $WS_DIR (on mount '$BEST_MNT', ${BEST_AVAIL}G free)"

# Send disk info back to master so it's visible on master
ssh "$MASTER_HOST" "cat > /tmp/worker_${WORKER_NAME}_disks.txt" <<EOF
Worker: ${WORKER_NAME}
Selected Workspace: ${WS_DIR}
$(lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,MODEL)
---
$(df -h -x tmpfs -x devtmpfs -x squashfs)
---
$(nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv)
EOF

# Determine execution environment: Docker (ros2_dev) or local venv
USE_DOCKER=0
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^ros2_dev$'; then
    USE_DOCKER=1
    echo "--> Found running Docker container 'ros2_dev' on ${WORKER_NAME}."
elif [ -f "$WS_DIR/venv/bin/activate" ]; then
    echo "--> Found Python venv at $WS_DIR/venv."
elif [ -f "$HOME/training_ws/venv/bin/activate" ]; then
    WS_DIR="$HOME/training_ws"
    mkdir -p "$WS_DIR/data/zarr_datasets" "$WS_DIR/data/outputs" "$WS_DIR/logs"
    echo "--> Using existing venv at $WS_DIR/venv."
else
    echo "--> Setting up Python venv at $WS_DIR/venv..."
    python3 -m venv "$WS_DIR/venv"
fi

while true; do
    # Check GPU utilization before claiming a job
    FREE_MB=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    UTIL_PCT=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    if [ "${FREE_MB:-0}" -lt 10000 ] || [ "${UTIL_PCT:-100}" -gt 25 ]; then
        echo "[$(date '+%H:%M:%S')] Local GPU busy (${FREE_MB}MB free, ${UTIL_PCT}% util). Sleeping 30s..."
        sleep 30
        continue
    fi

    CLAIM_LINE=$(ssh "$MASTER_HOST" "python3 ${MASTER_REPO}/scripts/gpu_queue.py claim-next --worker ${WORKER_NAME}")
    if [ "$CLAIM_LINE" = "NONE" ] || [ -z "$CLAIM_LINE" ]; then
        echo "[$(date '+%H:%M:%S')] No pending jobs in master queue. Waiting 30s..."
        sleep 30
        continue
    fi

    IFS='|' read -r _ JOB_ID ALG TASK DATASET EXTRA_ARGS SEED EXP_NAME REL_DIR MASTER_LOG <<< "$CLAIM_LINE"
    echo "=========================================================="
    echo " Claimed Job: ${JOB_ID}"
    echo " Dataset    : ${DATASET}"
    echo "=========================================================="

    # Sync ManiFlow_Policy code from master
    echo "[Sync] Pulling ManiFlow_Policy code from ${MASTER_HOST}..."
    rsync -az "${MASTER_HOST}:${MASTER_REPO}/third_party/ManiFlow_Policy/" "$WS_DIR/ManiFlow_Policy/"

    # Sync Zarr dataset if not present locally
    if [ ! -d "$WS_DIR/data/zarr_datasets/${DATASET}/state" ]; then
        echo "[Sync] Pulling dataset ${DATASET} from ${MASTER_HOST}..."
        rsync -az --info=progress2 "${MASTER_HOST}:${MASTER_REPO}/data/zarr_datasets/${DATASET}" "$WS_DIR/data/zarr_datasets/"
    fi

    LOCAL_LOG="$WS_DIR/logs/${JOB_ID}.log"
    RUN_DIR="$WS_DIR/data/outputs/${REL_DIR}"
    rm -rf "$RUN_DIR"

    # Background log streamer to master every 15s so `gpu_queue.py status` shows live progress
    (
        while true; do
            sleep 15
            [ -f "$LOCAL_LOG" ] && rsync -az "$LOCAL_LOG" "${MASTER_HOST}:${MASTER_LOG}" 2>/dev/null || true
        done
    ) &
    STREAM_PID=$!

    set +e
    if [ "$USE_DOCKER" -eq 1 ]; then
        docker exec ros2_dev bash -c "
            set -e
            source /ros2_ws/venv/bin/activate
            export PYTHONPATH='$WS_DIR/ManiFlow_Policy/ManiFlow':\${PYTHONPATH:-}
            export VK_ICD_FILENAMES='$WS_DIR/ManiFlow_Policy/scripts/nvidia_icd.json'
            export TOKENIZERS_PARALLELISM=false
            export HYDRA_FULL_ERROR=1
            export CUDA_VISIBLE_DEVICES=0
            cd '$WS_DIR/ManiFlow_Policy/ManiFlow/maniflow/workspace'
            python3 -u train.py \
                --config-name=${ALG}.yaml \
                task=${TASK} \
                task.dataset.zarr_path='$WS_DIR/data/zarr_datasets/${DATASET}' \
                hydra.run.dir='${RUN_DIR}' \
                training.debug=False \
                training.seed=${SEED} \
                training.device='cuda:0' \
                exp_name='${EXP_NAME}' \
                logging.mode=online \
                checkpoint.save_ckpt=True \
                ${EXTRA_ARGS}
        " > "$LOCAL_LOG" 2>&1
        RET=$?
    else
        (
            set -e
            source "$WS_DIR/venv/bin/activate"
            export PYTHONPATH="$WS_DIR/ManiFlow_Policy/ManiFlow:${PYTHONPATH:-}"
            export VK_ICD_FILENAMES="$WS_DIR/ManiFlow_Policy/scripts/nvidia_icd.json"
            export TOKENIZERS_PARALLELISM=false
            export HYDRA_FULL_ERROR=1
            export CUDA_VISIBLE_DEVICES=0
            cd "$WS_DIR/ManiFlow_Policy/ManiFlow/maniflow/workspace"
            python3 -u train.py \
                --config-name=${ALG}.yaml \
                task=${TASK} \
                task.dataset.zarr_path="$WS_DIR/data/zarr_datasets/${DATASET}" \
                hydra.run.dir="${RUN_DIR}" \
                training.debug=False \
                training.seed=${SEED} \
                training.device='cuda:0' \
                exp_name="${EXP_NAME}" \
                logging.mode=online \
                checkpoint.save_ckpt=True \
                ${EXTRA_ARGS}
        ) > "$LOCAL_LOG" 2>&1
        RET=$?
    fi
    set -e

    kill $STREAM_PID 2>/dev/null || true
    rsync -az "$LOCAL_LOG" "${MASTER_HOST}:${MASTER_LOG}" 2>/dev/null || true

    if [ "$RET" -eq 0 ]; then
        echo "[Sync] Pushing finished checkpoints back to ${MASTER_HOST}..."
        ssh "$MASTER_HOST" "mkdir -p '${MASTER_REPO}/data/outputs/${REL_DIR}'"
        rsync -az "${RUN_DIR}/" "${MASTER_HOST}:${MASTER_REPO}/data/outputs/${REL_DIR}/"
        rm -f "${RUN_DIR}/checkpoints/"*.ckpt
        ssh "$MASTER_HOST" "python3 ${MASTER_REPO}/scripts/gpu_queue.py finish-job --id ${JOB_ID} --status completed"
        echo "[DONE] ${JOB_ID} completed and synced!"
    else
        ssh "$MASTER_HOST" "python3 ${MASTER_REPO}/scripts/gpu_queue.py finish-job --id ${JOB_ID} --status failed"
        echo "[FAILED] ${JOB_ID} exited with code ${RET}. See ${LOCAL_LOG}"
        sleep 10
    fi
done
