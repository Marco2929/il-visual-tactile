#!/usr/bin/env bash
# Run the DG5F temperature watchdog with CSV logging to the host.
#
# WHY docker exec AND NOT A SEPARATE CONTAINER
# Fast DDS uses shared memory for same-host traffic, which requires a shared
# IPC namespace. --network host only shares the network namespace. The
# SenseGlove containers run with IpcMode=private, so a sidecar container
# receives no data at all (mounting /dev/shm does not help either, and
# --ipc=container: is refused because the donor is not shareable).
# Therefore the watchdog runs INSIDE the teleop container.
#
# WHY THE CSV IS COPIED OUT PERIODICALLY
# The teleop container has no host mount for logs (only /dev and
# /tmp/.X11-unix). The script writes the CSV inside the container and syncs it
# to ./logs/ on the host every few seconds, so a long run survives a crash or
# a container restart.
#
# WHICH CONTAINER
# The hand stack does not always live in the teleop container. During policy
# deployment the DG5F controller_manager runs in the ros2_dev container
# instead, and the watchdog is just as useful there - it never writes to the
# command path, so it works unchanged alongside an inference run.
# With no CONTAINER set, the candidates below are probed for a live
# /dg5f_<hand>/ controller_manager and the first match wins. The ROS workspace
# path is detected per container too (/ros_ws vs /ros2_ws).
#
# USAGE
#   bash run_watchdog.sh                      # auto-detect, trip at 65 C
#   bash run_watchdog.sh --stop-at 60         # lower trip point
#   bash run_watchdog.sh --dry-run            # warn only, never deactivate
#   HAND=left bash run_watchdog.sh
#   CONTAINER=ros2_dev bash run_watchdog.sh   # skip auto-detection
#
# Output lands in ./logs/watchdog_<timestamp>.csv and .log on the host.
set -euo pipefail

HAND="${HAND:-right}"
SYNC_EVERY="${SYNC_EVERY:-5}"
# Teleop container first, so an interactive session keeps its previous
# behaviour; ros2_dev is where the stack sits during deployment.
CANDIDATES="${CANDIDATES:-senseglove_tesollo ros2_dev}"

# Echo the ROS workspace setup script inside $1, or nothing if there is none.
find_ws_setup() {
  docker exec -i "$1" bash -lc '
    for p in /ros_ws /ros2_ws /root/ros2_ws /opt/ros_ws; do
      if [ -f "$p/install/setup.bash" ]; then echo "$p/install/setup.bash"; exit 0; fi
    done' 2>/dev/null || true
}

# Succeed if $1 can see the controller_manager for this hand. Checking the
# topic rather than the container name is what makes auto-detection reliable:
# a container can be up while the hand driver is not.
has_hand_stack() {
  local setup; setup="$(find_ws_setup "$1")"
  [ -n "$setup" ] || return 1
  docker exec -i "$1" bash -lc "
    source '$setup' >/dev/null 2>&1
    timeout 8 ros2 topic list 2>/dev/null |
      grep -q '^/dg5f_${HAND}/controller_manager'" 2>/dev/null
}

if [[ -z "${CONTAINER:-}" ]]; then
  for c in $CANDIDATES; do
    docker ps --format '{{.Names}}' | grep -qx "$c" || continue
    if has_hand_stack "$c"; then
      CONTAINER="$c"
      echo "auto-detected hand stack in container '$CONTAINER'"
      break
    fi
  done
fi

if [[ -z "${CONTAINER:-}" ]]; then
  echo "error: no running container publishes /dg5f_${HAND}/controller_manager." >&2
  echo "Searched: $CANDIDATES" >&2
  echo "Start the hand driver, or set CONTAINER explicitly:" >&2
  echo "  CONTAINER=ros2_dev bash run_watchdog.sh" >&2
  exit 1
fi

GUARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$GUARD_DIR/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
HOST_CSV="$LOG_DIR/watchdog_${STAMP}.csv"
HOST_LOG="$LOG_DIR/watchdog_${STAMP}.log"
IN_CSV="/tmp/watchdog_${STAMP}.csv"

mkdir -p "$LOG_DIR"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "error: container '$CONTAINER' is not running." >&2
  echo "Start teleop first, for example:" >&2
  echo "  uv run start.py tesollo --hand $HAND --glove-id 70" >&2
  echo "...or point CONTAINER at the deployment container:" >&2
  echo "  CONTAINER=ros2_dev bash run_watchdog.sh" >&2
  exit 1
fi

WS_SETUP="$(find_ws_setup "$CONTAINER")"
if [[ -z "$WS_SETUP" ]]; then
  echo "error: no install/setup.bash found in '$CONTAINER'." >&2
  echo "Build the workspace there, or set CONTAINER to the right one." >&2
  exit 1
fi

docker cp "$GUARD_DIR/dg5f_temp_watchdog.py" \
  "$CONTAINER:/tmp/dg5f_temp_watchdog.py" >/dev/null

# Pull the CSV out of the container until this helper is killed.
sync_loop() {
  while true; do
    sleep "$SYNC_EVERY"
    docker cp "$CONTAINER:$IN_CSV" "$HOST_CSV" 2>/dev/null || true
  done
}

cleanup() {
  [[ -n "${SYNC_PID:-}" ]] && kill "$SYNC_PID" 2>/dev/null || true
  # Final copy so the last samples are not lost.
  docker cp "$CONTAINER:$IN_CSV" "$HOST_CSV" 2>/dev/null || true
  echo
  echo "CSV: $HOST_CSV"
  echo "log: $HOST_LOG"
  if [[ -f "$HOST_CSV" ]]; then
    echo "rows: $(( $(wc -l < "$HOST_CSV") - 1 ))"
  fi
}
trap cleanup EXIT INT TERM

sync_loop &
SYNC_PID=$!

echo "watchdog on $CONTAINER [$HAND]   CSV -> $HOST_CSV"
echo "workspace: $WS_SETUP"
echo "press Ctrl-C to stop"
echo

# No -t: keeps the log file free of escape sequences. tee keeps it on screen.
docker exec -i "$CONTAINER" bash -c \
  "source '$WS_SETUP' &&
   python3 /tmp/dg5f_temp_watchdog.py --hand '$HAND' --csv '$IN_CSV' $*" \
  2>&1 | tee "$HOST_LOG"