#!/usr/bin/env bash
# Runs compare_foil_height.py inside zed_container
set -euo pipefail
docker exec zed_container python3 /root/ros2_ws/data/rollouts/tools/compare_foil_height.py "$@"
