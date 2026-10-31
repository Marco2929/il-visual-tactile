#!/usr/bin/env bash
set -euo pipefail

S="${1:-dev}"
exec tmux new -A -s "$S" \
  "docker exec -it -w /ros2_ws ros2_dev bash -c 'source install/setup.bash; source venv/bin/activate; exec bash'"