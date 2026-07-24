#!/usr/bin/env bash
# Generate FRONT point clouds for one episode across a grid of ZED depth
# parameters (confidence + texture_confidence), both varied 100..70 step 10.
#
# Usage:
#   ./data/record_extractor/run_pc_variants.sh \
#       /root/ros2_ws/data/episodes/external/test_collection_small/episode_000_success

set -e

EPISODE="${1:-/root/ros2_ws/data/episodes/external/test_collection_small/episode_000_success}"
SCRIPT="$(dirname "$0")/extract_pc_variants.py"

# Vary both conf and tex from 100 down to 70 in steps of 10.
VALUES=(100 90 80 70)

echo "Episode: $EPISODE"
echo "Grid: conf x tex in {100,90,80,70} = 16 configs"
echo "---------------------------------------------"

for CONF in "${VALUES[@]}"; do
    for TEX in "${VALUES[@]}"; do
        echo ">>> conf=$CONF tex=$TEX"
        python3 "$SCRIPT" --episode "$EPISODE" --conf "$CONF" --tex "$TEX"
    done
done

echo "---------------------------------------------"
echo "Done. Browse with:"
echo "  python3 data/record_extractor/browse_pc_variants.py --episode $EPISODE"