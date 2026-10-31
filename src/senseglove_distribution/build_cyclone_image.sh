#!/usr/bin/env bash
# Build the Cyclone DDS overlay image for the SenseGlove stack.
#
# Produces <repo>:cyclone-<tag> from <repo>:<tag>, adding only
# ros-jazzy-rmw-cyclonedds-cpp so the SenseGlove containers can join the
# Cyclone DDS domain the rest of the workspace already uses.
#
# Re-run this after every "uv run image_manager.py --pull", because the
# overlay is pinned to the base image it was built from.
#
# Usage:
#   bash build_cyclone_image.sh              # from :latest
#   BASE_TAG=tesollo-1.2.0 bash build_cyclone_image.sh
set -euo pipefail

REPO="${REPO:-zippeth/senseglove_robotics}"
BASE_TAG="${BASE_TAG:-latest}"
OUT_TAG="cyclone-${BASE_TAG}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! docker image inspect "${REPO}:${BASE_TAG}" >/dev/null 2>&1; then
  echo "error: base image ${REPO}:${BASE_TAG} not found locally." >&2
  echo "Pull it first:  uv run image_manager.py --pull" >&2
  exit 1
fi

echo "building ${REPO}:${OUT_TAG}  (from ${REPO}:${BASE_TAG})"
docker build \
  --build-arg "BASE_IMAGE=${REPO}:${BASE_TAG}" \
  -f "${HERE}/Dockerfile.cyclone" \
  -t "${REPO}:${OUT_TAG}" \
  "${HERE}"

echo
echo "done. Start both stacks with the new tag:"
echo "  uv run start.py r1      --tag ${OUT_TAG} --num-gloves 1"
echo "  uv run start.py tesollo --tag ${OUT_TAG} --hand right --glove-id 70"
echo
echo "Both must use it: r1 publishes the glove states the tesollo mapper"
echo "subscribes to, so they have to share the same DDS implementation."
