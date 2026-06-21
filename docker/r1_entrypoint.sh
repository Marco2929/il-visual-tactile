#!/bin/bash
set -e

: "${ROS_DISTRO:=jazzy}"
: "${ROS_WORKSPACE:=/r1_ws}"

cd "${ROS_WORKSPACE}"

python3 -m venv .venv

source .venv/bin/activate && source /opt/ros/${ROS_DISTRO}/setup.bash

cd "${ROS_WORKSPACE}/src"

pip install -r rembrandt_ros/requirements.txt

cd ${ROS_WORKSPACE}/src/rembrandt-api
pip install -e .

cd ${ROS_WORKSPACE}

if [[ -f "${ROS_WORKSPACE}/install/setup.bash" ]]; then
  source "${ROS_WORKSPACE}/install/setup.bash"
else
  colcon build --symlink-install
  source "${ROS_WORKSPACE}/install/setup.bash"
  echo "SenseGlove container ready"
fi

exec "$@"