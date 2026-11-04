#!/usr/bin/env bash
set -e

# Setup ROS 2 environment
source /opt/ros/${ROS_DISTRO}/setup.bash

# Setup ManiFlow workspace / interfaces if built
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

# Use default CycloneDDS without custom XML
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# If custom cyclonedds.xml is mounted into /etc/cyclonedds.xml, use it, otherwise use CycloneDDS default
if [ -f "/etc/cyclonedds.xml" ]; then
    export CYCLONEDDS_URI=file:///etc/cyclonedds.xml
fi

exec "$@"
