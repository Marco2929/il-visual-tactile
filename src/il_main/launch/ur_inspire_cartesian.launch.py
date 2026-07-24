#!/usr/bin/env python3
"""Launch file for UR10e with CartesianMotionController and Inspire hand.

Spawns cartesian_motion_controller as the active arm controller.
No joint_trajectory_controller or scaled_joint_trajectory_controller is started.
"""
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    # Package share directories
    ur_pkg = get_package_share_directory('cartesian_controllers_universal_robots')
    demo_pkg = get_package_share_directory('il_main')

    # Path to our custom controller parameters
    cartesian_controller_config = os.path.join(
        demo_pkg, 'config', 'cartesian_motion_controller.yaml'
    )

    # UR Robot Driver with CartesianMotionController as the initial (active) controller
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_pkg, 'launch', 'robot.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur10e',
            'robot_ip': '192.168.56.101',
            'use_fake_hardware': 'false',
            'launch_rviz': 'false',
            'kinematics_params_file': '/ros2_ws/src/il_main/my_robot_calibration.yaml',
            'initial_joint_controller': 'cartesian_motion_controller',
            'controllers_file': cartesian_controller_config,
        }.items()
    )

    # Inspire hand serial control
    inspire_control = Node(
        package="inspire_hand_modbus_ros2",
        executable="inspire_hand_modbus_topic.py",
        output="screen",
    )

    return LaunchDescription([
        ur_control,
        inspire_control,
    ])
