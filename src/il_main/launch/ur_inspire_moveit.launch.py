#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    # Package share directories
    ur_pkg = get_package_share_directory('ur_robot_driver')
    demo_pkg = get_package_share_directory('il_main')

    # UR Robot Driver
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_pkg, 'launch', 'ur_control.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur10e',
            'robot_ip': '192.168.56.101',
            'use_fake_hardware': 'false',
            'launch_rviz': 'false',
            'kinematics_params_file': '/ros2_ws/src/il_main/my_robot_calibration.yaml',
            'initial_joint_controller': 'joint_trajectory_controller'
        }.items()
    )

    inspire_control = Node(
        package="inspire_hand_modbus_ros2",
        executable="inspire_hand_modbus_topic.py",
        output="screen",
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(demo_pkg, 'launch', 'moveit.launch.py')
        )
    )

    static_tf_zed = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '1', '0', '0', '0', 'world', 'zed_camera_link']
    )

    static_tf_vive_tracker = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['-0.010930', '0.018770', '0.114410', '0.0', '0.0', '0.0', 'LHR-9F6576C3', 'virtual_palm_link']
    )

    teleop_inspire = Node(
        package="il_main",
        executable="teleop_inspire_r1",
        output="screen",
    )

    return LaunchDescription([
    ur_control,
    static_tf_zed,
    # static_tf_vive_tracker,
    inspire_control,
    TimerAction(period=5.0, actions=[
                                     moveit, 
                                     teleop_inspire
                                     ])
    ])
