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
            'initial_joint_controller': 'forward_position_controller'
        }.items()
    )

    tesollo_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('dg5f_driver'),
                'launch', # Standard ROS 2 Unterordner für Launch-Dateien
                'dg5f_right_driver.launch.py'
            )
        ),
        # Hier werden die Parameter (delto_ip und delto_port) übergeben
        launch_arguments={
            'delto_ip': '169.254.186.72',
            'delto_port': '502',
        }.items()
    )


    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(demo_pkg, 'launch', 'moveit.launch.py')
        ),
        launch_arguments={
            'hand_type': 'tesollo_hand',
                }.items()
    )

    # tesollo_teleop = 

    return LaunchDescription([
    ur_control,
    # tesollo_control,
    TimerAction(period=5.0, actions=[
                                     moveit, 
                                     #tesollo_teleop
                                     ])
    ])
