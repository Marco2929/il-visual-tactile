#!/usr/bin/env python3
"""
Launch calibration for ZED 2i (hand-to-eye) or RealSense 435i (eye-in-hand).

Usage:
  ros2 launch calibration_camera calibration.launch.py camera:=zed
  ros2 launch calibration_camera calibration.launch.py camera:=realsense
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ── Arguments ──────────────────────────────────────────────
    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='zed',
        choices=['zed', 'realsense'],
        description='Which camera to calibrate: zed or realsense'
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Robot base frame'
    )

    tcp_frame_arg = DeclareLaunchArgument(
        'tcp_frame',
        default_value='tool0',
        description='Robot TCP frame'
    )

    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic',
        default_value='/robot/pose',
        description='Topic for robot TCP pose'
    )

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='/ros2_ws/data/calibration',
        description='Base directory for calibration output'
    )

    return LaunchDescription([
        camera_arg,
        base_frame_arg,
        tcp_frame_arg,
        pose_topic_arg,
        output_dir_arg,
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context):
    camera = LaunchConfiguration('camera').perform(context)
    base_frame = LaunchConfiguration('base_frame').perform(context)
    tcp_frame = LaunchConfiguration('tcp_frame').perform(context)
    pose_topic = LaunchConfiguration('pose_topic').perform(context)
    output_dir = LaunchConfiguration('output_dir').perform(context)

    nodes = []

    # ── Info ───────────────────────────────────────────────────
    if camera == 'zed':
        info_msg = (
            '\n'
            '╔══════════════════════════════════════════════════════╗\n'
            '║  CALIBRATION MODE: ZED 2i (Hand-to-Eye)             ║\n'
            '║  Board: mounted on robot flange                     ║\n'
            '║  Camera: fixed in workspace                         ║\n'
            '╚══════════════════════════════════════════════════════╝'
        )
        rgbd_service = '/zed/get_rgbd'
        output_subdir = os.path.join(output_dir, 'zed_hand_to_eye')
    else:
        info_msg = (
            '\n'
            '╔══════════════════════════════════════════════════════╗\n'
            '║  CALIBRATION MODE: RealSense 435i (Eye-in-Hand)     ║\n'
            '║  Board: fixed in workspace                          ║\n'
            '║  Camera: mounted on robot flange                    ║\n'
            '╚══════════════════════════════════════════════════════╝'
        )
        rgbd_service = '/realsense/get_rgbd'
        output_subdir = os.path.join(output_dir, 'realsense_eye_in_hand')

    nodes.append(LogInfo(msg=info_msg))

    # ── TCP Pose Publisher ─────────────────────────────────────
    tcp_pose_node = Node(
        package='calibration_camera',
        executable='tcp_pose_publisher',
        name='tcp_pose_publisher',
        parameters=[{
            'base_frame': base_frame,
            'tcp_frame': tcp_frame,
            'publish_rate': 50.0,
            'pose_topic': pose_topic,
        }],
        output='screen',
    )
    nodes.append(tcp_pose_node)

    # ── RGBD Service (kamera-abhängig) ─────────────────────────
    if camera == 'zed':
        rgbd_node = Node(
            package='calibration_camera',
            executable='rgbd_service_zed',
            name='zed_rgbd_service',
            parameters=[{
                'color_topic': '/zed/zed_node/rgb/color/rect/image',
                'depth_topic': '/zed/zed_node/depth/depth_registered',
                'camera_info_topic': '/zed/zed_node/rgb/color/rect/camera_info',
            }],
            output='screen',
        )
    else:
        rgbd_node = Node(
            package='calibration_camera',
            executable='rgbd_service_realsense',
            name='realsense_rgbd_service',
            parameters=[{
                'color_topic': '/cam_right/cam_right/color/image_raw',
                'depth_topic': '/cam_right/cam_right/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/cam_right/cam_right/color/camera_info',
            }],
            output='screen',
        )
    nodes.append(rgbd_node)

    # ── Calibration Node ───────────────────────────────────────
    calibration_node = Node(
        package='calibration_camera',
        executable='calibration_node',
        name='calibration',
        parameters=[{
            'rgbd_service': rgbd_service,
            'rgbd_service_timeout': 20.0,
            'pose_topic': pose_topic,
            'output_base_dir': output_subdir,
            'board.num_squares_x': 14,
            'board.num_squares_y': 9,
            'board.square_length': 0.020,
            'board.marker_length': 0.015,
            'board.dictionary': 'DICT_5X5_250',
            'min_marker_ids': 4,
            'min_charuco_ids': 6,
        }],
        output='screen',
    )
    nodes.append(calibration_node)

    return nodes