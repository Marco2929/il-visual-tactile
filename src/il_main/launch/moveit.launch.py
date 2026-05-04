#!/usr/bin/env python3
import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def load_yaml(pkg_share, relpath):
    """Small helper to load yaml files safely."""
    path = os.path.join(pkg_share, relpath)
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except EnvironmentError: 
        return {}

def generate_launch_description():
    # 1. Configuration & Path Setup
    pkg_name = "moveit_ur_hand_config"
    pkg_share = get_package_share_directory(pkg_name)

    # 2. Declare Launch Arguments (Enables dynamic switching at runtime)
    # Available flange_types: robotiq, ring, none
    # Available hand_types: inspire_strong_hand, tesollo_hand, etc.
    hand_type_arg = DeclareLaunchArgument("hand_type", default_value="inspire_strong_hand")
    flange_type_arg = DeclareLaunchArgument("flange_type", default_value="ring")
    ur_type_arg = DeclareLaunchArgument("ur_type", default_value="ur10e")

    hand_type = LaunchConfiguration("hand_type")
    flange_type = LaunchConfiguration("flange_type")
    ur_type = LaunchConfiguration("ur_type")

    # 3. Dynamic Robot Description (URDF)
    robot_description_content = ParameterValue(
        Command([
            "xacro ", os.path.join(pkg_share, "urdf", "workcell.urdf.xacro"),
            " ur_type:=", ur_type,
            " hand_type:=", hand_type,
            " flange_type:=", flange_type,
        ]),
        value_type=str
    )
    # Important: MoveIt expects the key 'robot_description'
    robot_description = {"robot_description": robot_description_content}

    # 4. Dynamic Semantic Robot Description (SRDF)
    robot_description_semantic_content = ParameterValue(
        Command([
            "xacro ", os.path.join(pkg_share, "config", "srdf", "workcell.srdf.xacro"),
            " hand_type:=", hand_type,
        ]),
        value_type=str
    )
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # 5. Load Static YAML Parameters
    kinematics_yaml = load_yaml(pkg_share, "config/kinematics.yaml")
    joint_limits_yaml = load_yaml(pkg_share, "config/joint_limits.yaml")
    planning_yaml = load_yaml(pkg_share, "config/ompl_planning.yaml")
    
    # Load MoveIt Controller configurations
    controllers_yaml = load_yaml(pkg_share, "config/moveit_controllers.yaml")

    # 6. Define Nodes
    # Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": kinematics_yaml},
            {"robot_description_planning": joint_limits_yaml},
            {"planning_pipelines": ["ompl"], 
             "default_planning_pipeline": "ompl", 
             "ompl": planning_yaml},
            controllers_yaml,
            {"trajectory_execution.allowed_start_tolerance": 0.01},
        ],
    )

    rviz_cfg = os.path.join(pkg_share, "config", "moveit.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": kinematics_yaml},
        ],
    )

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    servo_yaml_content = load_yaml(pkg_share, "config/ur_servo.yaml")
    
    servo_params = {"moveit_servo": servo_yaml_content}

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": kinematics_yaml},
            servo_params,
        ],
        output="screen",
    )

    return LaunchDescription([
        ur_type_arg,
        hand_type_arg,
        flange_type_arg,
        rsp_node,
        move_group_node,
        rviz_node,
        servo_node
    ])