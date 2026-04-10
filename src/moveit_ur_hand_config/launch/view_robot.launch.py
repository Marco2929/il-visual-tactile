import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Specify the package name
    pkg_name = 'moveit_ur_hand_config'
    
    # Declare Launch Configurations
    ur_type = LaunchConfiguration('ur_type')
    hand_type = LaunchConfiguration('hand_type')
    flange_type = LaunchConfiguration('flange_type')
    
    # Declare Launch Arguments
    declare_ur_type_cmd = DeclareLaunchArgument(
        'ur_type',
        default_value='ur10e',
        description='Type of UR robot (e.g., ur5e, ur10e)'
    )
    
    declare_hand_type_cmd = DeclareLaunchArgument(
        'hand_type',
        default_value='inspire_strong_hand',
        description='Type of end effector to attach (e.g., inspire_strong_hand, artus_hand)'
    )

    declare_flange_type_cmd = DeclareLaunchArgument(
        'flange_type',
        default_value='robotiq', # Now accepts: robotiq, ring, or none
        description='Flange adapter type: robotiq, ring, or none'
    )
    
    # Path to the generic top-level xacro file
    xacro_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'workcell.urdf.xacro'
    )
    
    # Process the xacro file dynamically with the injected arguments
    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file, 
            ' ur_type:=', ur_type, 
            ' hand_type:=', hand_type,
            ' flange_type:=', flange_type
        ]),
        value_type=str
    )

    # Node to publish the robot state (TF transforms)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # Node to publish fake joint states for testing
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        declare_ur_type_cmd,
        declare_hand_type_cmd,
        declare_flange_type_cmd,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])