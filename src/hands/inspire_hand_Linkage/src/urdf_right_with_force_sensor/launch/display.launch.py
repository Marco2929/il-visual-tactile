from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('urdf_right_with_force_sensor'), 'urdf','urdf_right_with_force_sensor.urdf'
    )

    return LaunchDescription([
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf_file).read()}]  
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            output='screen',
            arguments=['-d', os.path.join(get_package_share_directory('urdf_right_with_force_sensor'),'rviz', 'right.rviz')]  
        ),
    ])


