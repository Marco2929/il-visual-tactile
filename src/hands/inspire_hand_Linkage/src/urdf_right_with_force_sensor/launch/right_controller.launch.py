import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 定义包名称
    package_name = 'urdf_right_with_force_sensor'

    # 包含自定义机器人描述的启动文件
    right = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory(package_name), 'launch', 'right.launch.py'
        )]), launch_arguments={'use_sim_time': 'true'}.items()
    )
    
    # 启动关节状态广播器节点
    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state"],
    )

    # 启动联合轨迹控制器
    joint_trajectory_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller"],
    )

    # 启动六维力传感器控制器
    palm_force_sensor_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["palm_force_sensor_controller"],
    )
    
    thumb_force_sensor_1_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["thumb_force_sensor_1_controller"],
    )    
    
    thumb_force_sensor_2_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["thumb_force_sensor_2_controller"],
    )  

    thumb_force_sensor_3_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["thumb_force_sensor_3_controller"],
    )    
    
    thumb_force_sensor_4_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["thumb_force_sensor_4_controller"],
    )  

    index_force_sensor_1_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["index_force_sensor_1_controller"],
    )  

    index_force_sensor_2_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["index_force_sensor_2_controller"],
    ) 

    index_force_sensor_3_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["index_force_sensor_3_controller"],
    ) 

    middle_force_sensor_1_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["middle_force_sensor_1_controller"],
    )  

    middle_force_sensor_2_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["middle_force_sensor_2_controller"],
    ) 
  
    middle_force_sensor_3_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["middle_force_sensor_3_controller"],
    )

    ring_force_sensor_1_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ring_force_sensor_1_controller"],
    )  

    ring_force_sensor_2_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ring_force_sensor_2_controller"],
    ) 
  
    ring_force_sensor_3_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ring_force_sensor_3_controller"],
    )

    little_force_sensor_1_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["little_force_sensor_1_controller"],
    )  

    little_force_sensor_2_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["little_force_sensor_2_controller"],
    ) 
  
    little_force_sensor_3_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["little_force_sensor_3_controller"],
    )
     
    return LaunchDescription([
        right,
        joint_broad_spawner,
        joint_trajectory_spawner,
        palm_force_sensor_spawner,
        thumb_force_sensor_1_spawner,
        thumb_force_sensor_2_spawner,
        thumb_force_sensor_3_spawner,
        thumb_force_sensor_4_spawner,
        index_force_sensor_1_spawner,
        index_force_sensor_2_spawner,
        index_force_sensor_3_spawner,
        middle_force_sensor_1_spawner,
        middle_force_sensor_2_spawner,        
        middle_force_sensor_3_spawner,     
        ring_force_sensor_1_spawner,
        ring_force_sensor_2_spawner,        
        ring_force_sensor_3_spawner, 
        little_force_sensor_1_spawner,
        little_force_sensor_2_spawner,        
        little_force_sensor_3_spawner,    
    ])

