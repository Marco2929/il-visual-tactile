from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    # 定义 mode 参数
    mode = LaunchConfiguration('mode')

    return LaunchDescription([
        # 打印当前模式值，便于调试
        LogInfo(msg=['Current mode is: ', mode]),

        # 启动 hand_modbus_control_node
        Node(
            package='inspire_hand_modbus_ros2',
            executable='hand_modbus_control_node',
            name='hand_modbus_control_node',
            output='screen',
        ),

        ExecuteProcess(
            cmd=['ros2', 'run', 'inspire_hand_modbus_ros2', 'handcontrol_topic_publisher_modbus.py'],
            output='screen',
            condition=IfCondition(PythonExpression(["'", mode, "' == '2'"])),
        ),

        ExecuteProcess(
            cmd=['ros2', 'run', 'inspire_hand_modbus_ros2', 'handcontrol_topic_subscriber_modbus.py'],
            output='screen',
            condition=IfCondition(PythonExpression(["'", mode, "' == '2'"])),
        ),
    ])

