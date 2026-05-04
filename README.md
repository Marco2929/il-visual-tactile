# Useful commands

sudo ip addr add 192.168.56.100/24 dev enp5s0

## Inspire commands:

ros2 run inspire_hand_modbus_ros2 hand_modbus_control_node
ros2 run inspire_hand_modbus_ros2 inspire_hand_modbus_topic.py

ros2 topic pub --once /set_angle_data service_interfaces/msg/SetAngle1 "{finger_ids: [1, 2, 3, 4, 5, 6], angles: [1000, 1000, 1000, 1000, 1000, 1000]}"


## Senseglove commands:

ros2 launch senseglove_bringup senseglove.launch.py

ros2 topic echo /senseglove/glove0/rh/joint_states --field position

## Moveit comands:

XACRO viewing: ros2 launch moveit_ur_hand_config view_robot.launch.py ur_type:=ur10e hand_type:=inspire_strong_hand flange_type:=ring

URDF generation: ./src/moveit_ur_hand_config/scripts/generate_urdf.sh ur10e tesollo_hand ring

ros2 control switch_controllers --deactivate forward_position_controller --activate scaled_joint_trajectory_controller
ros2 launch il_main ur_inspire_moveit.launch.py

## Tesollo:

Mock:
ros2 topic pub -r 10 /joint_states sensor_msgs/msg/JointState "{
  header: {frame_id: 'base_link'}, 
  name: [
    'rj_dg_1_1', 'rj_dg_1_2', 'rj_dg_1_3', 'rj_dg_1_4', 
    'rj_dg_2_1', 'rj_dg_2_2', 'rj_dg_2_3', 'rj_dg_2_4', 
    'rj_dg_3_1', 'rj_dg_3_2', 'rj_dg_3_3', 'rj_dg_3_4', 
    'rj_dg_4_1', 'rj_dg_4_2', 'rj_dg_4_3', 'rj_dg_4_4', 
    'rj_dg_5_1', 'rj_dg_5_2', 'rj_dg_5_3', 'rj_dg_5_4'
  ], 
  position: [
    0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0
  ], 
  velocity: [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  ], 
  effort: [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  ]
}"

Run:
ros2 launch dg5f_driver dg5f_right_driver.launch.py delto_ip:=169.254.186.72 delto_port:=502 fingertip_sensor:=true

## Teleop:

python3 src/il_main/il_main/utils/move_utils/move_robot_pose.py 
ros2 control switch_controllers --deactivate forward_position_controller --activate scaled_joint_trajectory_controller
ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller --activate forward_position_controller

ros2 launch il_main ur_inspire_moveit.launch.py
ros2 run il_main teleop_ur
ros2 launch senseglove_bringup senseglove.launch.py

ros2 run tf2_ros static_transform_publisher 0.02250 0.02236 0.09635 0.78228 0.0 0.0 LHR-9F6576C3 virtual_palm_link

## Train:
/ros2_ws/data/episodes/episode_000/point_cloud
python record_utils/remove_recordings.py

python data/record_utils/generatePointCloud.py
python data/record_utils/visualize_pointcloud.py --path /ros2_ws/data/test_collection/episode_001/point_cloud
python data/record_utils/merge_episodes.py 

bash third_party/scripts/train_policy.sh simple_dp3 pillar_grasping 0112 0 0