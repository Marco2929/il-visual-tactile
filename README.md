# Useful commands

touch /ros2_ws/src/il-main/il/venv/COLCON_IGNORE
export PYTHONPATH=$PYTHONPATH:/ros2_ws/src/il-main/il/venv/lib/python3.12/site-packages

## Remapping

ros2 run il_main teleop_hand

## Senseglove commands:

ros2 launch senseglove_bringup senseglove.launch.py

ros2 topic echo /senseglove/glove0/rh/joint_states --field position

## Moveit comands:

XACRO viewing: ros2 launch moveit_ur_hand_config view_robot.launch.py ur_type:=ur10e hand_type:=inspire_strong_hand flange_type:=ring

URDF generation: ./src/moveit_ur_hand_config/scripts/generate_urdf.sh ur10e tesollo_hand ring

## Deploy

ros2 launch imitation_grasping ur_inspire_moveit.launch.py