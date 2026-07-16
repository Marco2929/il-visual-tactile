# Robot System Commands

## Network Setup

sudo ip addr add 192.168.56.100/24 dev enp5s0

---

## Hardware Drivers

### Inspire Hand

ros2 run inspire_hand_modbus_ros2 hand_modbus_control_node
ros2 run inspire_hand_modbus_ros2 inspire_hand_modbus_topic.py
ros2 run inspire_hand_modbus_ros2 touch_data.py

# Test hand angles
ros2 topic pub --once /set_angle_data service_interfaces/msg/SetAngle1 "{finger_ids: [1, 2, 3, 4, 5, 6], angles: [1000, 1000, 1000, 1000, 1000, 000]}"

### Tesollo Hand

# Run driver
ros2 launch dg5f_driver dg5f_right_driver.launch.py delto_ip:=169.254.186.72 delto_port:=502 fingertip_sensor:=true

### SenseGlove

ros2 launch senseglove_bringup senseglove.launch.py
ros2 topic echo /senseglove/glove0/rh/joint_states --field position

---

## MoveIt & Controllers

# View XACRO
ros2 launch moveit_ur_hand_config view_robot.launch.py ur_type:=ur10e hand_type:=inspire_strong_hand flange_type:=ring

# Generate URDF
./src/moveit_ur_hand_config/scripts/generate_urdf.sh ur10e tesollo_hand ring

# Switch controllers
ros2 control switch_controllers --deactivate forward_position_controller --activate scaled_joint_trajectory_controller
ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller --activate forward_position_controller

# Launch MoveIt
ros2 launch il_main ur_inspire_moveit.launch.py

---

## Calibration

### Launch Calibration Node

# ZED (Hand-to-Eye)
ros2 launch calibration_camera calibration.launch.py camera:=zed

# RealSense (Eye-in-Hand)
ros2 launch calibration_camera calibration.launch.py camera:=realsense

### Calibration Workflow

# Log poses (repeat 8-20 times in different positions)
ros2 service call /calibration/log_data interfaces/srv/LogCalibrationData "{}"

# Check status
ros2 service call /calibration/get_status interfaces/srv/GetCalibrationStatus "{}"

# Save data
ros2 service call /calibration/save_data interfaces/srv/SaveCalibrationData "{folder_name: 'run01'}"

# Run calibration (RealSense = eye-in-hand)
ros2 service call /calibration/run_calibration interfaces/srv/RunCalibration "{hand_in_eye: true, method: '', data_dir: ''}"

# Run calibration (ZED = hand-to-eye)
ros2 service call /calibration/run_calibration interfaces/srv/RunCalibration "{hand_in_eye: false, method: '', data_dir: '/ros2_ws/data/calibration/realsense_eye_in_hand'}"

### Legacy Calibration

ros2 run charuco_detector charuco_detector_node
ros2 run calibrate_ur5_to_realsense_d435i calibrate_ur5_to_realsense_d435i
ros2 run charuco_processor charuco_processor

---

## Teleoperation

### Setup

ros2 launch il_main ur_inspire_moveit.launch.py
ros2 launch r1_bringup r1.launch.py
ros2 launch senseglove_bringup senseglove.launch.py

# Static TF for tracker
ros2 run tf2_ros static_transform_publisher 0.02250 0.02236 0.09635 0.78228 0.0 0.0 LHR-9F6576C3 virtual_palm_link

### Run Teleop

python3 src/il_main/il_main/utils/move_utils/move_robot_pose.py
ros2 run il_main teleop_ur

---

## Record & Train Pipeline

### Environment Variables

export FOLDER_NAME="box_opening_collection"
export FOLDER_NAME="foil_lifting_collection"
export FOLDER_NAME="dataset_collection"
export URDF_PATH="/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf"
export ROBOT_TIP_FRAME="right_inspire_base_link"

### Recording

python data/record_utils/record.py

### Post-Processing

python data/record_utils/remove_recordings.py
python data/record_utils/generatePointCloud.py
python data/record_utils/merge_episodes.py

# Merge to LeRobot format
python data/record_utils/merge_episodes_lerobot.py --repo_id your-hf-user/box_opening_collection --task "test test" --fps 30

### Visualization

# Visualize point cloud
python data/record_utils/visualize/visualize_pointcloud.py --path /ros2_ws/data/episodes/foil_lifting_collection/episode_001_success/point_cloud/frame_0001.npy

# Visualize LeRobot dataset
lerobot-dataset-viz --repo-id Marco2929/box_opening_baseline --episode-index 0 --root /root/.cache/huggingface/lerobot/Marco2929/box_opening_baseline --mode local

### Training

# ManiFlow policy
tmux new -s training

bash scripts/train_policy.sh maniflow_r3d_pointcloud_policy foil_lifting_augmented_r3d 0112 0 0
ctrl+b d

tmux attach -t training

# LeRobot SARM policy
lerobot-train --dataset.repo_id=Marco2929/foil_lifting_baseline --dataset.root=/root/.cache/huggingface/lerobot/Marco2929/foil_lifting_baseline --policy.type=sarm --policy.annotation_mode=single_stage --policy.image_key=observation.images.front --output_dir=outputs/train/sarm_single --batch_size=28 --steps=5000 --wandb.enable=true --wandb.project=sarm --policy.repo_id=your-hf-user/your-model-name --policy.push_to_hub=false

---

## Deployment

### Launch Stack

# Terminal 1: Robot hardware
ros2 launch il_main ur_inspire.launch.py

# Terminal 2: ZED camera
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i

# Terminal 3: Control node (inactive by default)
ros2 run il_main control_ur_inspire

# Terminal 4: Policy (runs inference continuously)
python third_party/ManiFlow_Policy/ManiFlow/deploy.py

### Standalone Control (without DAgger recorder)

# Start robot (open policy gate)
ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: true}"

# Stop robot (close policy gate)
ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: false}"

### DAgger Recording Mode

# Terminal 5: DAgger recorder (controls policy gate + teleop automatically)
python data/record_utils/record_dagger.py

---

## Debugging

ros2 run plotjuggler plotjuggler


# ─── Datensatz generieren ────────────────────────────
python data/dataset_builder/cli.py --config box_opening_baseline.yaml
python data/dataset_builder/cli.py --config foil_lifting.yaml

# ─── Episoden in Rohdaten umklassifizieren ───────────
# success → failed umbenennen (LeRobot-Indizes, 0-basiert, nur success gezählt)
python relabel_episodes.py 1 30 55 59 66

# Episoden komplett löschen
python relabel_episodes.py 1 30 55 59 66 --action delete

# Anderes Datenverzeichnis
python relabel_episodes.py 1 30 55 59 66 --data-dir /path/to/episodes