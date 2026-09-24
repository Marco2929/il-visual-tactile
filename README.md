# il-visual-tactile

Visual-tactile imitation learning on a UR10e workcell with interchangeable
multi-finger hands (Tesollo DG5F, Inspire), SenseGlove-based teleoperation and
policy deployment.

This document is a task-oriented reference: each section covers one stage of the
workflow, from bringing up hardware to training and deploying a policy.

## Contents

1. [Hardware drivers](#1-hardware-drivers)
2. [Robot model, MoveIt and controllers](#2-robot-model-moveit-and-controllers)
3. [Camera calibration](#3-camera-calibration)
4. [Teleoperation](#4-teleoperation)
5. [Recording and dataset preparation](#5-recording-and-dataset-preparation)
6. [Training](#6-training)
7. [Deployment](#7-deployment)
8. [Debugging](#8-debugging)

---

## 1. Hardware drivers

### Tesollo DG5F (right hand)

Communicates over Modbus TCP.

```bash
ros2 launch dg5f_driver dg5f_right_driver.launch.py \
  delto_ip:=192.168.56.102 delto_port:=502 fingertip_sensor:=true
```

`fingertip_sensor:=true` enables the fingertip force/torque topics, which the
recording scripts consume.

### Inspire hand

```bash
ros2 run inspire_hand_modbus_ros2 hand_modbus_control_node
ros2 run inspire_hand_modbus_ros2 inspire_hand_modbus_topic.py
ros2 run inspire_hand_modbus_ros2 touch_data.py
```

Send a test pose to verify the hand responds:

```bash
ros2 topic pub --once /set_angle_data service_interfaces/msg/SetAngle1 \
  "{finger_ids: [1, 2, 3, 4, 5, 6], angles: [1000, 1000, 1000, 1000, 1000, 000]}"
```

### SenseGlove (R1 glove + Tesollo mapper)

Run in containers via the distribution launcher. Start the glove first, then the
hand:

```bash
uv run start.py r1 --num-gloves 1
uv run start.py tesollo --delto-ip 192.168.56.102 --hand right --glove-id 70
```

See [`src/senseglove_distribution/README.md`](src/senseglove_distribution/README.md)
for the lab-specific setup and the reasoning behind each argument.

---

## 2. Robot model, MoveIt and controllers

### Inspecting and generating the robot description

The workcell URDF is assembled from an arm, a hand and a flange adapter.
Visualise a combination before committing to it:

```bash
ros2 launch moveit_ur_hand_config view_robot.launch.py \
  ur_type:=ur10e hand_type:=inspire_strong_hand flange_type:=ring
```

Generate the flattened URDF used by the launch files and the policy code:

```bash
./src/moveit_ur_hand_config/scripts/generate_urdf.sh ur10e inspire_strong_hand ring
```

### Full stack with MoveIt

Each launch file brings up the arm, the selected hand and MoveIt together:

```bash
ros2 launch il_main ur_tesollo_moveit.launch.py
ros2 launch il_main ur_inspire_moveit.launch.py
```

### Switching controllers

Trajectory control and direct position streaming are mutually exclusive, so
switching requires deactivating one while activating the other:

```bash
# To trajectory control (MoveIt, teleop)
ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate scaled_joint_trajectory_controller

# To direct position streaming
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller
```

---

## 3. Camera calibration

Two camera mountings are supported and each uses a different calibration model:
the ZED is fixed in the scene (hand-to-eye), the RealSense is mounted on the
wrist (eye-in-hand).

### Starting the calibration node

```bash
ros2 launch calibration_camera calibration.launch.py camera:=zed
ros2 launch calibration_camera calibration.launch.py camera:=realsense
```

### Workflow

**Step 1 - collect samples.** Move the robot to a new pose and log it. Repeat
8-20 times, spreading the poses across the workspace and varying orientation:

```bash
ros2 service call /calibration/log_data interfaces/srv/LogCalibrationData "{}"
```

**Step 2 - check progress** at any point:

```bash
ros2 service call /calibration/get_status interfaces/srv/GetCalibrationStatus "{}"
```

**Step 3 - save the collected data** under a named run:

```bash
ros2 service call /calibration/save_data interfaces/srv/SaveCalibrationData \
  "{folder_name: 'run01'}"
```

**Step 4 - solve.** The `hand_in_eye` flag selects the model, so it must match
the camera being calibrated:

```bash
# RealSense: eye-in-hand
ros2 service call /calibration/run_calibration interfaces/srv/RunCalibration \
  "{hand_in_eye: true, method: '', data_dir: ''}"

# ZED: hand-to-eye
ros2 service call /calibration/run_calibration interfaces/srv/RunCalibration \
  "{hand_in_eye: false, method: '', data_dir: '/ros2_ws/data/calibration/realsense_eye_in_hand'}"
```

---

## 4. Teleoperation

### Bring-up

Start the robot stack and the glove tracking:

```bash
ros2 launch il_main ur_tesollo_moveit.launch.py
ros2 launch r1_bringup r1.launch.py
ros2 launch senseglove_bringup senseglove.launch.py
```

### Running

Move the arm to a defined starting pose, then start the teleop node:

```bash
python3 src/il_main/il_main/utils/move_utils/move_robot_pose.py
ros2 run il_main teleop_ur
```

`teleop_ur` stays idle until it is enabled - the recorder publishes the enable
signal, so during data collection it is armed automatically.

---

## 5. Recording and dataset preparation

### Environment

These variables select the output folder and the kinematic model, and must be set
in every shell that runs the recording or post-processing scripts:

```bash
export FOLDER_NAME="dataset_collection"
export URDF_PATH="/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf"
export ROBOT_TIP_FRAME="right_inspire_base_link"
```

`FOLDER_NAME` is set per collection campaign, for example:

```bash
export FOLDER_NAME="foil_lifting_collection"
export FOLDER_NAME="box_opening_collection"
```

### Recording

```bash
python data/record_utils/record.py
```

### Post-processing

Run in order: discard unwanted takes, derive point clouds, then merge into a
dataset.

```bash
python data/record_utils/remove_recordings.py
python data/record_utils/generatePointCloud.py
python data/record_utils/merge_episodes.py
```

To produce a LeRobot-format dataset instead:

```bash
python data/record_utils/merge_episodes_lerobot.py \
  --repo_id your-hf-user/box_opening_collection --task "test test" --fps 30
```

### Building configured datasets

```bash
python data/dataset_builder/cli.py --config box_opening_baseline.yaml
python data/dataset_builder/cli.py --config foil_lifting.yaml
```

### Relabelling episodes

Indices refer to LeRobot indices: zero-based, counting successful episodes only.

```bash
# Mark as failed (renames success -> failed)
python relabel_episodes.py 1 30 55 59 66

# Delete outright
python relabel_episodes.py 1 30 55 59 66 --action delete

# Operate on a different episode directory
python relabel_episodes.py 1 30 55 59 66 --data-dir /path/to/episodes
```

### Inspecting data

```bash
# Single point-cloud frame
python data/record_utils/visualize/visualize_pointcloud.py \
  --path /ros2_ws/data/episodes/foil_lifting_collection/episode_001_success/point_cloud/frame_0001.npy

# A LeRobot dataset episode
lerobot-dataset-viz --repo-id Marco2929/box_opening_baseline --episode-index 0 \
  --root /root/.cache/huggingface/lerobot/Marco2929/box_opening_baseline --mode local
```

---

## 6. Training

### ManiFlow policy

Training runs for a long time, so start it inside a tmux session and detach with
`Ctrl+b d`:

```bash
tmux new -s training
bash scripts/train_policy.sh maniflow_r3d_pointcloud_policy foil_lifting_augmented_r3d 0112 0 0
```

Reattach later with:

```bash
tmux attach -t training
```

### LeRobot SARM policy

```bash
lerobot-train \
  --dataset.repo_id=Marco2929/foil_lifting_baseline \
  --dataset.root=/root/.cache/huggingface/lerobot/Marco2929/foil_lifting_baseline \
  --policy.type=sarm \
  --policy.annotation_mode=single_stage \
  --policy.image_key=observation.images.front \
  --output_dir=outputs/train/sarm_single \
  --batch_size=28 --steps=5000 \
  --wandb.enable=true --wandb.project=sarm \
  --policy.repo_id=your-hf-user/your-model-name --policy.push_to_hub=false
```

---

## 7. Deployment

### Launching the stack

Four terminals, in this order:

| Terminal | Purpose | Command |
|---|---|---|
| 1 | Robot hardware | `ros2 launch il_main ur_tesollo_cartesian.launch.py` |
| 2 | ZED camera | `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i` |
| 3 | Control node | `ros2 run il_main cartesian_inference_control` |
| 4 | Policy inference | `python third_party/ManiFlow_Policy/ManiFlow/deploy.py` |

The control node starts inactive: the policy runs inference continuously, but no
motion is executed until the policy gate is opened.

### Standalone control

Without the DAgger recorder, the gate is toggled manually:

```bash
# Start moving
ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: true}"

# Stop moving
ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: false}"
```

### DAgger recording mode

The recorder drives the policy gate and teleop handover itself, so the manual
gate commands above are not needed:

```bash
python data/record_utils/record_dagger.py
```

---

## 8. Debugging

Live plotting of topics:

```bash
ros2 run plotjuggler plotjuggler
```