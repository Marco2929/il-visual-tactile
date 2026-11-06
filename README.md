# il-flexible-object-grasping

Imitation learning on a UR10e workcell with interchangeable
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
8. [Rollout logging and analysis](#8-rollout-logging-and-analysis)

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

Send a test trajectory to verify the hand responds:

```bash
ros2 topic pub --once /dg5f_right/dg5f_right_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory "{
  joint_names: [
    rj_dg_1_1, rj_dg_1_2, rj_dg_1_3, rj_dg_1_4,
    rj_dg_2_1, rj_dg_2_2, rj_dg_2_3, rj_dg_2_4,
    rj_dg_3_1, rj_dg_3_2, rj_dg_3_3, rj_dg_3_4,
    rj_dg_4_1, rj_dg_4_2, rj_dg_4_3, rj_dg_4_4,
    rj_dg_5_1, rj_dg_5_2, rj_dg_5_3, rj_dg_5_4],
  points: [{
    positions: [0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0],
    velocities: [0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0],
    time_from_start: {sec: 2, nanosec: 500000000}}]}"
```

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

### SenseGlove + Tesollo

```bash
cd ~/dev/il-visual-tactile/src/senseglove_distribution && uv run start.py r1 --tag cyclone-latest --num-gloves 1
cd ~/dev/il-visual-tactile/src/senseglove_distribution && uv run start.py tesollo --tag cyclone-latest --hand right --glove-id 70 --delto-ip 192.168.56.102
bash senseglove_guard/run_watchdog.sh
```

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
export FOLDER_NAME="foil_lifting_collection_tesollo_middle"
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

To produce a LeRobot-format dataset instead run this command in lerobot-dataset-visualizer:

```bash
bun dev
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

# Interactive progress label viewer (with camera feeds and grasp signals)
export DISPLAY=:1
python data/record_utils/label_viewer.py --review
python data/record_utils/label_viewer.py --only folien_unten_b_sauele/episode_041_success
```

*Controls:* `Arrow keys` (frame +/-1), `Shift + Arrows` (+/-10), `a`/`d` (next/previous event), `n`/`p` (episode), `q` (quit).

---

## 6. Training

### Multi-GPU Training Queue & Night-Window Dispatcher (Standard)

All multi-run training campaigns are managed by the always-on Multi-GPU Queue (`scripts/gpu_queue.py` + `scripts/gpu_queue.yaml`), which runs permanently as a `systemd --user` background service (`gpu-queue.service`):

- **Local GPU (`local_pc`):** Dispatches inside the `ros2_dev` Docker container **24/7 (`schedule: "always"`)** whenever free VRAM and GPU utilization thresholds are met.
- **Remote colleague GPUs (e.g. `monty`):** Dispatches over SSH **only during the night window (`schedule: "18:00-05:00"`) and on weekends**, after verifying via `nvidia-smi` that no one else is using the GPU. Automatically syncs code and missing `.zarr` datasets before training and pulls finished checkpoints back to `data/outputs/` via `rsync`.

```bash
# Live status of all GPUs, time windows, and queued/running jobs
python3 scripts/gpu_queue.py status

# Enqueue a new training run (picked up automatically within 30s)
python3 scripts/gpu_queue.py add \
  --id 0031_my_experiment \
  --alg ablation/abl_rel_wrist_height_delta_to4 \
  --task ablation/abl_rel_wrist_height_delta_to4 \
  --dataset foil_lifting_box_recorded_action.zarr \
  --extra-args "policy.state_history=true"

# Reset a job (or 'all') back to pending
python3 scripts/gpu_queue.py reset 0031_my_experiment

# Manage the always-on background service
systemctl --user status gpu-queue.service
systemctl --user restart gpu-queue.service
```

To add another remote GPU machine, copy the `monty` worker block in [`scripts/gpu_queue.yaml`](scripts/gpu_queue.yaml) and set `target: <ssh-host>`.

### Single manual ManiFlow policy run

For a single manual run outside the queue, start it inside a tmux session and detach with
`Ctrl+b d`:

```bash
tmux new -s training
bash scripts/train_policy.sh maniflow_multimodal_policy foil_lifting_wrist 0112 0 0
```

Reattach later with:

```bash
tmux attach -t training
```

### Evaluating checkpoints

The run ranks checkpoints by `val_loss` (flow/velocity loss), which does not
necessarily match the action error after the full ODE rollout. This script
computes the action MSE on held-out episodes for every saved checkpoint and
prints the best one:

```bash
python third_party/ManiFlow_Policy/ManiFlow/scripts/sweep_checkpoints.py \
  --ckpt-dir /ros2_ws/data/outputs/<run>/checkpoints --n-batches 5
```

### Progress estimation head (DINOv3)

Trained completely independently of the policy on offline cached DINOv3 wrist features.

#### 1. Generate or update progress labels (optional)

```bash
python third_party/ManiFlow_Policy/ManiFlow/scripts/generate_progress_labels.py \
  --zarr_path /ros2_ws/data/zarr_datasets/foil_lifting_baseline_cleaned.zarr
```

#### 2. Cache raw DINOv3 wrist patch tokens (one-time offline run)

```bash
python third_party/ManiFlow_Policy/ManiFlow/scripts/cache_wrist_dinov3.py \
  --zarr_path /ros2_ws/data/zarr_datasets/foil_lifting_baseline_cleaned.zarr \
  --key_out wrist_dinov3_tokens \
  --batch_size 64
```

#### 3. Train the progress head

```bash
python third_party/ManiFlow_Policy/ManiFlow/scripts/train_progress_head.py \
  --zarr_path /ros2_ws/data/zarr_datasets/foil_lifting_baseline_cleaned.zarr \
  --token_key wrist_dinov3_tokens \
  --save_dir checkpoints/progress_head \
  --epochs 30 --lr 1e-3
```

Generates `best_progress_head.pt` and `val_predictions.png` in `--save_dir`.

#### 4. Configure policy & deploy

Set the checkpoint in `maniflow_multimodal_policy.yaml`:

```yaml
policy:
  progress_checkpoint: "checkpoints/progress_head/best_progress_head.pt"
```

Deploy with progress monitoring / threshold:

```bash
python third_party/ManiFlow_Policy/ManiFlow/deploy.py \
  --use-progress \
  --progress-threshold 0.90
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
| 5 | Temperature watchdog | `bash senseglove_guard/run_watchdog.sh --table` |

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

## 8. Rollout logging and analysis

### Recording a rollout

Start the logger **before** opening the policy gate: `ros2 topic pub --once`
is volatile, so a subscriber that joins afterwards never sees the message.

```bash
python data/rollout_logs/log_rollout.py \
  --ckpt  "epoch=0040-val_action_mse=0.022836" \
  --label "foil_lifting_relative_e40" \
  --run-dir /ros2_ws/data/outputs/foil_lifting_relative-maniflow_multimodal_policy-0112_seed0 \
  --inference-steps 10 \
  --notes "Palette Nominalhoehe"
```

`--notes` becomes the label in every plot and table, so record the condition
that was varied (pallet height, foil thickness) rather than leaving it empty.

Each gate open/close pair writes one `rollout_NNN/` directory. `Ctrl-C` closes
a rollout that is still open.

```bash
ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: true}"    # start
ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: false}"   # stop
```

### Labeling success and behaviour

To label success and qualitative behaviour, a separate script extracts the info from the policy run:

```bash
python3 data/rollout_logs/annotate_rollouts.py --ckpt "foil_lifting_relative_e40"
```

### Analysing rollouts

Note the interpreter: the logger needs `rclpy` (system `python`), the viewer
needs `plotly`/`rerun` (venv only).

```bash
python rollout_viz.py table              # per-rollout numbers + aggregates
python rollout_viz.py dash               # -> reports/<ckpt>.html, opens via file://
python rollout_viz.py filters 10         # what the executor's filters cost
python rollout_viz.py rerun 10 --save reports/r010.rrd
```

`dash` compares all rollouts (scatter / single rollout / sortable table),
`rerun` is for understanding one: 3D path, point cloud and all time series on
one timeline. Rerun runs headless here, so write an `.rrd` and open it on the
host with `rerun data/rollout_logs/reports/r010.rrd`.

Rollouts can be named `rollout_010`, `010` or `10`; without a name the last
one is used. Add `--video` to overlay the wrist camera (off by default - the
bags are ~1 GB each; `--max-frames`, default 200, spreads frames over the
run). Generated files land in `reports/` and are gitignored.

`rerun` also plots the **pinch width** (thumb tip to index tip, mm) via URDF
forward kinematics, because the closure metric sums joint angles and a sum is
not a grasp width: a curled fingertip and a flat finger can sum alike while
gripping very differently. The URDF is found automatically (installed share
dir or source checkout); override with `--urdf` / `$DG5F_URDF`, or skip the
kinematics with `--no-pinch`.

```bash
rerun - < data/rollout_logs/reports/r010.rrd
```

or raise the limit once:

```bash
echo 'fs.inotify.max_user_instances=1024' | sudo tee /etc/sysctl.d/99-inotify.conf
sudo sysctl --system
```
