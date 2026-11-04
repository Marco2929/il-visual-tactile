# Production Deployment & Integration Guide: ManiFlow Policy Inference

Operational reference for deploying the policy inference container and integrating it with MoveIt Pro, Cartesian controllers, and cameras.

---

## 1. Testing the Inference Server (Without Hardware Drivers)

Do **NOT** start hardware drivers (UR robot driver, hand driver) when testing with MoveIt Pro, as MoveIt Pro's drivers container owns the robot hardware.

### Start the Inference Container

```bash
cd docker/production
docker compose up -d
docker compose logs -f policy_inference
```

Verify discovery from the MoveIt Pro container:
```bash
moveit_pro shell
ros2 action list
# /execute_policy must appear
```

### Test Action Server & VRAM Preload

Verify policy weight loading into VRAM (no camera or joint observations required):
```bash
ros2 action send_goal /execute_policy maniflow_msgs/action/ExecutePolicy \
  "{task_name: 'foil_lifting_wrist', preload_only: true}"
```
*Expected Result:* `success=true`, `termination_reason: PRELOADED`.

---

## 2. Controller Integration: Cartesian Motion Controller & Hand

The policy executor commands end-effector target poses to the FZI `cartesian_motion_controller` on topic `/target_frame` (`geometry_msgs/PoseStamped`).

### A. Clone & Build `cartesian_controllers`
Clone the official ROS 2 branch into your MoveIt Pro workspace:
```bash
cd <PREHENSIO_WS>/src
git clone -b ros2 https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers.git
touch cartesian_controllers/cartesian_controller_simulation/COLCON_IGNORE
touch cartesian_controllers/cartesian_controller_tests/COLCON_IGNORE
```

Build within MoveIt Pro:
```bash
moveit_pro build user_workspace --packages-up-to cartesian_motion_controller
```

### B. Controller Parameters Reference

The configuration parameters required for the robot and hand controllers:

#### 1. Arm Cartesian Motion Controller (`cartesian_motion_controller`)
* **Controller Type:** `cartesian_motion_controller/CartesianMotionController`
* **Frames:**
  * `end_effector_link`: `"tool0"`
  * `robot_base_link`: `"base_link"`
* **Solver & Gains:**
  * `solver.error_scale`: `1.0`
  * `solver.iterations`: `10`
  * `solver.publish_state_feedback`: `True`
  * `pd_gains.trans_x`: `{p: 2.0}`
  * `pd_gains.trans_y`: `{p: 2.0}`
  * `pd_gains.trans_z`: `{p: 2.0}`
  * `pd_gains.rot_x`: `{p: 0.5}`
  * `pd_gains.rot_y`: `{p: 0.5}`
  * `pd_gains.rot_z`: `{p: 0.5}`
* *Reference file in repo:* [`src/cartesian_controllers_universal_robots/config/controller_manager.yaml`](file:///home/mm/dev/il-visual-tactile/src/cartesian_controllers_universal_robots/config/controller_manager.yaml)

#### 2. Hand Controller (`dg5f_right_controller`)
* **Controller Type:** `joint_trajectory_controller/JointTrajectoryController`
* **Command Interfaces:** `effort`
* **State Interfaces:** `position`, `velocity`
* **Rate:** `state_publish_rate: 500.0`, `action_monitor_rate: 500.0`
* **Joints (20 DoF):** `rj_dg_1_1` .. `rj_dg_1_4` through `rj_dg_5_1` .. `rj_dg_5_4`
* **PID Gains per Joint:**
  * `p: 4.5, i: 0.0, d: 0.02, i_clamp_max: 0.0, i_clamp_min: 0.0, ff_velocity_scale: 0.0001`
* *Reference file in repo:* [`src/hands/delto_m_ros2/dg5f_ros2/dg5f_driver/config/dg5f_right_controller.yaml`](file:///home/mm/dev/il-visual-tactile/src/hands/delto_m_ros2/dg5f_ros2/dg5f_driver/config/dg5f_right_controller.yaml)

---

## 3. Camera Integration: ZED 2i & Wrist Camera

During active execution (without `preload_only: true`), `deploy.py` synchronizes 3 camera streams and joint states at 20 Hz (`ApproximateTimeSynchronizer`, slop 30 ms):

### A. Required Topics
* `/joint_states` (`sensor_msgs/JointState`, RELIABLE)
* `/zed/zed_node/rgb/color/rect/image` (`sensor_msgs/Image`, RELIABLE)
* `/zed/zed_node/depth/depth_registered` (`sensor_msgs/Image`, RELIABLE)
* `/cam_gemini/color/image_raw` (`sensor_msgs/Image`, override via `WRIST_COLOR_TOPIC`)

### B. ZED 2i Settings (from `data/record_utils/record.py`)
Demonstration data was recorded with:

* **Resolution**: `HD720` (1280x720)
* **Frame rate**: `30` or `60` FPS
* **Depth mode**: `NEURAL`
* **Depth range**: `min_depth: 0.2 m` (200 mm), `max_depth: 2.0 m` (2000 mm)

Launch command:
```bash
ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zed2i \
  camera_name:=zed \
  node_name:=zed_node
```
