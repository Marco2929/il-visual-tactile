# Production Inference Container and MoveIt Pro Action Server Architecture for ManiFlow Deployment

* **Author:** mm
* **Date:** 2026-11-03

## Overview

Transition ManiFlow deployment from interactive, terminal-driven lab scripts (`deploy.py`, manual `/policy_gate` topics) into an industrial-grade, containerized ROS 2 Jazzy service orchestrated by MoveIt Pro Behavior Trees (BT).

The system consists of three decoupled layers:
1. **Production Inference Container (`Dockerfile.Inference`)**: Minimal, headless ROS 2 Jazzy container running PyTorch, CUDA, ManiFlow, and a standardized ROS 2 Action Server (`ExecutePolicy.action`).
2. **MoveIt Pro Behavior Tree Integration**: The BT orchestrates execution through the Action interface, with dynamic goal parameterization, live progress streaming (`progress_score`), and preemption/cancellation handling.
3. **Cartesian Controller & Executor Isolation (`control_ur_tesollo_cartesian.py`)**: Runs on the controller/MoveIt Pro level, enforcing the 20 Hz real-time loop, low-pass trajectory smoothing, and immediate safety hold on chunk starvation or preemption.

---

## Analysis

Synthesized across the multi-lens Goldfish reviews (`egm-design`, `egm-codebase-details`, `egm-zoom-out`, `egm-comprehension`, `egm-implementation-readiness`, `egm-structural`):

### 1. The "0 MB Warm Idle" Fallacy & Latency Impact
* **Critique:** Completely clearing VRAM down to 0 MB via `torch.cuda.empty_cache()` does not return virtual memory to the Linux kernel and introduces a 4–8 second latency penalty (disk I/O, PyTorch deserialization, and 3 steps of ODE flow matching warmup) on every single BT execution.
* **Resolution:**
  - Support **Model Residency with LRU Caching**: Keep the active task model resident in VRAM (~2–3 GB VRAM).
  - Add a `preload_only: bool` flag to `ExecutePolicy.action`. This allows MoveIt Pro to trigger model preloading in the background while the arm executes its gross Cartesian trajectory to the pre-grasp waypoint.
  - If hard VRAM reclamation is strictly required between shifts, isolate the inference loop into a worker subprocess (`torch.multiprocessing`) so OS memory is cleanly reclaimed upon termination.

### 2. Action Buffer Preemption & In-Flight Chunk Flushing
* **Critique:** `control_ur_tesollo_cartesian.py` buffers 16–26 actions (0.8–1.3 seconds of trajectory). If MoveIt Pro cancels an action goal and only the inference loop stops, the robot will continue executing in-flight motion until starvation.
* **Resolution:**
  - The Action Server `cancel_callback` must explicitly publish `std_msgs/Bool(data=True)` to `/policy_done` and `std_msgs/Bool(data=False)` to `/policy_gate`.
  - In `control_ur_tesollo_cartesian.py`, `_policy_done_cb` calls `_deactivate()`, which triggers `self.schedule.flush()` and locks the robot at its current Cartesian pose immediately.

### 3. Action Goal Parameter Surface
* **Critique:** `init_model()` in `deploy.py` requires more than just `task_name` and `ckpt_file`. It relies on Hydra configs needing `alg_name` (e.g. `maniflow_multimodal_policy`), `addition_info` (e.g. `1211`), and ODE parameters.
* **Resolution:**
  - Structure `ExecutePolicy.action` to include defaultable task metadata:
    - `string task_name`
    - `string run_dir`
    - `string ckpt_file`
    - `string alg_name "maniflow_multimodal_policy"`
    - `int32 addition_info 1211`
    - `int32 num_inference_steps 10`
    - `float32 progress_threshold 0.90`
    - `int32 progress_patience 3`
    - `builtin_interfaces/Duration max_duration`
    - `bool preload_only false`

### 4. High-Bandwidth Camera Transport & CycloneDDS SHM
* **Critique:** `RobotCameraSync` synchronizes 3 camera streams (ZED color 720p + depth + wrist RGB) at 20 Hz. Over standard Docker UDP loopback, kernel packet buffers overflow, causing dropped frames and sync failures.
* **Resolution:**
  - Docker container execution must mandate `--ipc=host -v /dev/shm:/dev/shm` in addition to `--net=host`.
  - Standardize `cyclonedds.xml` to enable Iceoryx / shared memory transport across container boundaries.

### 5. Kinematics URDF Dependency
* **Critique:** `cond_preprocess.py` uses `kinpy` to compute Forward Kinematics (FK) for the 7-DoF EE pose (`tool0`). It requires an accurate URDF matching the dataset.
* **Resolution:**
  - Embed the workcell URDF (`tesollo_hand_ur10e_flange_workcell.urdf`, ~45 KB) directly into the inference image at `/opt/robot_model/...` or set `URDF_PATH` via environment variable. Meshes are not required for `kinpy`.

---

## Goals vs. Non-Goals

* **In-Scope Goals:**
  * Dedicated, lightweight `Dockerfile.Inference` containing only PyTorch, CUDA, ManiFlow, and minimal ROS 2 Jazzy.
  * Standardized `ExecutePolicy.action` interface for MoveIt Pro Behavior Tree orchestration.
  * Dynamic VRAM lifecycle with warmup, model caching, and concurrent preloading (`preload_only`).
  * Real-time progress feedback streaming (`progress_score: float32`) to the Behavior Tree.
  * Robust preemption and safety holding via `/policy_done` triggering `schedule.flush()`.
  * Clean separation: `TesolloCartesianExecutor` remains outside the inference container.

* **Out-of-Scope Non-Goals:**
  * Packaging hardware drivers (UR robot driver, Tesollo DG-5F driver, camera drivers) inside the inference container.
  * Modifying the core ManiFlow model architecture or training pipelines.
  * Moving MoveIt Pro's Cartesian controller into Python.

---

## Alternatives Weighed

1. **Monolithic Container (Running Drivers, Executor, and Policy together)**:
   * *Rejected by author:* Violates fault isolation. A CUDA OOM or inference crash would kill the real-time Cartesian controller, risking uncontrolled hardware movement.
2. **Subprocess / `docker exec` via MoveIt Pro Behavior Tree**:
   * *Rejected by review:* Incurs 5–10s cold-start delays, requires exposing Docker daemon socket (`docker.sock`) into MoveIt Pro, and prevents real-time streaming feedback of the task progress score.
3. **Hard 0 MB VRAM Purge between Actions (`torch.cuda.empty_cache()` inside the main process)**:
   * *Rejected by review:* Does not release virtual memory back to the OS and introduces unacceptable cycle-time stalls (4–8s) at pre-grasp waypoints. Replaced with model residency / LRU caching + optional `preload_only` trigger.

---

## Key Decisions & Staging for Next Steps

* **Decision 1:** Implement a ROS 2 Action Server node (`policy_action_server.py`) wrapping the DRTC loop from `deploy.py`.
* **Decision 2:** Isolate hardware controllers and `TesolloCartesianExecutor` on the MoveIt Pro / controller level; communicate with the inference container solely over ROS 2 topics and actions.
* **Decision 3:** Configure CycloneDDS shared memory (`--ipc=host`, `/dev/shm`) to support 20 Hz multi-camera streaming without frame drops.
* **Decision 4:** Model checkpoints mounted as read-only volumes from host (`/ros2_ws/data/outputs:ro`).

