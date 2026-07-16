#!/usr/bin/env python3
import time
import threading
from queue import Queue

import math
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, String, Bool
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, SimpleFilter
import collections
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import message_filters

from visualization_msgs.msg import Marker

from model_serve import init_model, normalize
from cond_preprocess import PointCloudProcessor, setup_kinematics, compute_fk
from rtc_utils import LatencyTracker, RTCProcessor


class RobotCameraSync:
    """
    Synchronizes ZED cameras using strict timestamp matching (message_filters)
    and uses a delayed Time-Machine to wait out Python GIL / Executor thread starvation.
    """
    def __init__(self, node, callback_function, joint_names):
        self.node = node
        self.callback = callback_function
        self.joint_names = joint_names

        self.latest_joint_state_dict = {name: None for name in self.joint_names}
        self.joint_history = collections.deque(maxlen=500)
        self.camera_buffer = collections.deque(maxlen=15)

        qos_fast = QoSProfile(
            depth=500,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        qos_slow = QoSProfile(
            depth=15,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.sub_joint = node.create_subscription(
            JointState, '/joint_states', self._joint_cb, qos_fast)

        self.sub_color = message_filters.Subscriber(
            node, Image, '/zed/zed_node/rgb/color/rect/image', qos_profile=qos_slow)
        self.sub_depth = message_filters.Subscriber(
            node, Image, '/zed/zed_node/depth/depth_registered', qos_profile=qos_slow)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth],
            queue_size=15,
            slop=0.015
        )
        self.ts.registerCallback(self._camera_sync_cb)

        self.node.get_logger().info("RobotCameraSync initialized.")

    def _get_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_joint_state_dict:
                self.latest_joint_state_dict[name] = pos

        is_arm_msg = 'shoulder_pan_joint' in msg.name

        if is_arm_msg and all(val is not None for val in self.latest_joint_state_dict.values()):
            full_msg = JointState()
            full_msg.header = msg.header
            full_msg.name = list(self.latest_joint_state_dict.keys())
            full_msg.position = list(self.latest_joint_state_dict.values())

            self.joint_history.append(full_msg)
            self._try_process_buffers()

    def _camera_sync_cb(self, color_msg, depth_msg):
        self.camera_buffer.append((color_msg, depth_msg))
        self._try_process_buffers()

    def _try_process_buffers(self):
        while self.camera_buffer and self.joint_history:
            color_msg, depth_msg = self.camera_buffer[0]

            t_cam = self._get_sec(color_msg.header.stamp)
            t_newest_joint = self._get_sec(self.joint_history[-1].header.stamp)

            if t_newest_joint >= (t_cam - 0.002):
                color_msg, depth_msg = self.camera_buffer.popleft()

                closest_joint_msg = None
                min_time_diff = float('inf')

                for j_msg in self.joint_history:
                    t_joint = self._get_sec(j_msg.header.stamp)
                    diff = abs(t_cam - t_joint)
                    if diff < min_time_diff:
                        min_time_diff = diff
                        closest_joint_msg = j_msg

                if min_time_diff > 0.035:
                    self.node.get_logger().warn(
                        f"SEVERE SYNC MISMATCH! Gap: {min_time_diff*1000:.2f} ms")

                if closest_joint_msg is not None:
                    self.callback(closest_joint_msg, color_msg, depth_msg)
            else:
                break


class RealDexEnv(Node):
    """Environment management — policy runs continuously."""
    def __init__(self, cfg: Node, fps: float, progress_threshold: float = None):
        super().__init__('flow_policy')

        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.n_obs_steps = cfg.n_obs_steps
        self.n_action_steps = cfg.n_action_steps
        self.fps = fps
        self.progress_threshold = progress_threshold

        self.JOINT_NAMES = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
            'right_inspire_right_thumb_1_joint', 'right_inspire_right_thumb_2_joint',
            'right_inspire_right_index_1_joint', 'right_inspire_right_middle_1_joint',
            'right_inspire_right_ring_1_joint', 'right_inspire_right_little_1_joint'
        ]

        self.point_cloud_array = []
        self.robot_state_array = []

        self.pc_processor = PointCloudProcessor()
        self.bridge = CvBridge()

        self.syncer = RobotCameraSync(self, self.observation_callback, self.JOINT_NAMES)

        self.action_pub = self.create_publisher(
            Float64MultiArray, '/predicted_action_chunk', 10)

        # Progress publishers
        self.progress_pub = self.create_publisher(
            Float64MultiArray, '/policy_progress', 10)
        self.progress_marker_pub = self.create_publisher(
            Marker, '/policy_progress_marker', 10)
        self.done_pub = self.create_publisher(
            Bool, '/policy_done', 10)

        self.static_pc_tensor = None
        self.static_state_tensor = None

        self.status_pub = self.create_publisher(String, '/policy_status', 10)

        self.last_obs_time = 0.0
        self.target_dt = (1.0 / self.fps) - 0.005

        self.debug_pub = self.create_publisher(PointCloud2, '/debug_pc', 10)
        self.new_obs_event = threading.Event()
        self.debug_action_pub = self.create_publisher(PoseStamped, '/debug_action_pose', 10)
        self.chain = setup_kinematics()
        self.traj_pub = self.create_publisher(PoseArray, '/predicted_trajectory', 10)

        self.get_logger().info("RealDexEnv initialized. Policy will run continuously.")

    def publish_debug_pointcloud(self, point_cloud: np.ndarray):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"

        if point_cloud.shape[1] >= 6:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
            ]
            points = point_cloud[:, :6].astype(np.float32)
        else:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            points = point_cloud[:, :3].astype(np.float32)

        msg = point_cloud2.create_cloud(header, fields, points)
        self.debug_pub.publish(msg)

    def publish_status(self, status_text):
        msg = String()
        msg.data = status_text
        self.status_pub.publish(msg)

    def publish_progress(self, progress_score: float):
        """Publish progress score as Float64MultiArray and as RViz Marker."""
        msg = Float64MultiArray()
        msg.data = [progress_score]
        self.progress_pub.publish(msg)

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "policy_progress"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.08
        progress_clamped = max(0.0, min(1.0, progress_score))
        marker.color.a = 1.0
        marker.color.r = 1.0 - progress_clamped
        marker.color.g = progress_clamped
        marker.color.b = 0.0
        marker.text = f"Progress: {progress_score:.3f}"
        if self.progress_threshold is not None:
            marker.text += f" / {self.progress_threshold:.2f}"
        marker.lifetime.sec = 1
        self.progress_marker_pub.publish(marker)

    def check_progress_done(self, progress_score: float) -> bool:
        """Check if progress threshold is reached. Returns True if done."""
        if self.progress_threshold is not None and progress_score >= self.progress_threshold:
            self.get_logger().info(
                f"[DONE] Progress {progress_score:.3f} >= threshold {self.progress_threshold:.2f}. Stopping.")
            done_msg = Bool()
            done_msg.data = True
            self.done_pub.publish(done_msg)
            return True
        return False

    def observation_callback(self, joint_msg, color_msg, depth_msg):
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9

            if (current_time - self.last_obs_time) < self.target_dt:
                return

            self.last_obs_time = current_time

            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            pos_dict = dict(zip(joint_msg.name, joint_msg.position))
            ordered_state = [pos_dict[name] for name in self.JOINT_NAMES]
            robot_state = np.array(ordered_state, dtype=np.float32)

            cartesian_state_phys = compute_fk(self.chain, robot_state)

            point_cloud = self.pc_processor.process_rgbd(color_img, depth_img)

            if point_cloud is not None:
                self.publish_debug_pointcloud(point_cloud)

                self.point_cloud_array.append(point_cloud)
                self.robot_state_array.append(cartesian_state_phys)

                if len(self.point_cloud_array) > self.n_obs_steps:
                    self.point_cloud_array.pop(0)
                if len(self.robot_state_array) > self.n_obs_steps:
                    self.robot_state_array.pop(0)
                self.new_obs_event.set()
            else:
                print("[OBS ERROR] Point cloud generation returned None!")

        except Exception as e:
            self.get_logger().error(f"Observation processing error: {e}")

    def get_latest_obs(self):
        self.new_obs_event.wait()
        self.new_obs_event.clear()

        while len(self.robot_state_array) < self.n_obs_steps:
            time.sleep(0.01)

        return self._get_obs()

    def publish_action_chunk(self, full_action_sequence):
        msg = Float64MultiArray()
        msg.data = full_action_sequence[:, :13].flatten().tolist()
        self.action_pub.publish(msg)

    def _get_obs(self):
        pc_np = np.stack(self.point_cloud_array[-self.n_obs_steps:], axis=0)[np.newaxis, ...]
        state_np = np.stack(self.robot_state_array[-self.n_obs_steps:], axis=0)[np.newaxis, ...]

        pc_tensor_cpu = torch.from_numpy(pc_np)
        state_tensor_cpu = torch.from_numpy(state_np)

        if self.static_pc_tensor is None:
            self.static_pc_tensor = pc_tensor_cpu.to(self.device)
            self.static_state_tensor = state_tensor_cpu.to(self.device)
        else:
            self.static_pc_tensor.copy_(pc_tensor_cpu, non_blocking=True)
            self.static_state_tensor.copy_(state_tensor_cpu, non_blocking=True)

        return {
            'point_cloud': self.static_pc_tensor,
            'agent_pos': self.static_state_tensor
        }


def sport_mode(action_chunk: np.ndarray, speedup: float = 1.2) -> np.ndarray:
    """
    Resample action chunk to be shorter → faster execution at same Hz.
    
    Args:
        action_chunk: (T, action_dim) array
        speedup: 1.2 = 20% faster, 1.5 = 50% faster
    
    Returns:
        (T_new, action_dim) array where T_new = T / speedup
    """
    T, action_dim = action_chunk.shape
    T_new = int(T / speedup)
    
    # Original timesteps and new (compressed) timesteps
    t_original = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, T_new)
    
    # Linear interpolation per dimension
    resampled = np.zeros((T_new, action_dim))
    for dim in range(action_dim):
        resampled[:, dim] = np.interp(t_new, t_original, action_chunk[:, dim])
    
    return resampled


def run_policy(alg_name: str, task_name: str, run_dir: str, addition_info: int = 1211, fps: float = 30.0, sport_mode_active=False, progress_threshold: float = None, warmup_steps: int = 3, num_inference_steps: int = None, use_rtc: bool = True, rtc_execution_horizon: int = 16, rtc_max_guidance: float = 5.0):
    if not rclpy.ok():
        rclpy.init()

    cfg, policy = init_model(
        task_name=task_name,
        alg_name=alg_name,
        addition_info=addition_info,
        run_dir=run_dir
    )

    # Override inference steps (ManiFlow paper: 1-2 steps BETTER than 10 with consistency training)
    if num_inference_steps is not None:
        old_steps = getattr(policy, 'num_inference_steps', None)
        policy.num_inference_steps = num_inference_steps
        print(f"[INFO] num_inference_steps: {old_steps} → {num_inference_steps}")
    else:
        print(f"[INFO] num_inference_steps: {getattr(policy, 'num_inference_steps', '?')}")

    env = RealDexEnv(cfg=cfg, fps=fps, progress_threshold=progress_threshold)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(env)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    latency_tracker = LatencyTracker(maxlen=100)
    time_per_chunk = 1.0 / fps

    # Real-Time Chunking: guides new chunk to continue previous chunk's tail
    rtc = RTCProcessor(
        execution_horizon=rtc_execution_horizon,
        max_guidance_weight=rtc_max_guidance,
        prefix_attention_schedule="exp") if use_rtc else None
    prev_action_pred = None  # (B, T, Da) unnormalized full-horizon prediction from last step
    if use_rtc:
        print(f"[INFO] RTC ENABLED (exec_horizon={rtc_execution_horizon}, max_guidance={rtc_max_guidance})")
    else:
        print("[INFO] RTC DISABLED")

    try:
        env.publish_status("RUNNING")
        print("[INFO] Policy running continuously. Waiting for observations...")

        # Wait for initial observation buffer to fill
        obs = env.get_latest_obs()

        # Warmup: run inference a few times to stabilize CUDA kernels
        print(f"[INFO] Warming up model ({warmup_steps} steps)...")
        for i in range(warmup_steps):
            with torch.no_grad():
                _ = policy.predict_action(obs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{warmup_steps} done.")
        print("[INFO] Warmup complete. Starting real-time control.")

        while rclpy.ok():
            start_time = time.perf_counter()

            # 0. RTC PREP: compute inference delay from recent latency, slice leftover
            prev_leftover = None
            prev_leftover_np = None
            rtc_delay = 0
            if rtc is not None and prev_action_pred is not None:
                est_latency = latency_tracker.mean() if len(latency_tracker) > 0 else 0.0
                rtc_delay = math.ceil(est_latency / time_per_chunk)
                # Cap delay at execution_horizon: beyond that RTC has nothing
                # meaningful to bind to, and hard-binding the whole prefix hurts.
                rtc_delay = max(0, min(rtc_delay, rtc_execution_horizon,
                                       prev_action_pred.shape[1] - 1))
                # Leftover: unexecuted tail, aligned so leftover[0] == new_chunk[0]
                prev_leftover = prev_action_pred[:, rtc_delay:, :]
                prev_leftover_np = prev_leftover[0].cpu().numpy()  # (Tp, Da) for metric

            # 1. GPU INFERENCE
            t_model_start = time.perf_counter()
            with torch.no_grad():
                action_queue_raw = policy.predict_action(
                    obs,
                    rtc_processor=rtc,
                    prev_chunk_left_over=prev_leftover,
                    inference_delay=rtc_delay)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            time_model = time.perf_counter() - t_model_start

            # Save full-horizon prediction (unnormalized) for next step's RTC leftover
            prev_action_pred = action_queue_raw['action_pred'].detach()

            # 2. CPU POST-PROCESSING & PUBLISHING
            t_post_start = time.perf_counter()
            action_queue = normalize(action_queue_raw)

            if sport_mode_active:
                action_queue = sport_mode(action_queue, speedup=1.3)

            first_pose = action_queue[0][:3]
            last_pose = action_queue[-1][:3]

            print("\n" + "=" * 50)
            print(f"NEW CHUNK (Length: {len(action_queue)} steps)")
            print(f"Start (X,Y,Z): {np.round(first_pose, 4).tolist()}")
            print(f"End   (X,Y,Z): {np.round(last_pose, 4).tolist()}")

            # TRUE RTC CONTINUITY METRIC: compare new chunk to aligned leftover
            # at the SAME timesteps. RTC should make these match at the start.
            if prev_leftover_np is not None:
                k = min(rtc_execution_horizon, len(action_queue), len(prev_leftover_np))
                if k > 0:
                    new_xyz = action_queue[:k, :3]
                    old_xyz = prev_leftover_np[:k, :3]
                    per_step = np.linalg.norm(new_xyz - old_xyz, axis=1)
                    print(f"[RTC CONTINUITY over {k} steps] "
                          f"step0={per_step[0]*1000:.1f}mm "
                          f"mean={per_step.mean()*1000:.1f}mm "
                          f"max={per_step.max()*1000:.1f}mm")

            if hasattr(env, 'last_published_end'):
                gap = np.linalg.norm(first_pose - env.last_published_end)
                print(f"GAP TO OLD CHUNK: {gap:.4f} m")

            env.last_published_end = last_pose
            print("=" * 50 + "\n")

            # Publish progress + check done
            if 'progress_score' in action_queue_raw:
                progress_val = float(action_queue_raw['progress_score'][0].item())
                env.publish_progress(progress_val)
                thr_str = f" / {env.progress_threshold:.2f}" if env.progress_threshold is not None else ""
                bar_len = 20
                filled = int(max(0.0, min(1.0, progress_val)) * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"[PROGRESS] {bar} {progress_val:.3f}{thr_str}")
                if env.check_progress_done(progress_val):
                    env.publish_status("DONE")
                    break

            inference_time = time.perf_counter() - start_time
            latency_tracker.add(inference_time)

            max_latency = latency_tracker.max()
            delay_steps = math.ceil(inference_time / time_per_chunk)

            if delay_steps < len(action_queue):
                future_actions = action_queue[delay_steps:]
            else:
                print(f"[WARN] Latency extremely high! ({max_latency:.3f}s).")
                future_actions = action_queue[-1:]

            rtc_tag = f"RTC(d={rtc_delay}) " if rtc is not None else ""
            print(f"[SYNC] {rtc_tag}Inference: {inference_time:.3f}s | Shift: {delay_steps} | Sending: {len(future_actions)}")

            env.publish_action_chunk(future_actions)
            time_post = time.perf_counter() - t_post_start

            # 3. WAIT FOR NEXT OBSERVATION
            t_obs_start = time.perf_counter()
            obs = env.get_latest_obs()
            time_obs = time.perf_counter() - t_obs_start

            print(f"[PROFILING] GPU: {time_model*1000:.1f} ms | Post: {time_post*1000:.1f} ms | Obs: {time_obs*1000:.1f} ms")

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    run_policy(
        alg_name="maniflow_force_pointcloud_policy_v2",
        task_name="foil_lifting_augmented_v2",
        run_dir="/ros2_ws/data/outputs/foil_lifting_augmented_v2-maniflow_force_pointcloud_policy_v2-0112_seed0",
        addition_info=1211,
        fps=20.0,
        progress_threshold=0.9,
        warmup_steps=3,
        num_inference_steps=10,
        use_rtc=True,
        rtc_execution_horizon=16,
        rtc_max_guidance=5.0,
    )

