#!/usr/bin/env python3
"""
DRTC-style Policy Inference Node for ManiFlow - BASE-RELATIVE variant.

Copy of deploy.py for the relative-EE-pose experiment. deploy.py itself is
untouched, so the absolute-pose deployment and its RTC path keep working.

Differences from deploy.py:
  - The policy consumes and produces 17-dim vectors (xyz 3 + rot6d 6 +
    hand 8) expressed relative to the last OBSERVED EE pose, matching
    RelativeRealDexDataset.
  - Observations are relativised before predict_action(); predictions are
    decoded back to absolute 15-dim (xyz + quat + hand) before
    expand_hand_action(), so the wire format to the executor is unchanged.
  - sport_mode is not available here.
  - RTC needs re-basing; see RTC_MODE below.

RTC_MODE
--------
rtc_processor.guide() runs INSIDE sample_ode, once per denoising step, with
prev_chunk_left_over as an inpainting target in the flow field. It is not a
post-hoc cross-fade, so the leftover has to be in the same frame the model
is currently denoising in.

Every chunk is relative to its own observation base. A leftover from the
previous chunk lives in the PREVIOUS base, so passing it through unchanged
asks the model to match coordinates that are offset by whatever motion
happened during the inference delay - a silent, systematic drift.

  "off"     RTC disabled. Verifies encode/decode in isolation. Chunk
            boundaries are visibly less smooth.
  "rebase"  The decoded ABSOLUTE chunk is cached and its unexecuted tail is
            re-expressed in the NEW observation base before being handed
            back. Correct; use once encode/decode is confirmed.

Adapted from the Interlatent DRTC architecture:
  - Observation capture runs continuously (callback-driven)
  - Inference is DEMAND-DRIVEN: triggered when the executor's buffer
    drops below the execution horizon (not by observation arrival)
  - Jacobson-Karels latency estimator (decays after spikes, unlike max)
  - Cooldown counter prevents double-firing during inference
  - RTC (Real-Time Chunking) guides new chunks to continue previous ones

Architecture:
  [Camera + Joints] --> observation_callback --> [obs buffer]
                                                      |
  /action_buffer_status --> [buffer_depth < horizon?] |
                                       |              |
                                  [cooldown ready?]   |
                                       |              |
                                  [trigger inference] <-
                                       |
                                  [GPU: policy.predict_action()]
                                       |
                                  /predicted_action_chunk --> [Executor Node]

Key differences from old design:
  - Inference not blocked on observation arrival
  - Buffer depth is the natural backpressure / scheduling signal
  - Cooldown prevents wasteful double-inference
  - Latency estimator adapts (Jacobson-Karels) instead of growing monotonically
"""
import os
import time
import threading
import math
from collections import deque

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, String, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
import message_filters
import collections

from maniflow.dataset.relative_pose import (
    HAND_SLICE, POS_SLICE, QUAT_SLICE, RELATIVE_DIM, assert_identity_pose,
    assert_valid_rotation, from_relative, matrix_to_quat, matrix_to_rot6d,
    quat_to_matrix, rot6d_to_matrix, to_relative,
)
from model_serve import init_model, normalize
from cond_preprocess import (PointCloudProcessor, setup_kinematics, compute_fk,
                             preprocess_wrist_rgb, WRIST_RGB_HW)
from rtc_utils import RTCProcessor
from progress_utils import ProgressTracker, flush as flush_dds, latched_qos



WRIST_RGB_TOPIC = os.environ.get(
    "WRIST_COLOR_TOPIC", "/cam_gemini/color/image_raw")


# ---------------------------------------------------------------------------
# Hand action layout: policy output -> full-hand wire format
# ---------------------------------------------------------------------------
# The policy is trained on a SUBSET of the hand's DoF (see
# proprioception.hand.pinch in data/configs/foil_lifting_wrist.yaml), but the
# executor should not have to know which subset. So the expansion to all 20
# joints happens HERE, where the model's configuration is already loaded, and
# the wire format stays fixed at 7 pose + 20 hand = 27 regardless of model.
ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Finger-major order, matching record_utils/record.py's HAND_JOINT_NAMES, the
# recorded state vector, and dg5f_right_controller.yaml's `joints:` list.
FULL_HAND_JOINTS = [f"rj_dg_{finger}_{joint}"
                    for finger in range(1, 6)
                    for joint in range(1, 5)]

# The hand joints this checkpoint actually predicts, in the order they appear
# in the action vector. For the pinch models that is thumb + index; for a
# full-hand model set this to FULL_HAND_JOINTS and the expansion below becomes
# an identity mapping.
POLICY_HAND_JOINTS = [f"rj_dg_{finger}_{joint}"
                      for finger in (1, 2)
                      for joint in range(1, 5)]

# Pose where the un-predicted joints are held, degrees. This is the URDF zero
# configuration (straight, unspread), which is inside every joint's limits.
#
# Verified against the recordings rather than assumed, because these fingers
# are not unobserved - they appear in the wrist camera and the front point
# cloud: over 252 episodes / 51719 frames the median of the 12 non-pinch
# joints deviates from zero by at most 12.7 deg (mean 4.7). The demonstrations
# kept them near straight, so holding them at zero is not a configuration the
# policy never saw.
HELD_JOINT_POSE_DEG = 0.0

POSE_DIM = 7
WIRE_ACTION_DIM = POSE_DIM + len(FULL_HAND_JOINTS)  # 27

# "off" | "rebase" - see the RTC_MODE section in the module docstring.
RTC_MODE = os.environ.get("RTC_MODE", "off")


def encode_obs_relative(state_abs, n_obs_steps):
    """(T, 15) absolute observations -> (T, 17) relative + the base pose.

    The base is the LAST row, i.e. the newest observation. snapshot_obs()
    returns exactly the last n_obs_steps frames, so this is the same choice
    RelativeRealDexDataset makes with state[n_obs_steps - 1] - and it uses
    nothing the policy could not measure at this instant.
    """
    state_abs = np.asarray(state_abs, dtype=np.float64)
    if state_abs.shape[-1] != 15:
        raise ValueError(
            f"expected 15-dim absolute state, got {state_abs.shape[-1]}")

    T = state_abs[:, POS_SLICE]
    R = quat_to_matrix(state_abs[:, QUAT_SLICE])
    hand = state_abs[:, HAND_SLICE]

    T_base = T[-1].copy()
    R_base = R[-1].copy()

    T_rel, R_rel = to_relative(T, R, T_base, R_base)
    assert_identity_pose(T_rel[-1], R_rel[-1], what="deploy base pose")

    out = np.concatenate(
        [T_rel, matrix_to_rot6d(R_rel), hand], axis=-1).astype(np.float32)
    return out, T_base, R_base


def decode_action_relative(action_rel, T_base, R_base):
    """(N, 17) relative predictions -> (N, 15) absolute xyz + quat + hand.

    Must run BEFORE expand_hand_action, which slices at POSE_DIM=7 and
    assumes a quaternion.
    """
    action_rel = np.asarray(action_rel, dtype=np.float64)
    if action_rel.shape[-1] != RELATIVE_DIM:
        raise ValueError(
            f"expected {RELATIVE_DIM}-dim relative action, got "
            f"{action_rel.shape[-1]}")

    R_rel = rot6d_to_matrix(action_rel[:, 3:9])
    assert_valid_rotation(R_rel, what="decoded action rotation")

    T_abs, R_abs = from_relative(
        action_rel[:, 0:3], R_rel, T_base, R_base)
    assert_valid_rotation(R_abs, what="absolute action rotation")

    return np.concatenate([
        T_abs,
        matrix_to_quat(R_abs),
        action_rel[:, 9:RELATIVE_DIM],   # hand joints, absolute throughout
    ], axis=-1).astype(np.float32)


def rebase_leftover(prev_abs, T_base, R_base):
    """(N, 15) absolute leftover -> (N, 17) relative to the NEW base.

    The inverse direction of decode_action_relative, used to hand the
    unexecuted tail of the previous chunk to RTC in the frame the model is
    denoising in now.
    """
    prev_abs = np.asarray(prev_abs, dtype=np.float64)
    T_rel, R_rel = to_relative(
        prev_abs[:, POS_SLICE],
        quat_to_matrix(prev_abs[:, QUAT_SLICE]),
        T_base, R_base)
    return np.concatenate([
        T_rel, matrix_to_rot6d(R_rel), prev_abs[:, HAND_SLICE],
    ], axis=-1).astype(np.float32)


def expand_hand_action(actions: np.ndarray) -> np.ndarray:
    """(N, 7 + len(POLICY_HAND_JOINTS)) -> (N, 27), by joint NAME.

    Scatters the predicted hand values into their slots in the full 20-joint
    vector and fills the rest with HELD_JOINT_POSE_DEG. Mapping is by name, so
    a reordering of either list cannot silently permute joints.
    """
    expected = POSE_DIM + len(POLICY_HAND_JOINTS)
    if actions.shape[1] != expected:
        raise ValueError(
            f"policy produced action_dim={actions.shape[1]}, but "
            f"POLICY_HAND_JOINTS implies {expected} "
            f"({POSE_DIM} pose + {len(POLICY_HAND_JOINTS)} hand). Update "
            f"POLICY_HAND_JOINTS in deploy.py to match the checkpoint.")

    out = np.empty((actions.shape[0], WIRE_ACTION_DIM), dtype=actions.dtype)
    out[:, :POSE_DIM] = actions[:, :POSE_DIM]
    out[:, POSE_DIM:] = HELD_JOINT_POSE_DEG

    for src, name in enumerate(POLICY_HAND_JOINTS):
        dst = POSE_DIM + FULL_HAND_JOINTS.index(name)
        out[:, dst] = actions[:, POSE_DIM + src]

    return out


# ---------------------------------------------------------------------------
# Jacobson-Karels Latency Estimator
# (ported from Interlatent inference/client/latency.py)
# ---------------------------------------------------------------------------

class JacobsonKarels:
    """Smoothed RTT estimator that decays after spikes.

    TCP's classic algorithm (Jacobson & Karels 1988), tuned for DRTC:
      srtt   = (1 - alpha) * srtt   + alpha * sample
      rttvar = (1 - beta)  * rttvar + beta  * |sample - srtt|
      estimate = srtt + K * rttvar

    K=1.5 (DRTC paper) instead of TCP's K=4 for tighter tracking.
    """

    def __init__(self, alpha=1/8, beta=1/4, k=1.5):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self._srtt = 0.0
        self._rttvar = 0.0
        self._initialized = False
        self.SEED_CAP_S = 1.0

    def observe(self, sample_s: float):
        if not self._initialized:
            sample_s = min(sample_s, self.SEED_CAP_S)
            self._srtt = sample_s
            self._rttvar = sample_s / 2
            self._initialized = True
            return
        err = sample_s - self._srtt
        self._srtt += self.alpha * err
        self._rttvar += self.beta * (abs(err) - self._rttvar)

    @property
    def estimate_s(self) -> float:
        if not self._initialized:
            return 0.0
        return self._srtt + self.k * self._rttvar

    def estimate_steps(self, control_period_s: float) -> int:
        if control_period_s <= 0:
            return 0
        return max(0, math.ceil(self.estimate_s / control_period_s))

    @property
    def mean_s(self) -> float:
        return self._srtt if self._initialized else 0.0


# ---------------------------------------------------------------------------
# Cooldown Counter (ported from Interlatent inference/client/cooldown.py)
# ---------------------------------------------------------------------------

class Cooldown:
    """Step-counting cooldown that prevents double-firing inference.

    Armed when inference is triggered; decrements each check cycle.
    Ready when counter reaches 0.
    """

    def __init__(self, epsilon: int = 2):
        self._n = 0
        self._epsilon = max(1, int(epsilon))
        self._lock = threading.Lock()

    def arm(self, latency_steps: int = 0):
        with self._lock:
            self._n = max(1, int(latency_steps) + self._epsilon)

    def tick(self):
        with self._lock:
            if self._n > 0:
                self._n -= 1

    def ready(self) -> bool:
        with self._lock:
            return self._n == 0

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._n


# ---------------------------------------------------------------------------
# Camera + Robot Synchronizer (unchanged logic)
# ---------------------------------------------------------------------------

class RobotCameraSync:
    """Synchronizes ZED + wrist cameras with joint state by timestamp.

    The wrist camera joins the same ApproximateTimeSynchronizer group as the
    ZED colour/depth pair rather than being sampled independently. The policy
    consumes point cloud and wrist image as ONE observation, so a wrist frame
    paired with a cloud from a different instant would present a hand in a
    place the cloud does not show it - and nothing downstream could detect it.
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
        self.sub_wrist = message_filters.Subscriber(
            node, Image, WRIST_RGB_TOPIC, qos_profile=qos_slow)

        # slop is raised from 15 ms to 30 ms because the wrist camera runs on
        # its own clock and is not hardware-synced to the ZED. At 20 Hz
        # (50 ms/frame) 30 ms still rejects a full-frame offset, while 15 ms
        # would drop most triples and starve the observation buffer.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_wrist],
            queue_size=15,
            slop=0.030
        )
        self.ts.registerCallback(self._camera_sync_cb)

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

    def _camera_sync_cb(self, color_msg, depth_msg, wrist_msg):
        self.camera_buffer.append((color_msg, depth_msg, wrist_msg))
        self._try_process_buffers()

    def _try_process_buffers(self):
        while self.camera_buffer and self.joint_history:
            color_msg, depth_msg, wrist_msg = self.camera_buffer[0]
            t_cam = self._get_sec(color_msg.header.stamp)
            t_newest_joint = self._get_sec(self.joint_history[-1].header.stamp)

            if t_newest_joint >= (t_cam - 0.002):
                color_msg, depth_msg, wrist_msg = self.camera_buffer.popleft()

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
                    self.callback(closest_joint_msg, color_msg, depth_msg,
                                  wrist_msg)
            else:
                break


class RealDexEnv(Node):
    """Environment node: captures observations, publishes action chunks.

    Observation capture is callback-driven (continuous). Inference is
    triggered externally by the main loop based on buffer demand.
    """

    def __init__(self, cfg, fps: float, use_progress: bool = False,
                 progress_threshold: float = None,
                 progress_patience: int = 3):
        super().__init__('flow_policy')

        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.n_obs_steps = cfg.n_obs_steps
        self.n_action_steps = cfg.n_action_steps
        self.fps = fps

        # 6 arm + 20 hand, in the exact order record_utils/record.py writes
        # state.npy: UR_JOINT_NAMES then HAND_JOINT_NAMES (finger-major). The
        # order matters because compute_fk() slices positionally.
        #
        # Both groups arrive on the single global /joint_states:
        # dg5f_right_driver.launch.py remaps the controller's
        # /dg5f_right/joint_states onto it so MoveIt sees the 20 rj_dg_*
        # joints, and ur_tesollo_cartesian.launch.py includes that file.
        self.JOINT_NAMES = ARM_JOINT_NAMES + FULL_HAND_JOINTS

        self.point_cloud_array = []
        self.robot_state_array = []
        self.wrist_rgb_array = []

        self.pc_processor = PointCloudProcessor()
        self.bridge = CvBridge()
        self.chain = setup_kinematics()

        self._check_wrist_shape_meta()

        self.syncer = RobotCameraSync(self, self.observation_callback, self.JOINT_NAMES)

        # --- Publishers ---
        self.action_pub = self.create_publisher(
            Float64MultiArray, '/predicted_action_chunk', 10)
        # Latched: the final status is written immediately before teardown,
        # and a volatile sample would be lost to any subscriber not matched
        # at that instant (log_rollout.py in particular).
        self.status_pub = self.create_publisher(
            String, '/policy_status', latched_qos())

        # Owns /policy_progress, /policy_progress_marker and /policy_done.
        self.progress = ProgressTracker(
            self,
            use_progress=use_progress,
            progress_threshold=progress_threshold,
            patience=progress_patience)
        self.debug_pub = self.create_publisher(PointCloud2, '/debug_pc', 10)
        self.debug_action_pub = self.create_publisher(PoseStamped, '/debug_action_pose', 10)
        self.traj_pub = self.create_publisher(PoseArray, '/predicted_trajectory', 10)

        # --- Buffer status feedback (from executor node) ---
        self._buffer_depth = 0
        self._buffer_cursor = 0
        self._buffer_next_step = 0
        self._buffer_lock = threading.Lock()
        self.buffer_status_sub = self.create_subscription(
            Float64MultiArray, '/action_buffer_status',
            self._buffer_status_cb, 10)

        # --- Observation gating ---
        self.static_pc_tensor = None
        self.static_state_tensor = None
        self.static_wrist_tensor = None
        self.last_obs_time = 0.0
        self.target_dt = (1.0 / self.fps) - 0.005

        # --- Observation readiness ---
        self.new_obs_event = threading.Event()
        self._obs_lock = threading.Lock()

        self.get_logger().info(
            f"RealDexEnv initialized. FPS={fps}, "
            f"n_obs={self.n_obs_steps}, n_action={self.n_action_steps}")

    def _check_wrist_shape_meta(self):
        """Fail at startup if the live wrist resolution differs from training.

        The encoder would not: DINOv3ObsEncoder.forward() silently
        F.interpolate()s any mismatched input to the configured size. That
        turns a preprocessing divergence into a quiet accuracy loss instead of
        an error, so the comparison is made here, where it is cheap and loud.
        """
        try:
            shape = tuple(self.cfg.task.shape_meta['obs']['wrist_rgb']['shape'])
        except (AttributeError, KeyError, TypeError):
            self.get_logger().warn(
                "shape_meta has no wrist_rgb entry; skipping resolution check. "
                "If this checkpoint has no wrist camera, remove the wrist "
                "stream from the observation path as well.")
            return

        # shape_meta is CHW (the encoder's layout); WRIST_RGB_HW is (H, W).
        expected_hw = (int(shape[1]), int(shape[2]))
        if expected_hw != tuple(WRIST_RGB_HW):
            raise ValueError(
                f"wrist_rgb resolution mismatch: shape_meta wants "
                f"{expected_hw[0]}x{expected_hw[1]}, but cond_preprocess "
                f"produces {WRIST_RGB_HW[0]}x{WRIST_RGB_HW[1]}. Align "
                f"WRIST_RGB_HW with the task config (and with "
                f"dataset_builder rgb.resolution) before deploying.")
        self.get_logger().info(
            f"wrist_rgb: {expected_hw[0]}x{expected_hw[1]} uint8 HWC RGB, "
            f"matches shape_meta.")

    # ------------------------------------------------------------------
    # Buffer status feedback
    # ------------------------------------------------------------------

    def _buffer_status_cb(self, msg: Float64MultiArray):
        with self._buffer_lock:
            if len(msg.data) >= 3:
                self._buffer_depth = int(msg.data[0])
                self._buffer_cursor = int(msg.data[1])
                self._buffer_next_step = int(msg.data[2])
            elif len(msg.data) >= 2:
                self._buffer_depth = int(msg.data[0])
                self._buffer_cursor = int(msg.data[1])
                self._buffer_next_step = int(msg.data[1])

    def get_buffer_depth(self) -> int:
        with self._buffer_lock:
            return self._buffer_depth

    def get_buffer_next_step(self) -> int:
        with self._buffer_lock:
            return self._buffer_next_step

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_action_chunk(self, full_action_sequence: np.ndarray):
        """Publish without anchor (legacy, anchors at receiver cursor)."""
        msg = Float64MultiArray()
        # Convention: first element = -1 means "no anchor, use local cursor"
        wire = expand_hand_action(full_action_sequence)
        msg.data = [-1.0] + wire.flatten().tolist()
        self.action_pub.publish(msg)

    def publish_action_chunk_with_anchor(self, full_action_sequence: np.ndarray,
                                         anchor_step: int):
        """Publish with absolute anchor step (True LWW).

        Convention: msg.data[0] = anchor_step (>=0 means absolute anchor).
        Remaining data = flattened (N, 13) actions.
        The consumer reads anchor FROM the message — no time dilation
        even if DDS delivery is delayed.
        """
        msg = Float64MultiArray()
        wire = expand_hand_action(full_action_sequence)
        msg.data = [float(anchor_step)] + wire.flatten().tolist()
        self.action_pub.publish(msg)

    def publish_status(self, status_text: str):
        msg = String()
        msg.data = status_text
        self.status_pub.publish(msg)

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

    # ------------------------------------------------------------------
    # Observation capture (callback-driven, continuous)
    # ------------------------------------------------------------------

    def observation_callback(self, joint_msg, color_msg, depth_msg, wrist_msg):
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if (current_time - self.last_obs_time) < self.target_dt:
                return
            self.last_obs_time = current_time

            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            # bgr8 so the channel order entering preprocess_wrist_rgb matches
            # the builder's cv2.imread, which is also BGR. That function does
            # the flip to RGB.
            wrist_img = self.bridge.imgmsg_to_cv2(wrist_msg, "bgr8")

            # Indexed BY NAME, never by position: JointState ordering is not
            # guaranteed stable across driver versions, and a permutation here
            # would be invisible - the FK would just return a wrong pose.
            # record.py indexes by name for the same reason.
            pos_dict = dict(zip(joint_msg.name, joint_msg.position))
            missing = [n for n in self.JOINT_NAMES if n not in pos_dict]
            if missing:
                self.get_logger().error(
                    f"/joint_states is missing {len(missing)} expected joints "
                    f"(e.g. {missing[:4]}). Is the dg5f driver up and remapped "
                    f"onto /joint_states? Skipping this observation.",
                    throttle_duration_sec=2.0)
                return

            ordered_state = [pos_dict[name] for name in self.JOINT_NAMES]
            robot_state = np.array(ordered_state, dtype=np.float32)

            # ros2_control publishes radians; the dataset stores hand joints in
            # DEGREES (record.py's HAND_JOINTS_IN_DEGREES=1, which converts on
            # the way in). Without this the hand block of agent_pos is off by
            # ~57x - a silent error, since the vector stays well-formed and
            # only the policy's view of the hand is wrong. Arm joints stay in
            # radians: they feed FK, and record.py does not convert them.
            robot_state[len(ARM_JOINT_NAMES):] = np.degrees(
                robot_state[len(ARM_JOINT_NAMES):])

            cartesian_state_phys = compute_fk(self.chain, robot_state)
            point_cloud = self.pc_processor.process_rgbd(color_img, depth_img)

            if point_cloud is not None:
                self.publish_debug_pointcloud(point_cloud)
                wrist_rgb = preprocess_wrist_rgb(wrist_img)

                # Appended under one lock so the three buffers cannot be
                # observed at different lengths: snapshot_obs() takes the last
                # n_obs_steps of each and would otherwise pair a cloud with a
                # wrist frame from a different tick.
                with self._obs_lock:
                    self.point_cloud_array.append(point_cloud)
                    self.robot_state_array.append(cartesian_state_phys)
                    self.wrist_rgb_array.append(wrist_rgb)

                    if len(self.point_cloud_array) > self.n_obs_steps:
                        self.point_cloud_array.pop(0)
                    if len(self.robot_state_array) > self.n_obs_steps:
                        self.robot_state_array.pop(0)
                    if len(self.wrist_rgb_array) > self.n_obs_steps:
                        self.wrist_rgb_array.pop(0)

                self.new_obs_event.set()
            else:
                self.get_logger().warn("Point cloud returned None",
                                       throttle_duration_sec=2.0)

        except Exception as e:
            self.get_logger().error(f"Observation processing error: {e}")

    # ------------------------------------------------------------------
    # Observation access (non-blocking snapshot vs. blocking wait)
    # ------------------------------------------------------------------

    def obs_ready(self) -> bool:
        with self._obs_lock:
            return len(self.robot_state_array) >= self.n_obs_steps

    def wait_for_obs(self, timeout: float = 1.0) -> bool:
        return self.new_obs_event.wait(timeout=timeout)

    def snapshot_obs(self):
        """Non-blocking: return latest observation as GPU tensors.

        Returns (obs_dict, T_base, R_base). The base pose is the newest
        observed EE pose and must be kept by the caller: the prediction is
        expressed in that frame and cannot be decoded without it.
        """
        with self._obs_lock:
            if len(self.robot_state_array) < self.n_obs_steps:
                return None, None, None

            pc_np = np.stack(
                self.point_cloud_array[-self.n_obs_steps:], axis=0)[np.newaxis, ...]
            state_np = np.stack(
                self.robot_state_array[-self.n_obs_steps:], axis=0)[np.newaxis, ...]
            # (1, T, H, W, 3) uint8, channels-LAST. That is the zarr layout,
            # and maniflow_multimodal_policy._encode_wrist() is the single
            # place the HWC->CHW permute happens - handing it CHW here would
            # double-permute. Kept uint8: DINOv3ObsEncoder does /255 plus its
            # own mean/std, and scaling here would be applied twice.
            wrist_np = np.stack(
                self.wrist_rgb_array[-self.n_obs_steps:], axis=0)[np.newaxis, ...]

        self.new_obs_event.clear()

        # (1, T, 15) absolute -> (1, T, 17) relative to the newest frame.
        state_rel, T_base, R_base = encode_obs_relative(
            state_np[0], self.n_obs_steps)
        state_np = state_rel[np.newaxis, ...]

        pc_tensor_cpu = torch.from_numpy(pc_np)
        state_tensor_cpu = torch.from_numpy(state_np)
        wrist_tensor_cpu = torch.from_numpy(wrist_np)

        if self.static_pc_tensor is None:
            self.static_pc_tensor = pc_tensor_cpu.to(self.device)
            self.static_state_tensor = state_tensor_cpu.to(self.device)
            self.static_wrist_tensor = wrist_tensor_cpu.to(self.device)
        else:
            self.static_pc_tensor.copy_(pc_tensor_cpu, non_blocking=True)
            self.static_state_tensor.copy_(state_tensor_cpu, non_blocking=True)
            self.static_wrist_tensor.copy_(wrist_tensor_cpu, non_blocking=True)

        return {
            'point_cloud': self.static_pc_tensor,
            'agent_pos': self.static_state_tensor,
            'wrist_rgb': self.static_wrist_tensor,
        }, T_base, R_base


# sport_mode is intentionally absent in this variant.


def run_policy(
    alg_name: str,
    task_name: str,
    run_dir: str,
    addition_info: int = 1211,
    fps: float = 20.0,
    use_progress: bool = False,
    progress_threshold: float = None,
    progress_patience: int = 3,
    warmup_steps: int = 3,
    num_inference_steps: int = None,
    use_rtc: bool = True,
    rtc_execution_horizon: int = 16,
    rtc_max_guidance: float = 5.0,
    # DRTC scheduling parameters
    control_hz: float = 20.0,
    ckpt_file: str = None,
    min_execution_horizon: int = 8,
    cooldown_epsilon: int = 2,
):
    if not rclpy.ok():
        rclpy.init()

    # --- Model loading ---
    cfg, policy = init_model(
        task_name=task_name,
        alg_name=alg_name,
        addition_info=addition_info,
        run_dir=run_dir, 
        ckpt_file=ckpt_file
    )

    if num_inference_steps is not None:
        old_steps = getattr(policy, 'num_inference_steps', None)
        policy.num_inference_steps = num_inference_steps
        print(f"[INFO] num_inference_steps: {old_steps} → {num_inference_steps}")
    else:
        print(f"[INFO] num_inference_steps: {getattr(policy, 'num_inference_steps', '?')}")

    # --- Environment node ---
    env = RealDexEnv(cfg=cfg, fps=fps, use_progress=use_progress,
                     progress_threshold=progress_threshold,
                     progress_patience=progress_patience)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(env)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # --- DRTC scheduling components ---
    latency_estimator = JacobsonKarels()
    cooldown = Cooldown(epsilon=cooldown_epsilon)
    control_period_s = 1.0 / control_hz

    # --- RTC (Real-Time Chunking) ---
    # Gated on RTC_MODE as well: guidance happens inside the denoising loop,
    # so the leftover must be re-based into the current observation frame.
    # "off" runs without guidance to verify encode/decode in isolation.
    rtc_active = use_rtc and RTC_MODE == "rebase"
    rtc = RTCProcessor(
        execution_horizon=rtc_execution_horizon,
        max_guidance_weight=rtc_max_guidance,
        prefix_attention_schedule="exp") if rtc_active else None
    prev_action_abs = None

    if rtc_active:
        print(f"[INFO] RTC ENABLED, mode=rebase "
              f"(exec_horizon={rtc_execution_horizon}, "
              f"max_guidance={rtc_max_guidance})")
    elif use_rtc:
        print(f"[INFO] RTC DISABLED (RTC_MODE={RTC_MODE!r}); "
              f"set RTC_MODE=rebase to enable")
    else:
        print("[INFO] RTC DISABLED (use_rtc=False)")

    print(f"[INFO] DRTC scheduling: control_hz={control_hz}, "
          f"min_execution_horizon={min_execution_horizon}, "
          f"cooldown_epsilon={cooldown_epsilon}")

    try:
        env.publish_status("RUNNING")
        print("[INFO] Waiting for initial observations...")

        # --- Wait for initial observation buffer to fill ---
        while not env.obs_ready():
            if not env.wait_for_obs(timeout=1.0):
                print("[INFO] Still waiting for observations...")
            if not rclpy.ok():
                return

        # snapshot_obs() returns (obs_dict, T_base, R_base) in this variant,
        # unlike deploy.py where it returns the dict directly. Unpack here -
        # handing the tuple to predict_action() sends a non-dict into
        # LinearNormalizer, which then takes the '_default' branch and raises
        # "Not initialized" (the normalizer is fitted per-key).
        snapshot = env.snapshot_obs()
        if snapshot is None:
            print("[ERROR] Failed to get initial observation!")
            return
        obs, _T_base_warmup, _R_base_warmup = snapshot

        # --- Warmup: stabilize CUDA kernels ---
        print(f"[INFO] Warming up model ({warmup_steps} steps)...")
        warmup_result = None
        for i in range(warmup_steps):
            with torch.no_grad():
                warmup_result = policy.predict_action(obs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{warmup_steps} done.")

        # Compare the declared progress config against what the policy
        # actually returns, before the robot moves. Raises if
        # use_progress=True on a checkpoint with no progress head.
        env.progress.validate(warmup_result)
        print("[INFO] Warmup complete. Starting DRTC inference loop.")

        # --- Telemetry ---
        inference_count = 0

        # ===================================================================
        # DRTC INFERENCE LOOP (demand-driven)
        #
        # Key difference from old design:
        #   OLD: infer → publish → BLOCK on next observation → repeat
        #   NEW: poll (buffer_low AND obs_fresh AND cooldown) → infer → publish
        #
        # The executor node pops actions at fixed rate independently.
        # This loop only runs inference when the buffer needs refill.
        #
        # Cooldown: In Interlatent, cooldown.tick() is called once per
        # CONTROL STEP (in step()). Here we're in a separate process, so
        # we track wall time to tick at the correct rate.
        # ===================================================================
        last_cooldown_tick_time = time.perf_counter()

        while rclpy.ok():
            # --- 1. WAIT FOR TRIGGER CONDITION ---
            while True:
                if not rclpy.ok():
                    break

                # Tick cooldown at control rate (not poll rate!)
                now = time.perf_counter()
                ticks_due = int((now - last_cooldown_tick_time) / control_period_s)
                if ticks_due > 0:
                    for _ in range(min(ticks_due, 10)):  # cap to avoid burst
                        cooldown.tick()
                    last_cooldown_tick_time = now

                buffer_depth = env.get_buffer_depth()
                delay_steps = latency_estimator.estimate_steps(control_period_s)
                horizon = delay_steps + min_execution_horizon

                needs_refill = buffer_depth < horizon
                cd_ready = cooldown.ready()
                obs_fresh = env.obs_ready()

                if needs_refill and cd_ready and obs_fresh:
                    break

                time.sleep(0.005)

            if not rclpy.ok():
                break

            # --- 2. SNAPSHOT OBSERVATION (non-blocking) ---
            # T_base/R_base define the frame this prediction lives in; the
            # chunk cannot be decoded without them.
            obs, T_base, R_base = env.snapshot_obs()
            if obs is None:
                time.sleep(0.01)
                continue

            # --- 3. RTC PREP ---
            prev_leftover = None
            leftover_abs = None
            rtc_delay = 0
            if rtc is not None and prev_action_abs is not None:
                rtc_delay = latency_estimator.estimate_steps(control_period_s)
                rtc_delay = max(0, min(rtc_delay, rtc_execution_horizon,
                                       len(prev_action_abs) - 1))
                # Re-base: the cached chunk is ABSOLUTE, so expressing its
                # tail in the current base puts it in the frame the model is
                # about to denoise in. Passing the previous chunk's relative
                # values straight through would make RTC match coordinates
                # offset by the motion during the inference delay.
                leftover_abs = prev_action_abs[rtc_delay:]
                prev_leftover = torch.from_numpy(
                    rebase_leftover(leftover_abs, T_base, R_base)[np.newaxis]
                ).to(policy.device)

            # --- 4. GPU INFERENCE ---
            t_infer_start = time.perf_counter()
            with torch.no_grad():
                action_queue_raw = policy.predict_action(
                    obs,
                    rtc_processor=rtc,
                    prev_chunk_left_over=prev_leftover,
                    inference_delay=rtc_delay)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            inference_time = time.perf_counter() - t_infer_start

            # --- 5. UPDATE LATENCY ESTIMATOR + ARM COOLDOWN ---
            latency_estimator.observe(inference_time)
            delay_steps = latency_estimator.estimate_steps(control_period_s)
            cooldown.arm(delay_steps)

            # --- 6. POST-PROCESSING ---
            # Decode BEFORE expand_hand_action, which slices at POSE_DIM=7
            # and assumes a quaternion. model_serve.normalize() is not used:
            # it renormalises columns 3:7 as a quaternion, which here are the
            # first four components of the 6D rotation.
            action_rel = action_queue_raw['action'][0].cpu().numpy()
            action_queue = decode_action_relative(action_rel, T_base, R_base)

            # Cache the ABSOLUTE chunk for the next cycle's RTC re-basing.
            prev_action_abs = action_queue.copy()

            # --- 6b. PROGRESS CHECK (BEFORE publishing) ---
            # Checked ahead of the publish: a chunk sent now would be
            # executed in full (~26 steps, over a second) before anything
            # could stop it, so "done" would always mean "plus one more
            # chunk of motion". Monitor-only mode returns False here and
            # the loop continues untouched.
            task_done = env.progress.update(action_queue_raw)
            bar = env.progress.render_bar()
            if bar:
                print(bar)
            if task_done:
                env.publish_status("DONE")
                break

            # --- 7. LATENCY SHIFT + PUBLISH WITH ABSOLUTE ANCHOR ---
            # The anchor step is determined NOW (at publish time), not at
            # receive time. This prevents time dilation if DDS delivery is
            # delayed: the consumer reads the anchor FROM the message and
            # merges at the correct absolute step regardless of when it
            # arrives. This is the "True LWW" approach from Interlatent:
            # the step index is set at SEND time, not RECEIVE time.
            #
            # anchor = cursor_at_obs_time + delay_steps
            # But since we read cursor from the 10Hz feedback, and delay_steps
            # accounts for time elapsed during inference, the anchor is simply
            # the current cursor (which has advanced during inference).
            delay_steps = latency_estimator.estimate_steps(control_period_s)
            anchor_step = env.get_buffer_next_step()  # executor's cursor NOW

            if delay_steps < len(action_queue):
                future_actions = action_queue[delay_steps:]
            else:

                future_actions = action_queue[-1:]


            # Encode anchor as first element: [anchor_step, a0_dim0, a0_dim1, ...]
            env.publish_action_chunk_with_anchor(future_actions, anchor_step)

            # --- 8. RTC CONTINUITY METRIC ---
            # Both operands absolute: action_queue is decoded, leftover_abs
            # is the cached absolute chunk.
            if leftover_abs is not None:
                k = min(rtc_execution_horizon, len(action_queue),
                        len(leftover_abs))
                if k > 0:
                    new_xyz = action_queue[:k, :3]
                    old_xyz = leftover_abs[:k, :3]
                    per_step = np.linalg.norm(new_xyz - old_xyz, axis=1)
                    print(f"[RTC] step0={per_step[0]*1000:.1f}mm "
                          f"mean={per_step.mean()*1000:.1f}mm "
                          f"max={per_step.max()*1000:.1f}mm")

            # --- 9. TELEMETRY ---
            inference_count += 1
            if inference_count % 10 == 0:
                rtc_tag = f"RTC(d={rtc_delay}) " if rtc is not None else ""
                print(
                    f"[DRTC] {rtc_tag}#{inference_count} | "
                    f"infer={inference_time*1000:.0f}ms "
                    f"(est={latency_estimator.estimate_s*1000:.0f}ms) | "
                    f"chunk={len(action_queue)} | "
                    f"buffer={env.get_buffer_depth()} | "
                    f"horizon={horizon} | "
                    f"cooldown={cooldown.remaining}")

    except KeyboardInterrupt:
        pass
    finally:
        # Settle before teardown: destroying the node drops the
        # transient-local history with it, and /policy_done is what the
        # controllers act on to stop the arm.
        flush_dds()
        executor.shutdown()
        env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    run_policy(
        alg_name="maniflow_multimodal_policy",
        task_name="foil_lifting_relative",
        run_dir="/ros2_ws/data/outputs/foil_lifting_relative-maniflow_multimodal_policy-0112_seed0",
        ckpt_file="epoch=0040-val_action_mse=0.022836",
        addition_info=1211,
        fps=20.0,
        # False: maniflow_multimodal_policy has no progress head, so
        # requesting one would (correctly) raise at startup now. Flip to
        # True once a checkpoint emits progress_score - leave
        # progress_threshold=None at first to monitor it before letting it
        # terminate runs. 
        use_progress=False,
        progress_threshold=0.9,
        progress_patience=3,
        warmup_steps=3,
        num_inference_steps=10,
        use_rtc=True,
        rtc_execution_horizon=18,
        rtc_max_guidance=5.0,
        # DRTC scheduling
        control_hz=20.0,
        min_execution_horizon=20,
        cooldown_epsilon=2,

    )

