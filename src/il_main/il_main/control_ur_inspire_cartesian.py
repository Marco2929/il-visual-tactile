#!/usr/bin/env python3
"""
DRTC-style Action Executor for Cartesian impedance control.

Adapted from the Interlatent DRTC architecture:
  - Receives overlapping action chunks from the policy inference node
  - Buffers them in an LWW (Last-Write-Wins) action schedule
  - Pops exactly ONE action per control tick at a fixed rate (30 Hz)
  - Applies 2nd-order Butterworth low-pass smoothing (chunk-boundary jitter)
  - Validates workspace bounds + quaternion sanity
  - Publishes smoothed PoseStamped to the C++ impedance controller

Architecture:
  [Policy Node] --/predicted_action_chunk--> [ActionSchedule]
                                                    |
                                              pop_next() @ 30 Hz
                                                    |
                                              [ButterworthLowPass]
                                                    |
                                              [workspace validate]
                                                    |
                                              /target_frame (PoseStamped)
                                                    |
                                         /action_buffer_status --> [Policy Node]
                                              (demand-driven trigger)

Key properties vs. the old design:
  - Every action in a chunk is executed exactly once (no discarding)
  - Fixed-rate control regardless of inference speed
  - Smooth chunk boundaries via low-pass filter
  - Buffer depth feedback enables demand-driven inference
"""
import os
import sys
import math
import time
import threading
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, Bool, Int32
from service_interfaces.msg import SetAngle1
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, "/ros2_ws/data/record_utils")
from camera_info import workspace

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROBOT_BASE_LINK = os.environ.get("ROBOT_BASE_LINK", "base_link")
TARGET_TOPIC = "/target_frame"
CONTROL_HZ = 20.0          # Fixed execution rate (Hz) — must match policy fps
BUFFER_MAXLEN = 200         # Max buffered actions (~10s at 20Hz)
WATCHDOG_TIMEOUT_S = 3.0    # Warn if no chunk received for this long
SMOOTHER_CUTOFF_HZ = 8.0    # Butterworth low-pass cutoff (Hz) — lower = smoother but laggier
MAX_POSITION_STEP = 0.1   # Delta clamp: max position jump per tick (meters)
MAX_ROTATION_STEP = 0.15    # Delta clamp: max quaternion distance per tick

# --- Hand (finger) anti-jitter ---
HAND_SMOOTHER_CUTOFF_HZ = 3.0  # Lower = smoother fingers (was 8.0 -> jittery)
HAND_DEADBAND = 0.015          # Ignore finger changes smaller than this (norm units 0-1)
HAND_MAX_STEP = 0.08           # Max finger change per tick (norm units 0-1)

# Frame correction: policy predicts for right_inspire_base_link,
# controller targets tool0. Offset = 18.6mm along tool-local Z.
INSPIRE_TO_TOOL0_Z = -0.0186  # meters (tool0 is 18.6mm behind inspire_base)


# ---------------------------------------------------------------------------
# Butterworth Low-Pass Filter (ported from Interlatent node/smoothing.py)
# ---------------------------------------------------------------------------

def _butter2_lowpass_coeffs(cutoff_hz: float, sample_hz: float):
    """2nd-order Butterworth biquad coefficients via bilinear transform."""
    k = math.tan(math.pi * cutoff_hz / sample_hz)
    k2 = k * k
    sqrt2 = math.sqrt(2.0)
    a0 = 1.0 + sqrt2 * k + k2
    b0 = k2 / a0
    b1 = 2.0 * k2 / a0
    b2 = k2 / a0
    a1 = (2.0 * (k2 - 1.0)) / a0
    a2 = (1.0 - sqrt2 * k + k2) / a0
    return (b0, b1, b2), (a1, a2)


class ButterworthLowPass:
    """Vectorized 2nd-order Butterworth low-pass over joint/pose vectors.

    Each element is filtered independently with the same coefficients but
    its own delay-line state. Warm-started: first sample seeds the delay
    line so output begins at the current pose (no startup ramp toward zero).
    """

    def __init__(self, cutoff_hz: float, sample_hz: float):
        if cutoff_hz <= 0.0 or sample_hz <= 0.0:
            raise ValueError(f"cutoff/sample Hz must be positive: {cutoff_hz}, {sample_hz}")
        nyq = 0.5 * sample_hz
        if cutoff_hz >= nyq:
            cutoff_hz = 0.99 * nyq
        self.cutoff_hz = cutoff_hz
        self.sample_hz = sample_hz
        (self._b0, self._b1, self._b2), (self._a1, self._a2) = (
            _butter2_lowpass_coeffs(cutoff_hz, sample_hz)
        )
        self._x1: Optional[np.ndarray] = None
        self._x2: Optional[np.ndarray] = None
        self._y1: Optional[np.ndarray] = None
        self._y2: Optional[np.ndarray] = None

    def reset(self):
        """Drop delay line — next sample warm-starts fresh."""
        self._x1 = self._x2 = self._y1 = self._y2 = None

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filter one vector. Returns smoothed output (same shape)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._x1 is None or self._x1.shape != x.shape:
            # Warm start: seed entire delay line with current sample
            self._x1 = x.copy()
            self._x2 = x.copy()
            self._y1 = x.copy()
            self._y2 = x.copy()
            return x.copy()

        y = (
            self._b0 * x
            + self._b1 * self._x1
            + self._b2 * self._x2
            - self._a1 * self._y1
            - self._a2 * self._y2
        )

        self._x2 = self._x1
        self._x1 = x.copy()
        self._y2 = self._y1
        self._y1 = y.copy()
        return y


# ---------------------------------------------------------------------------
# LWW Action Schedule (adapted from Interlatent inference/client/merge.py)
# ---------------------------------------------------------------------------

class ActionSchedule:

    """Thread-safe LWW action schedule (dict keyed by step index).




    Faithful port of Interlatent's merge.py CRDT:
      - Each action is keyed by (action_step) and carries a control_timestamp.
      - When two chunks arrive for the same step, newer control_timestamp wins.
      - pop_next() returns the action at the cursor and advances by one.
      - Actions for already-executed steps (action_step < cursor) are dropped.



    Key contract: "every step of every chunk is executed exactly once."
    """

    def __init__(self, maxlen: int = BUFFER_MAXLEN):

        self._data: dict = {}  # {action_step: (control_timestamp, np.ndarray)}
        self._cursor: int = 0

        self._lock = threading.Lock()
        self._maxlen = maxlen



    def merge_chunk(self, actions: np.ndarray, timestamp: float,
                    anchor_step: Optional[int] = None) -> int:
        """Merge a chunk into the schedule using LWW per-step semantics.





        Args:
            actions: (N, action_dim) array of actions

            timestamp: monotonic time (control_timestamp for LWW)
            anchor_step: The step index that actions[0] maps to.
                         If None, anchors at current cursor (REPLACE mode).

        Returns:

            Number of actions actually installed (newer than existing).
        """
        with self._lock:







            if anchor_step is None:
                anchor_step = self._cursor

            installed = 0
            for i, action in enumerate(actions):
                step = anchor_step + i

                if step < self._cursor:
                    continue  # Already executed — ignore

                existing = self._data.get(step)
                if existing is None or timestamp > existing[0]:
                    self._data[step] = (timestamp, action.copy())
                    installed += 1

            # Trim entries far ahead of cursor to bound memory
            if len(self._data) > self._maxlen:
                steps_sorted = sorted(self._data.keys())
                for s in steps_sorted[:len(self._data) - self._maxlen]:
                    if s < self._cursor:
                        del self._data[s]

            return installed

    def pop_next(self) -> Optional[np.ndarray]:

        """Pop and return the action at the cursor, advancing by one.




        Returns None if the cursor step is not scheduled (cold start or
        queue drained). The cursor is NOT advanced — the rollout pauses.
        """
        with self._lock:




            entry = self._data.pop(self._cursor, None)
            if entry is not None:
                self._cursor += 1
                return entry[1]  # Return the action array
            return None

    def queue_depth(self) -> int:

        """Number of scheduled actions at or ahead of cursor."""
        with self._lock:

            return sum(1 for s in self._data if s >= self._cursor)

    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def flush(self) -> int:
        """Drop all queued actions without advancing cursor."""
        with self._lock:


            n = len(self._data)
            self._data.clear()
            return n


    def next_action_step(self) -> int:
        """The step the producer should anchor its next chunk at.

        Returns the current cursor (REPLACE mode): the new chunk
        overwrites from HERE forward with fresh predictions.
        """
        with self._lock:

            return self._cursor


# ---------------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------------

class CartesianInferenceControlNode(Node):
    def __init__(self):
        super().__init__('cartesian_inference_control_node')

        self.robot_base_frame = ROBOT_BASE_LINK
        self.policy_active = False

        # --- TF listener (to seed _last_pose from live robot on activation) ---
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # --- Action schedule (the core DRTC buffer) ---
        self.schedule = ActionSchedule(maxlen=BUFFER_MAXLEN)

        # --- Smoothing (Butterworth low-pass, attenuates chunk-boundary jitter) ---
        self.pose_smoother = ButterworthLowPass(
            cutoff_hz=SMOOTHER_CUTOFF_HZ, sample_hz=CONTROL_HZ)
        self.hand_smoother = ButterworthLowPass(
            cutoff_hz=HAND_SMOOTHER_CUTOFF_HZ, sample_hz=CONTROL_HZ)

        # --- Last executed pose (for delta clamp) ---
        self._last_pose: Optional[np.ndarray] = None
        self._last_hand: Optional[np.ndarray] = None

        # --- Publishers ---
        self.target_pub = self.create_publisher(PoseStamped, TARGET_TOPIC, 10)
        self.hand_pub = self.create_publisher(SetAngle1, '/set_angle_data', 10)
        self.buffer_status_pub = self.create_publisher(
            Float64MultiArray, '/action_buffer_status', 10)

        # --- Subscribers ---
        self.chunk_sub = self.create_subscription(
            Float64MultiArray, '/predicted_action_chunk', self._chunk_callback, 10)
        self.policy_gate_sub = self.create_subscription(
            Bool, '/policy_gate', self._policy_gate_cb, 10)
        self.policy_done_sub = self.create_subscription(
            Bool, '/policy_done', self._policy_done_cb, 10)

        # --- Fixed-rate execution timer (THE critical change) ---
        # This runs at CONTROL_HZ regardless of inference speed.
        # Every tick: pop one action → smooth → validate → publish.
        self._exec_timer = self.create_timer(1.0 / CONTROL_HZ, self._execute_step)

        # --- Buffer status timer (feedback to producer, lower rate) ---
        self._status_timer = self.create_timer(1.0 / 10.0, self._publish_buffer_status)

        # --- Watchdog ---
        self._last_chunk_wall = 0.0

        # --- Telemetry ---
        self._steps_executed = 0
        self._steps_starved = 0
        self._telemetry_timer = self.create_timer(5.0, self._log_telemetry)

        self.get_logger().info(
            f"DRTC Cartesian Executor initialized. "
            f"Control rate: {CONTROL_HZ} Hz | "
            f"Smoother cutoff: {SMOOTHER_CUTOFF_HZ} Hz | "
            f"Buffer max: {BUFFER_MAXLEN} | "
            f"Publishing to: {TARGET_TOPIC}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _policy_gate_cb(self, msg: Bool):
        if msg.data and not self.policy_active:
            self.policy_active = True
            self.pose_smoother.reset()
            self.hand_smoother.reset()

            self._last_hand = None

            # Seed _last_pose from current robot TF so delta clamp
            # doesn't jump from a stale position after manual move.
            self._last_pose = self._read_current_pose()
            if self._last_pose is not None:
                self.get_logger().info(
                    f"Policy ACTIVE — seeded from TF: "
                    f"{np.round(self._last_pose[:3], 4).tolist()}")
            else:
                self.get_logger().info(
                    "Policy ACTIVE — TF unavailable, first action unclamped.")
        elif not msg.data and self.policy_active:
            self.policy_active = False
            self.schedule.flush()
            self.pose_smoother.reset()
            self.hand_smoother.reset()
            self._last_pose = None
            self._last_hand = None
            self.get_logger().info("Policy INACTIVE — holding last pose.")

    def _policy_done_cb(self, msg: Bool):
        if msg.data and self.policy_active:
            self.policy_active = False
            self.schedule.flush()
            self.pose_smoother.reset()
            self.hand_smoother.reset()
            self._last_pose = None
            self._last_hand = None
            self.get_logger().info("Policy DONE — task complete.")

    def _chunk_callback(self, msg: Float64MultiArray):
        """Receive action chunk and merge into the LWW schedule.

        Wire format: msg.data[0] = anchor_step (absolute, from producer).
          If anchor >= 0: use it (True LWW — step determined at SEND time).
          If anchor < 0:  fallback to local cursor (legacy mode).
        Remaining: flattened (N, 13) actions.

        This prevents time dilation: even if DDS delivery is delayed by
        50ms (1 control step), the actions land at the correct absolute
        step index because the anchor was set at publish time.
        """
        if not self.policy_active:
            return

        raw_data = np.array(msg.data)
        if len(raw_data) < 2:
            return

        # First element: anchor step (or -1 for legacy)
        anchor_field = raw_data[0]
        action_data = raw_data[1:]  # remaining = flattened actions

        action_dim = 13  # 7 (pose) + 6 (hand)

        if len(action_data) % action_dim != 0:
            self.get_logger().warn(
                f"Malformed chunk size: {len(action_data)} (not divisible by {action_dim}). "
                f"Ignoring.")
            return

        chunk_size = len(action_data) // action_dim
        actions = action_data.reshape(chunk_size, action_dim)

        # Determine anchor: absolute from message (True LWW) or local fallback
        if anchor_field >= 0:
            anchor = int(anchor_field)
        else:
            anchor = self.schedule.next_action_step()

        # LWW merge with monotonic timestamp
        timestamp = time.monotonic()
        installed = self.schedule.merge_chunk(actions, timestamp, anchor_step=anchor)

        if installed > 0:
            self._last_chunk_wall = timestamp
            self.get_logger().debug(
                f"Merged chunk: {installed}/{chunk_size} actions @ step {anchor} | "
                f"Buffer depth: {self.schedule.queue_depth()}")

    # ------------------------------------------------------------------
    # Fixed-rate executor (runs at CONTROL_HZ — the core DRTC mechanism)
    # ------------------------------------------------------------------

    def _execute_step(self):
        """Pop exactly one action per tick, smooth it, validate, publish.

        This is the DRTC contract: every action in the schedule is executed
        exactly once, at a fixed rate. If the buffer is empty (starvation),
        the robot holds its last commanded pose — no actions are skipped.

        When policy is INACTIVE, continuously publish the current robot pose
        to /target_frame so the C++ cartesian controller's internal target
        stays at the robot's actual position. This prevents the "snap back"
        when the robot is moved manually and the controller is re-enabled.
        """
        if not self.policy_active:
            # Hold mode: keep the C++ controller's target at the live pose
            current = self._read_current_pose()
            if current is not None:
                self._publish_target_pose(current)
            return

        action = self.schedule.pop_next()

        if action is None:
            # Starvation: buffer empty, hold last pose (cursor not advanced)
            self._steps_starved += 1
            self._check_watchdog()
            return

        self._steps_executed += 1

        # Split into pose (7D: xyz + quat) and hand (6D)

        pose_7d = action[:7].copy()
        hand_6d = action[7:13]

        # --- Frame correction: inspire_base → tool0 ---
        # Policy predicted where inspire_base should be (world frame).
        # Controller moves tool0. Shift back 18.6mm along tool's local Z.
        pose_7d = self._inspire_to_tool0(pose_7d)

        # --- Butterworth smoothing (attenuates chunk-boundary jitter) ---
        smoothed_pose = self.pose_smoother.filter(pose_7d)
        smoothed_hand = self.hand_smoother.filter(hand_6d)
        smoothed_hand = self._stabilize_hand(smoothed_hand)

        # --- Delta clamp (execution safety — last line of defense) ---
        clamped_pose = self._clamp_pose_delta(smoothed_pose)
        if clamped_pose is None:
            return  # Out of workspace entirely

        # --- Workspace validation ---
        if not self._validate_cartesian(clamped_pose):
            self.get_logger().warn(
                f"Target out of bounds: {clamped_pose[:3].round(3)}. Holding.",
                throttle_duration_sec=1.0)
            return

        # --- Publish ---
        self._publish_target_pose(clamped_pose)
        self._publish_hand(smoothed_hand)

        # Update state for next tick's delta clamp
        self._last_pose = clamped_pose.copy()
        self._last_hand = smoothed_hand.copy()

    @staticmethod
    def _inspire_to_tool0(pose_7d: np.ndarray) -> np.ndarray:
        """Convert inspire_base_link target → tool0 target.

        The policy predicts where inspire_base should be (world frame).
        The controller moves tool0. tool0 is 18.6mm behind inspire_base
        along the tool's local Z axis. We shift the position backward
        along the tool's Z (using the quaternion to rotate the offset
        into world frame). Orientation is unchanged.

        Quaternion rotation of [0, 0, z] without scipy:
          R @ [0, 0, z] = [2z(qx*qz + qw*qy),
                           2z(qy*qz - qw*qx),
                           z*(1 - 2*qx² - 2*qy²)]
        """
        qx, qy, qz, qw = pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6]
        z = INSPIRE_TO_TOOL0_Z
        pose_7d[0] += 2.0 * z * (qx * qz + qw * qy)
        pose_7d[1] += 2.0 * z * (qy * qz - qw * qx)
        pose_7d[2] += z * (1.0 - 2.0 * qx * qx - 2.0 * qy * qy)
        return pose_7d

    def _stabilize_hand(self, hand_6d: np.ndarray) -> np.ndarray:
        """Deadband + per-tick step clamp to kill finger jitter.

        The policy's hand output has small high-freq noise that the
        Butterworth alone doesn't fully remove. Two guards:
          1. Deadband: ignore sub-threshold changes (hold last value) so
             the fingers stay still instead of micro-twitching.
          2. Step clamp: cap how far a finger can move in one tick.
        """
        hand = np.asarray(hand_6d, dtype=np.float64).ravel()
        if self._last_hand is None:
            return hand

        out = hand.copy()
        for i in range(len(out)):
            delta = out[i] - self._last_hand[i]
            # Deadband: sub-threshold change -> hold previous value
            if abs(delta) < HAND_DEADBAND:
                out[i] = self._last_hand[i]
                continue
            # Step clamp: cap per-tick movement
            if abs(delta) > HAND_MAX_STEP:
                out[i] = self._last_hand[i] + np.sign(delta) * HAND_MAX_STEP
        return out

    def _clamp_pose_delta(self, pose_7d: np.ndarray) -> Optional[np.ndarray]:
        """Cap per-tick position/rotation jump (execution safety guard).

        Prevents a single-tick slam if the policy outputs a wild action or
        a chunk boundary is discontinuous (despite smoothing). The arm
        advances toward the target by at most MAX_POSITION_STEP per tick.
        """
        if self._last_pose is None:
            # First action — no previous pose to clamp against
            if not self._validate_cartesian(pose_7d):
                return None
            return pose_7d

        # Position delta clamp
        pos_delta = pose_7d[:3] - self._last_pose[:3]
        pos_dist = np.linalg.norm(pos_delta)
        if pos_dist > MAX_POSITION_STEP:
            # Clamp: move toward target but cap the step
            clamped_pos = self._last_pose[:3] + (pos_delta / pos_dist) * MAX_POSITION_STEP
        else:
            clamped_pos = pose_7d[:3]

        # Quaternion: ensure shortest path, then SLERP-limit the step
        quat = pose_7d[3:7].copy()
        last_quat = self._last_pose[3:7]

        # Ensure same hemisphere (shortest path)
        if np.dot(quat, last_quat) < 0:
            quat = -quat

        # Normalize
        qn = np.linalg.norm(quat)
        if qn > 0:
            quat = quat / qn

        # Simple rotation distance check (angular distance approximation)
        quat_delta = np.linalg.norm(quat - last_quat)
        if quat_delta > MAX_ROTATION_STEP:
            # Linear interpolation toward target, capped
            t = MAX_ROTATION_STEP / quat_delta
            quat = last_quat + t * (quat - last_quat)
            qn = np.linalg.norm(quat)
            if qn > 0:
                quat = quat / qn

        result = np.concatenate([clamped_pos, quat])
        return result

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_target_pose(self, pose_7d: np.ndarray):
        """Publish a single PoseStamped to the impedance controller."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_base_frame

        msg.pose.position.x = float(pose_7d[0])
        msg.pose.position.y = float(pose_7d[1])
        msg.pose.position.z = float(pose_7d[2])

        # Quaternion (already normalized by delta clamp)
        msg.pose.orientation.x = float(pose_7d[3])
        msg.pose.orientation.y = float(pose_7d[4])
        msg.pose.orientation.z = float(pose_7d[5])
        msg.pose.orientation.w = float(pose_7d[6])

        self.target_pub.publish(msg)

    def _publish_hand(self, hand_6d: np.ndarray):
        """Publish hand joint targets."""
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        reversed_joints = hand_6d[::-1]
        msg.angles = [int((1.0 - float(x)) * 1000) for x in reversed_joints]
        self.hand_pub.publish(msg)

    def _publish_buffer_status(self):
        """Publish buffer depth + cursor so the inference node can schedule.

        This is the demand signal: when depth drops below the execution
        horizon, the producer triggers a new inference request.
        """
        msg = Float64MultiArray()
        msg.data = [
            float(self.schedule.queue_depth()),
            float(self.schedule.cursor()),
            float(self.schedule.next_action_step()),
        ]
        self.buffer_status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Validation & Safety
    # ------------------------------------------------------------------

    def _validate_cartesian(self, pose_7d: np.ndarray) -> bool:
        """Check workspace bounds and quaternion sanity."""
        x, y, z = pose_7d[0], pose_7d[1], pose_7d[2]
        if not (workspace['x'][0] <= x <= workspace['x'][1] and
                workspace['y'][0] <= y <= workspace['y'][1] and
                workspace['z'][0] <= z <= workspace['z'][1]):
            return False

        quat_norm = np.linalg.norm(pose_7d[3:7])
        return 0.90 <= quat_norm <= 1.10

    def _read_current_pose(self) -> Optional[np.ndarray]:
        """Read current tool0 pose from TF tree.

        Returns 7D array [x, y, z, qx, qy, qz, qw] or None if unavailable.
        Used to seed _last_pose on policy activation so the delta clamp
        starts from the robot's ACTUAL position (not a stale old target).
        """
        try:
            t = self._tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            return np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ])
        except Exception:
            return None

    def _check_watchdog(self):
        """Warn if no chunk has arrived recently."""
        if self._last_chunk_wall == 0.0:
            return
        elapsed = time.monotonic() - self._last_chunk_wall
        if elapsed > WATCHDOG_TIMEOUT_S:
            self.get_logger().warn(
                f"No action chunk for {elapsed:.1f}s — inference may be stalled.",
                throttle_duration_sec=WATCHDOG_TIMEOUT_S)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _log_telemetry(self):
        """Periodic health summary (like Interlatent's DRTC stats line)."""
        if not self.policy_active:
            return
        total = self._steps_executed + self._steps_starved
        if total == 0:
            return
        starvation_pct = 100.0 * self._steps_starved / total
        depth = self.schedule.queue_depth()
        self.get_logger().info(
            f"[EXEC] steps={self._steps_executed} starved={self._steps_starved} "
            f"({starvation_pct:.1f}%) | buffer={depth} | "
            f"cursor={self.schedule.cursor()}")
        # Reset window counters
        self._steps_executed = 0
        self._steps_starved = 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = CartesianInferenceControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()