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
import math
import os
import sys
import time
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ---------------------------------------------------------------------------
# Workspace bounds
# ---------------------------------------------------------------------------
# Measured from the training set rather than taken from camera_info.workspace,
# which is stale for this task: it allows no negative z, while the
# demonstrations reach z = -0.080 (the foil sits below the frame origin), so
# every grasp would be rejected as out of bounds.
#
# Observed over all 51719 frames of foil_lifting_wrist_tesollo.zarr:
#     x  -0.523 .. +0.123
#     y  -1.230 .. -0.535
#     z  -0.080 .. +0.587
#
# Each axis is padded by a deliberately generous 0.25 m. This is a last-resort
# guard against a diverged prediction, NOT a cage: clamping near the
# demonstrated range would fight the policy at the edges of its own training
# distribution. The per-tick delta clamp below is what limits normal motion.
WORKSPACE_MARGIN_M = 0.25
WORKSPACE = {
    "x": (-0.523 - WORKSPACE_MARGIN_M, 0.123 + WORKSPACE_MARGIN_M),
    "y": (-1.230 - WORKSPACE_MARGIN_M, -0.535 + WORKSPACE_MARGIN_M),
    "z": (-0.080 - WORKSPACE_MARGIN_M, 0.587 + WORKSPACE_MARGIN_M),
}

# Reuse the DRTC machinery from the Inspire executor: the LWW schedule and the
# Butterworth filter are interface-agnostic, and forking them would mean two
# copies of the "execute every step exactly once" contract.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from control_ur_inspire_cartesian import ActionSchedule, ButterworthLowPass

# ---------------------------------------------------------------------------
# Action layout
# ---------------------------------------------------------------------------
# The wire format is fixed: 7 pose + one value per hand joint, in the order of
# HAND_JOINT_NAMES below. It does NOT depend on how many DoF the policy was
# trained on - deploy.py scatters a reduced prediction (e.g. the 8 pinch DoF)
# into the full vector before publishing and fills the rest with their hold
# pose. Swapping in a full-hand checkpoint is therefore a change in deploy.py
# alone; this node needs no mode flag and no knowledge of which subset was
# trained.
POSE_DIM = 7                      # xyz + quat xyzw, tool0 in base_link

# Finger-major hand joint order, identical to record_utils/record.py's
# HAND_JOINT_NAMES and to dg5f_right_controller.yaml's `joints:` list, so the
# n-th hand value in a chunk belongs to the n-th name here.
HAND_JOINT_NAMES = [f"rj_dg_{finger}_{joint}"
                    for finger in range(1, 6)
                    for joint in range(1, 5)]

HAND_DIM = len(HAND_JOINT_NAMES)   # 20
ACTION_DIM = POSE_DIM + HAND_DIM   # 27

# ---------------------------------------------------------------------------
# Topics and rates
# ---------------------------------------------------------------------------
ROBOT_BASE_LINK = os.environ.get("ROBOT_BASE_LINK", "base_link")
ROBOT_TIP_LINK = os.environ.get("ROBOT_TIP_LINK", "tool0")
TARGET_TOPIC = "/target_frame"
# The DG-5F stack lives in its own namespace: dg5f_right_driver.launch.py puts
# the controller_manager at /dg5f_right/controller_manager, so the trajectory
# topic is /dg5f_right/dg5f_right_controller/joint_trajectory.
#
# Publishing to /dg5f_right_controller/joint_trajectory (no namespace) does NOT
# fail: it creates a second, brand-new topic with no subscriber. The commands
# then look perfect in the log, the buffer never starves, and the hand never
# moves. Keep HAND_NS in sync with record_utils/record.py and deploy.py.
HAND_NS = os.environ.get("HAND_NS", "dg5f_right")
HAND_CONTROLLER = os.environ.get("HAND_CONTROLLER", "dg5f_right_controller")
HAND_TRAJECTORY_TOPIC = f"/{HAND_NS}/{HAND_CONTROLLER}/joint_trajectory"

# Must match the policy's fps and the dataset's 20 Hz resampling: the actions
# are next_state at 20 Hz, so executing them at any other rate rescales every
# commanded velocity.
CONTROL_HZ = 20.0
BUFFER_MAXLEN = 200
WATCHDOG_TIMEOUT_S = 3.0

SMOOTHER_CUTOFF_HZ = 8.0
MAX_POSITION_STEP = 0.1      # metres per tick
MAX_ROTATION_STEP = 0.15     # quaternion chord distance per tick

# --- Hand anti-jitter, in DEGREES ---
# The Inspire executor's 0.015 / 0.08 were normalised 0-1 gripper units. Reused
# verbatim on a degree-valued signal they would be a 0.015 deg deadband, i.e.
# no deadband at all, and a 0.08 deg/tick clamp, i.e. a hand frozen at
# 1.6 deg/s. These are the degree-domain equivalents.
HAND_SMOOTHER_CUTOFF_HZ = 3.0
HAND_DEADBAND_DEG = 0.3      # recorded hand states are quantised to ~0.1 deg
HAND_MAX_STEP_DEG = 6.0      # 120 deg/s at 20 Hz; joint limit is 180 deg/s

# Time given to the controller to reach each commanded point. One control
# period is the natural choice: the next action arrives exactly then.
HAND_POINT_DURATION_S = 1.0 / CONTROL_HZ
# The initial move is a real motion from wherever the hand sits to the start
# pose, so it gets its own, slower budget.
HAND_INIT_DURATION_S = 2.5

# --- Arm approach ramp ---
# The arm has the same problem the hand solves with HAND_INIT_DURATION_S, but
# had no equivalent: measured over rollouts 029/033/034, the first commanded
# pose sits 523-586 mm from wherever the arm was parked, almost all of it in
# y. MAX_POSITION_STEP is no defence - 0.1 m per tick at 20 Hz permits 2 m/s,
# so a half-metre gap is closed in five ticks. The cartesian controller sees
# half a metre of "tracking error" and leans on it with full gain, which is
# the opening lunge: 439 mm/s and 13336 mm/s^2 in r034.
#
# This is not a tuning problem. It is a 0.5 m approach move disguised as a
# control error, and it survived every pd_gains/iterations change precisely
# because gain only sets how hard the lunge is, not whether it happens.
#
# So: slew-rate limit the TARGET for the first APPROACH_DISTANCE_M, giving
# the controller a reachable setpoint instead of a distant one. The arm
# tracks the ramp rather than chasing a step.
# 0.15 m/s, up from 0.08. At 0.08 a 665 mm gap (r035) took 8.3 s, during
# which the hand kept receiving its commands at full rate - the arm was still
# approaching while the policy had moved on. 0.15 halves that to ~4.4 s and
# is still 3x below the 439 mm/s lunge this replaced; measured acceleration
# during the r035 ramp was ~2000 mm/s^2 against 13336 unramped, so there is
# ample headroom.
APPROACH_SPEED_MPS = 0.15
APPROACH_ROT_RAD_S = 0.4       # rad/s on the quaternion geodesic
APPROACH_DONE_M = 0.015        # within 15 mm of target -> ramp is over


class TesolloCartesianExecutor(Node):
    def __init__(self):
        super().__init__("tesollo_cartesian_executor")

        self.policy_active = False
        self._hand_initialised = False

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.schedule = ActionSchedule(maxlen=BUFFER_MAXLEN)

        self.pose_smoother = ButterworthLowPass(
            cutoff_hz=SMOOTHER_CUTOFF_HZ, sample_hz=CONTROL_HZ)
        self.hand_smoother = ButterworthLowPass(
            cutoff_hz=HAND_SMOOTHER_CUTOFF_HZ, sample_hz=CONTROL_HZ)

        self._last_pose: Optional[np.ndarray] = None
        # Default ON, not OFF: the ramp is only disarmed by reaching the start
        # pose. If it defaulted to False, any path that bypasses
        # _policy_gate_cb - node restarted while the policy is already
        # running, gate message missed - would publish the first target
        # unramped, which is the 0.5 m lunge this exists to prevent.
        self._approaching = True
        self._last_hand_deg: Optional[np.ndarray] = None

        # --- Publishers ---
        self.target_pub = self.create_publisher(PoseStamped, TARGET_TOPIC, 10)
        self.hand_pub = self.create_publisher(
            JointTrajectory, HAND_TRAJECTORY_TOPIC, 10)
        self.buffer_status_pub = self.create_publisher(
            Float64MultiArray, "/action_buffer_status", 10)

        # --- Subscribers ---
        self.create_subscription(
            Float64MultiArray, "/predicted_action_chunk",
            self._chunk_callback, 10)
        self.create_subscription(
            Bool, "/policy_gate", self._policy_gate_cb, 10)
        self.create_subscription(
            Bool, "/policy_done", self._policy_done_cb, 10)

        self._exec_timer = self.create_timer(
            1.0 / CONTROL_HZ, self._execute_step)
        self._status_timer = self.create_timer(
            1.0 / 10.0, self._publish_buffer_status)
        self._telemetry_timer = self.create_timer(5.0, self._log_telemetry)

        self._last_chunk_wall = 0.0
        self._steps_executed = 0
        self._steps_starved = 0

        self.get_logger().info(
            f"Tesollo DRTC executor up. action_dim={ACTION_DIM} "
            f"(pose {POSE_DIM} + hand {HAND_DIM}) | {CONTROL_HZ} Hz | "
            f"arm -> {TARGET_TOPIC} ({ROBOT_TIP_LINK}) | "
            f"hand -> {HAND_TRAJECTORY_TOPIC}")

    # ------------------------------------------------------------------
    # Hand command construction
    # ------------------------------------------------------------------

    def _publish_hand_pose(self, hand_deg: np.ndarray, duration_s: float):
        """Send a single-point JointTrajectory for all hand joints.

        All joints are re-stated on every tick rather than left unspecified.
        allow_partial_joints_goal is true on this controller, so a partial goal
        would be accepted and the omitted joints would keep whatever the
        previous trajectory left them at - which drifts under an effort-based
        PID with no position hold.

        Degrees -> radians happens HERE, once. The dataset is in degrees
        because record_utils/record.py converts the driver's radians on the way
        in (HAND_JOINTS_IN_DEGREES=1), so the policy's outputs are degrees too,
        while ros2_control expects radians. Skipping this conversion is a
        silent factor-57 error: the command stays well-formed and the hand just
        slams into its limits.
        """
        full_deg = np.asarray(hand_deg, dtype=np.float64).ravel()

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(HAND_JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = [math.radians(v) for v in full_deg]
        # allow_nonzero_velocity_at_trajectory_end is false on this controller,
        # so the endpoint velocity must be zero or the goal is rejected.
        point.velocities = [0.0] * len(HAND_JOINT_NAMES)
        sec = int(duration_s)
        point.time_from_start = DurationMsg(
            sec=sec, nanosec=int((duration_s - sec) * 1e9))
        msg.points = [point]

        self.hand_pub.publish(msg)

    # ------------------------------------------------------------------
    # Gate callbacks
    # ------------------------------------------------------------------

    def _policy_gate_cb(self, msg: Bool):
        if msg.data and not self.policy_active:
            self.policy_active = True
            self.pose_smoother.reset()
            self.hand_smoother.reset()
            self._last_hand_deg = None
            self._hand_initialised = False
            self._last_pose = self._read_current_pose()
            # Re-armed on every activation: the arm can be parked anywhere
            # between runs, so a completed approach says nothing about the
            # next one.
            self._approaching = True

            # No hand pre-positioning here: the first action of the chunk
            # already carries every joint's target, so the opening move IS the
            # first executed step. It only needs a longer time_from_start than
            # a mid-rollout step, because the hand may be anywhere at this
            # point - see _execute_step.
            self.get_logger().info("Policy ACTIVE.")
            if self._last_pose is not None:
                self.get_logger().info(
                    f"Seeded arm from TF: "
                    f"{np.round(self._last_pose[:3], 4).tolist()}")
            else:
                self.get_logger().warn(
                    "TF unavailable; first arm action is unclamped.")

        elif not msg.data and self.policy_active:
            self._deactivate("Policy INACTIVE - holding last pose.")

    def _policy_done_cb(self, msg: Bool):
        if msg.data and self.policy_active:
            self._deactivate("Policy DONE - task complete.")

    def _deactivate(self, reason: str):
        self.policy_active = False
        self.schedule.flush()
        self.pose_smoother.reset()
        self.hand_smoother.reset()
        self._last_pose = None
        self._approaching = True
        self._last_hand_deg = None
        self._hand_initialised = False
        # The hand is deliberately NOT sent anywhere on deactivation: it may be
        # holding the foil, and opening here would drop it.
        self.get_logger().info(reason)

    # ------------------------------------------------------------------
    # Chunk intake
    # ------------------------------------------------------------------

    def _chunk_callback(self, msg: Float64MultiArray):
        """Wire format: [anchor_step, a0_0 .. a0_26, a1_0 .. ] (ACTION_DIM=27).

        anchor_step >= 0 is an absolute step index assigned by the producer at
        send time; < 0 means "anchor at my cursor".
        """
        if not self.policy_active:
            return

        raw = np.asarray(msg.data, dtype=np.float64)
        if len(raw) < 2:
            return

        anchor_field = raw[0]
        payload = raw[1:]

        if len(payload) % ACTION_DIM != 0:
            # Reject rather than guess. A chunk that still carries the policy's
            # raw 15-dim pinch output, or the old 13-dim Inspire layout, ends
            # up here; reinterpreting it against 27 would feed quaternion
            # components to finger joints. deploy.expand_hand_action() is what
            # turns a reduced prediction into this format.
            self.get_logger().error(
                f"Chunk of {len(payload)} values is not divisible by "
                f"action_dim={ACTION_DIM}. The producer must expand its "
                f"action to {POSE_DIM} pose + {HAND_DIM} hand values "
                f"(see expand_hand_action in deploy.py). Ignoring chunk.",
                throttle_duration_sec=2.0)
            return

        actions = payload.reshape(len(payload) // ACTION_DIM, ACTION_DIM)
        anchor = (int(anchor_field) if anchor_field >= 0
                  else self.schedule.next_action_step())

        timestamp = time.monotonic()
        installed = self.schedule.merge_chunk(
            actions, timestamp, anchor_step=anchor)
        if installed > 0:
            self._last_chunk_wall = timestamp

    # ------------------------------------------------------------------
    # Fixed-rate execution
    # ------------------------------------------------------------------

    def _execute_step(self):
        if not self.policy_active:
            # Keep the cartesian controller's internal target pinned to the
            # live pose, so re-enabling it cannot snap the arm back to a stale
            # setpoint after a manual move.
            current = self._read_current_pose()
            if current is not None:
                self._publish_target_pose(current)
            return

        action = self.schedule.pop_next()
        if action is None:
            self._steps_starved += 1
            self._check_watchdog()
            return

        self._steps_executed += 1

        pose_7d = action[:POSE_DIM].copy()
        hand_deg = action[POSE_DIM:ACTION_DIM].copy()

        # No tool-frame correction: the dataset's ee_pose is already tool0.
        smoothed_pose = self.pose_smoother.filter(pose_7d)
        smoothed_hand = self.hand_smoother.filter(hand_deg)
        smoothed_hand = self._stabilize_hand(smoothed_hand)

        clamped_pose = self._clamp_pose_delta(smoothed_pose)
        if clamped_pose is None:
            return

        if not self._validate_cartesian(clamped_pose):
            self.get_logger().warn(
                f"Arm target out of workspace: "
                f"{clamped_pose[:3].round(3)}. Holding.",
                throttle_duration_sec=1.0)
            return

        self._publish_target_pose(clamped_pose)

        # The very first commanded point is a move from an unknown hand pose to
        # the demonstrated start posture, which can be tens of degrees away. A
        # one-tick deadline would ask for that in 50 ms; the controller would
        # either saturate or reject it.
        if self._hand_initialised:
            hand_duration = HAND_POINT_DURATION_S
        else:
            hand_duration = HAND_INIT_DURATION_S
            self._hand_initialised = True
            self.get_logger().info(
                f"Hand -> first commanded pose over {hand_duration:.1f}s: "
                f"{np.round(smoothed_hand, 1).tolist()} deg")

        self._publish_hand_pose(smoothed_hand, hand_duration)

        self._last_pose = clamped_pose.copy()
        self._last_hand_deg = smoothed_hand.copy()

    def _stabilize_hand(self, hand_deg: np.ndarray) -> np.ndarray:
        """Deadband plus per-tick step clamp, in degrees.

        The deadband is set just above the ~0.1 deg quantisation of the
        recorded hand state, so it suppresses dither without swallowing real
        motion. Contact transients in this task are tens of degrees.

        Applied to every joint, including the ones the policy does not predict:
        those arrive as a constant, so the deadband holds them exactly and the
        step clamp ramps them in rather than stepping on the first tick.
        """
        out = np.asarray(hand_deg, dtype=np.float64).ravel().copy()
        if self._last_hand_deg is None:
            return out

        for i in range(len(out)):
            delta = out[i] - self._last_hand_deg[i]
            if abs(delta) < HAND_DEADBAND_DEG:
                out[i] = self._last_hand_deg[i]
            elif abs(delta) > HAND_MAX_STEP_DEG:
                out[i] = (self._last_hand_deg[i]
                          + math.copysign(HAND_MAX_STEP_DEG, delta))
        return out

    def _approach_limit(self, pose_7d: np.ndarray) -> np.ndarray:
        """Slew-rate limit the target while still approaching the start pose.

        Returns a pose at most APPROACH_SPEED_MPS * tick away from the last
        one, so the controller is always given a setpoint it can reach within
        a cycle. Clears itself once the gap is under APPROACH_DONE_M, after
        which the normal MAX_POSITION_STEP clamp takes over.
        """
        gap = pose_7d[:3] - self._last_pose[:3]
        dist = float(np.linalg.norm(gap))

        if dist < APPROACH_DONE_M:
            self._approaching = False
            self.get_logger().info(
                f"Approach complete ({dist * 1000:.0f} mm); "
                f"handing over to the policy stream.")
            return pose_7d

        max_step = APPROACH_SPEED_MPS / CONTROL_HZ
        pos = (pose_7d[:3] if dist <= max_step
               else self._last_pose[:3] + gap * (max_step / dist))

        quat = pose_7d[3:7].copy()
        last_quat = self._last_pose[3:7]
        if float(np.dot(quat, last_quat)) < 0.0:
            quat = -quat
        n = float(np.linalg.norm(quat))
        if n > 0.0:
            quat = quat / n

        max_chord = APPROACH_ROT_RAD_S / CONTROL_HZ
        chord = float(np.linalg.norm(quat - last_quat))
        if chord > max_chord:
            quat = last_quat + (quat - last_quat) * (max_chord / chord)
            n = float(np.linalg.norm(quat))
            if n > 0.0:
                quat = quat / n

        self.get_logger().info(
            f"Approaching start pose: {dist * 1000:.0f} mm to go",
            throttle_duration_sec=1.0)
        return np.concatenate([pos, quat])

    def _clamp_pose_delta(self, pose_7d: np.ndarray) -> Optional[np.ndarray]:
        """Cap per-tick translation and rotation; last line of defence."""
        if self._last_pose is None:
            # Seeding from TF failed at activation. Retry here rather than
            # waving the target through: without a reference pose there is no
            # way to rate-limit anything, and the first target is typically
            # half a metre out. TF usually arrives within a tick or two.
            self._last_pose = self._read_current_pose()
            if self._last_pose is None:
                self.get_logger().warn(
                    "No TF yet; holding instead of publishing an unramped "
                    "target.", throttle_duration_sec=1.0)
                return None

        # The approach ramp is a tighter limit on the same quantity, so it
        # replaces this clamp rather than stacking with it.
        if self._approaching:
            limited = self._approach_limit(pose_7d)
            return limited if self._validate_cartesian(limited) else None

        pos_delta = pose_7d[:3] - self._last_pose[:3]
        dist = float(np.linalg.norm(pos_delta))
        if dist > MAX_POSITION_STEP:
            pos = self._last_pose[:3] + (pos_delta / dist) * MAX_POSITION_STEP
        else:
            pos = pose_7d[:3]

        quat = pose_7d[3:7].copy()
        last_quat = self._last_pose[3:7]

        # q and -q are the same rotation; pick the hemisphere that makes the
        # chord distance below meaningful.
        if float(np.dot(quat, last_quat)) < 0.0:
            quat = -quat

        norm = float(np.linalg.norm(quat))
        if norm > 0.0:
            quat = quat / norm

        chord = float(np.linalg.norm(quat - last_quat))
        if chord > MAX_ROTATION_STEP:
            t = MAX_ROTATION_STEP / chord
            quat = last_quat + t * (quat - last_quat)
            norm = float(np.linalg.norm(quat))
            if norm > 0.0:
                quat = quat / norm

        return np.concatenate([pos, quat])

    # ------------------------------------------------------------------
    # Publishing, validation, telemetry
    # ------------------------------------------------------------------

    def _publish_target_pose(self, pose_7d: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ROBOT_BASE_LINK
        msg.pose.position.x = float(pose_7d[0])
        msg.pose.position.y = float(pose_7d[1])
        msg.pose.position.z = float(pose_7d[2])
        msg.pose.orientation.x = float(pose_7d[3])
        msg.pose.orientation.y = float(pose_7d[4])
        msg.pose.orientation.z = float(pose_7d[5])
        msg.pose.orientation.w = float(pose_7d[6])
        self.target_pub.publish(msg)

    def _publish_buffer_status(self):
        msg = Float64MultiArray()
        msg.data = [
            float(self.schedule.queue_depth()),
            float(self.schedule.cursor()),
            float(self.schedule.next_action_step()),
        ]
        self.buffer_status_pub.publish(msg)

    def _validate_cartesian(self, pose_7d: np.ndarray) -> bool:
        x, y, z = float(pose_7d[0]), float(pose_7d[1]), float(pose_7d[2])
        if not (WORKSPACE["x"][0] <= x <= WORKSPACE["x"][1] and
                WORKSPACE["y"][0] <= y <= WORKSPACE["y"][1] and
                WORKSPACE["z"][0] <= z <= WORKSPACE["z"][1]):
            return False
        return 0.90 <= float(np.linalg.norm(pose_7d[3:7])) <= 1.10

    def _read_current_pose(self) -> Optional[np.ndarray]:
        try:
            t = self._tf_buffer.lookup_transform(
                ROBOT_BASE_LINK, ROBOT_TIP_LINK, rclpy.time.Time(),
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
        if self._last_chunk_wall == 0.0:
            return
        elapsed = time.monotonic() - self._last_chunk_wall
        if elapsed > WATCHDOG_TIMEOUT_S:
            self.get_logger().warn(
                f"No action chunk for {elapsed:.1f}s - inference stalled?",
                throttle_duration_sec=WATCHDOG_TIMEOUT_S)

    def _log_telemetry(self):
        if not self.policy_active:
            return
        total = self._steps_executed + self._steps_starved
        if total == 0:
            return
        self.get_logger().info(
            f"[EXEC] steps={self._steps_executed} "
            f"starved={self._steps_starved} "
            f"({100.0 * self._steps_starved / total:.1f}%) | "
            f"buffer={self.schedule.queue_depth()} | "
            f"cursor={self.schedule.cursor()}")
        self._steps_executed = 0
        self._steps_starved = 0


def main(args=None):
    rclpy.init(args=args)
    node = TesolloCartesianExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()