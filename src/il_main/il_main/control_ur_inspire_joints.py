#!/usr/bin/env python3
"""
Inference Control Node for the AUXILIARY-ACTION (joint-space) policy,
using the JointTrajectoryController (JTC) for SAFE execution.

Why JTC (not forward_position_controller):
  - Spline interpolation between waypoints (no raw position jumps)
  - Enforces the velocity/acceleration limits configured in the controller
  - Rejects malformed trajectories at the controller level
  - Time-parameterized execution (time_from_start)

The aux policy predicts JOINT STATES directly (6 UR + 6 hand), so there is
NO IK. We simply validate for safety and forward the joint waypoints as a
trajectory. This is the joint-space analogue of control_ur_inspire_traj.py
but WITHOUT the Pinocchio/Pink IK stage.

Action layout per step (from deploy_aux.publish_action_chunk):
    [0:6]  = UR joint positions (rad)
    [6:12] = Inspire hand joint values (normalized 0..1)

SAFETY LAYERS:
  1. Per-joint position limits (URDF-derived, hard clamp + reject)
  2. Entry-jump check: first waypoint vs. current pose (RTC handles the rest)
  3. Per-step velocity check across the chunk (reject spiky chunks)
  4. JTC internal spline + limit enforcement
"""
import os
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from service_interfaces.msg import SetAngle1

UR_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]

# UR10e joint position limits (rad). Conservative — reject anything outside.
# Order matches UR_JOINT_NAMES.
UR_JOINT_LIMITS = [
    (-2.0 * np.pi, 2.0 * np.pi),   # shoulder_pan
    (-2.0 * np.pi, 2.0 * np.pi),   # shoulder_lift
    (-np.pi,       np.pi),         # elbow (mechanically limited on UR10e)
    (-2.0 * np.pi, 2.0 * np.pi),   # wrist_1
    (-2.0 * np.pi, 2.0 * np.pi),   # wrist_2
    (-2.0 * np.pi, 2.0 * np.pi),   # wrist_3
]


class InferenceControlNode(Node):
    def __init__(self):
        super().__init__('inference_control_node_joints')

        # --- Parameters ---
        self.declare_parameter('execution_fps', 20.0)
        self.declare_parameter('trajectory_speed_scale', 0.3)
        self.declare_parameter('lead_time', 0.1)
        # Safety thresholds
        self.declare_parameter('max_entry_jump', 0.15)      # rad, first-point vs current
        self.declare_parameter('max_step_vel', 1.0)        # rad/s between waypoints

        self.execution_fps = self.get_parameter('execution_fps').value
        self.speed_scale = self.get_parameter('trajectory_speed_scale').value
        self.dt_per_step = (1.0 / self.execution_fps) / self.speed_scale
        self.lead_time = self.get_parameter('lead_time').value
        self.max_entry_jump = self.get_parameter('max_entry_jump').value
        self.max_step_vel = self.get_parameter('max_step_vel').value

        # --- State ---
        self.current_joint_positions = np.zeros(6)
        self._joint_state_dict = {name: 0.0 for name in UR_JOINT_NAMES}
        self.joint_state_received = False
        self.last_sent_joints = None
        self.policy_active = False
        self.current_hand_goal = None

        # --- ROS Interfaces ---
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10)
        self.hand_pub = self.create_publisher(SetAngle1, '/set_angle_data', 10)

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10)
        self.chunk_sub = self.create_subscription(
            Float64MultiArray, '/predicted_action_chunk', self._chunk_callback, 10)
        self.policy_gate_sub = self.create_subscription(
            Bool, '/policy_gate', self._policy_gate_cb, 10)
        self.policy_done_sub = self.create_subscription(
            Bool, '/policy_done', self._policy_done_cb, 10)

        self.hand_timer = self.create_timer(1.0 / 30.0, self._hand_loop)

        self.get_logger().info("Joint-Space JTC Controller initialized (INACTIVE).")
        self.get_logger().info(f"Execution FPS: {self.execution_fps}, Speed: {self.speed_scale}")
        self.get_logger().info("Publish /policy_gate True to enable.")

    # =========================================================================
    # Gates
    # =========================================================================

    def _policy_gate_cb(self, msg: Bool):
        if msg.data and not self.policy_active:
            self.policy_active = True
            self.last_sent_joints = self.current_joint_positions.copy()
            self.get_logger().info("✅ Policy ACTIVE")
        elif not msg.data and self.policy_active:
            self._stop()
            self.get_logger().info("⏸️  Policy INACTIVE — Robot stopped.")

    def _policy_done_cb(self, msg: Bool):
        if msg.data and self.policy_active:
            self._stop()
            self.get_logger().info("🏁 Policy DONE — Robot stopped.")

    def _stop(self):
        self.policy_active = False
        self.last_sent_joints = None
        self.current_hand_goal = None
        self._send_stop()

    def _joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joint_state_dict:
                self._joint_state_dict[name] = pos
        self.current_joint_positions = np.array(
            [self._joint_state_dict[n] for n in UR_JOINT_NAMES])
        self.joint_state_received = True

    # =========================================================================
    # Chunk intake → validate → JointTrajectory
    # =========================================================================

    def _chunk_callback(self, msg: Float64MultiArray):
        if not self.policy_active or not self.joint_state_received:
            return

        data = np.array(msg.data)
        action_dim = 12  # 6 UR joints + 6 hand
        chunk_size = len(data) // action_dim
        actions = data.reshape(chunk_size, action_dim)

        arm = actions[:, :6]
        hand_last = actions[-1, 6:12]

        # --- SAFETY 1: joint position limits (reject if any waypoint OOB) ---
        for j, (lo, hi) in enumerate(UR_JOINT_LIMITS):
            if np.any(arm[:, j] < lo) or np.any(arm[:, j] > hi):
                self.get_logger().warn(
                    f"Chunk REJECTED: joint {UR_JOINT_NAMES[j]} out of limits "
                    f"[{arm[:, j].min():.2f}, {arm[:, j].max():.2f}] vs [{lo:.2f}, {hi:.2f}]")
                return

        # --- SAFETY 2: entry jump (first waypoint vs current pose) ---
        ref = (self.last_sent_joints
               if self.last_sent_joints is not None
               else self.current_joint_positions)
        entry_jump = np.max(np.abs(arm[0] - ref))
        if entry_jump > self.max_entry_jump:
            self.get_logger().warn(
                f"Chunk REJECTED: entry jump {entry_jump:.3f} rad > {self.max_entry_jump} rad.")
            return

        # --- SAFETY 3: per-step velocity across the chunk ---
        if len(arm) > 1:
            step_deltas = np.abs(np.diff(arm, axis=0))  # (T-1, 6)
            max_vel = step_deltas.max() / self.dt_per_step
            if max_vel > self.max_step_vel:
                self.get_logger().warn(
                    f"Chunk REJECTED: peak joint vel {max_vel:.2f} rad/s > {self.max_step_vel} rad/s.")
                return

        # All checks passed → publish trajectory + update hand goal
        self.last_sent_joints = arm[-1].copy()
        self.current_hand_goal = hand_last
        self._publish_trajectory(arm)

    def _publish_trajectory(self, joint_waypoints):
        traj = JointTrajectory()
        # stamp=0 → execute immediately from 'now' (avoids wall-clock drift/jerk)
        traj.header.stamp = rclpy.time.Time().to_msg()
        traj.joint_names = UR_JOINT_NAMES

        for i, joints in enumerate(joint_waypoints):
            point = JointTrajectoryPoint()
            point.positions = joints.tolist()
            t = self.lead_time + (i + 1) * self.dt_per_step
            point.time_from_start = Duration(seconds=t).to_msg()
            traj.points.append(point)

        self.traj_pub.publish(traj)
        self.get_logger().debug(
            f"Sent trajectory: {len(joint_waypoints)} points, "
            f"duration {len(joint_waypoints) * self.dt_per_step:.2f}s")

    def _send_stop(self):
        """Single point at current position = controlled stop."""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = UR_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = self.current_joint_positions.tolist()
        point.time_from_start = Duration(seconds=0.2).to_msg()
        traj.points.append(point)
        self.traj_pub.publish(traj)

    # =========================================================================
    # Hand Control
    # =========================================================================

    def _hand_loop(self):
        if not self.policy_active or self.current_hand_goal is None:
            return
        self._send_hand(self.current_hand_goal)

    def _send_hand(self, hand_joints):
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        reversed_joints = hand_joints[::-1]
        msg.angles = [int((1 - x) * 1000) for x in reversed_joints] + [0]
        self.hand_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()