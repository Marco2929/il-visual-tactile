#!/usr/bin/env python3
"""
Inference Control Node using MoveIt Servo.

Unlike the forward-position and joint-trajectory variants, MoveIt Servo does
the IK internally. We simply stream Cartesian pose targets from the RTC action
chunk at a fixed rate; Servo handles IK, singularity avoidance, and smoothing.

Architecture mirrors control_ur_inspire.py:
  - policy_gate / policy_done gates
  - queue-based streaming (non-blocking, RTC-compatible)
  - a timer pops one pose per tick and publishes to Servo + hand
"""

import os
import sys
import threading
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import ServoCommandType

from service_interfaces.msg import SetAngle1
sys.path.insert(0, "/ros2_ws/data/record_utils")
from camera_info import workspace

ROBOT_BASE_FRAME = os.environ.get("ROBOT_BASE_FRAME", "base_link")


class InferenceControlNode(Node):
    def __init__(self):
        super().__init__('inference_control_node')

        # --- State ---
        self.robot_base_frame = ROBOT_BASE_FRAME

        self._queue_lock = threading.Lock()
        self._action_queue = deque()
        self.CHUNK_SIZE = 32

        self.current_pose_goal = None   # 7D: xyz + quat
        self.current_hand_goal = None   # 6D
        self.joint_state_received = False

        # --- Policy gate (INACTIVE by default) ---
        self.policy_active = False

        # --- Execution rate: how fast we stream poses to Servo ---
        self.execution_fps = 20.0

        # --- EMA smoothing factor for streamed poses ---
        # Damps residual ~16mm chunk-boundary jitter from flow-matching sampling.
        # Lower = smoother but more lag; higher = more responsive but jerkier.
        #   0.4 = good starting point, 0.25 = very smooth, 0.6 = responsive
        self.ema_alpha = float(os.environ.get("SERVO_EMA_ALPHA", "0.4"))

        # --- ROS Interfaces ---
        # Servo pose target stream (Servo does IK internally)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/servo_node/pose_target_cmds', 10)
        self.hand_pub = self.create_publisher(
            SetAngle1, '/set_angle_data', 10)

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10)
        self.chunk_sub = self.create_subscription(
            Float64MultiArray, '/predicted_action_chunk', self._chunk_callback, 10)

        # Policy gate + done
        self.policy_gate_sub = self.create_subscription(
            Bool, '/policy_gate', self._policy_gate_cb, 10)
        self.policy_done_sub = self.create_subscription(
            Bool, '/policy_done', self._policy_done_cb, 10)

        # Switch Servo into pose-tracking mode
        self.switch_servo_mode()

        # --- Control Loop: stream one pose per tick ---
        self.stream_timer = self.create_timer(
            1.0 / self.execution_fps, self._stream_loop)

        self.get_logger().info("Servo Robot Controller Initialized (INACTIVE by default).")
        self.get_logger().info("Publish /policy_gate True to enable.")

    # -------------------------------------------------------------------------
    # Servo Setup
    # -------------------------------------------------------------------------

    def switch_servo_mode(self):
        """Switch Servo to POSE command type (command_type=2)."""
        client = self.create_client(
            ServoCommandType, '/servo_node/switch_command_type')
        if client.wait_for_service(timeout_sec=2.0):
            req = ServoCommandType.Request()
            req.command_type = 2  # POSE
            client.call_async(req)
            self.get_logger().info("Servo switched to POSE mode.")
        else:
            self.get_logger().warn("Servo service not available, skipping switch.")

    # -------------------------------------------------------------------------
    # ROS Callbacks
    # -------------------------------------------------------------------------

    def _policy_gate_cb(self, msg: Bool):
        """Gate: enable/disable policy execution on the robot."""
        if msg.data and not self.policy_active:
            self.policy_active = True
            self.get_logger().info("✅ Policy ACTIVE — Robot will follow policy commands.")
        elif not msg.data and self.policy_active:
            self.policy_active = False
            with self._queue_lock:
                self._action_queue.clear()
            self.current_pose_goal = None
            self.current_hand_goal = None
            self.get_logger().info("⏸️  Policy INACTIVE — Queue cleared, robot stopped.")

    def _policy_done_cb(self, msg: Bool):
        """Auto-stop: progress head signals task completion."""
        if msg.data and self.policy_active:
            self.policy_active = False
            with self._queue_lock:
                self._action_queue.clear()
            self.current_pose_goal = None
            self.current_hand_goal = None
            self.get_logger().info("🏁 Policy DONE — Progress threshold reached. Robot stopped.")

    def _joint_state_cb(self, msg):
        self.joint_state_received = True

    def _chunk_callback(self, msg: Float64MultiArray):
        """Receives action chunks — only processed when policy is active.

        Unlike the IK-based controllers, no threaded IK is needed: Servo does
        IK internally. We just validate and enqueue the Cartesian poses.
        """
        if not self.policy_active:
            return

        data = np.array(msg.data)
        action_dim = 13
        chunk_size = len(data) // action_dim
        actions_raw = data.reshape(chunk_size, action_dim)[:self.CHUNK_SIZE]

        valid_actions = []
        for i, action in enumerate(actions_raw):
            pose_7d = action[:7]
            if not self._validate_cartesian(pose_7d):
                self.get_logger().warn(
                    f"Action {i} FAILED validation: xyz={pose_7d[:3]}, "
                    f"quat_norm={np.linalg.norm(pose_7d[3:7]):.4f}")
                break
            valid_actions.append(action)

        if not valid_actions:
            self.get_logger().error("NO VALID ACTIONS! Queue stays empty → no movement!")
            return

        self.get_logger().info(
            f"Chunk received: {len(valid_actions)}/{len(actions_raw)} actions valid")

        # RTC (in deploy.py) guarantees continuity between chunks. Just replace
        # the queue — each new chunk supersedes the old one.
        with self._queue_lock:
            self._action_queue.clear()
            for act in valid_actions:
                self._action_queue.append(act)

    # -------------------------------------------------------------------------
    # Control Loop: stream poses to Servo
    # -------------------------------------------------------------------------

    def _stream_loop(self):
        """Pop one action per tick and stream it to Servo + hand."""
        if not self.policy_active:
            return

        with self._queue_lock:
            if self._action_queue:
                action = self._action_queue.popleft()

                new_pose = action[:7].copy()

                if self.current_pose_goal is not None:
                    a = self.ema_alpha
                    prev = self.current_pose_goal

                    # --- Quaternion hemisphere alignment (fixes wrist wobble) ---
                    # Network may output -q instead of +q (same rotation, but
                    # Servo reads the sign flip as a ~180° jump). Align first.
                    if np.dot(new_pose[3:7], prev[3:7]) < 0.0:
                        new_pose[3:7] = -new_pose[3:7]

                    # --- EMA smoothing on position ---
                    new_pose[:3] = a * new_pose[:3] + (1.0 - a) * prev[:3]

                    # --- EMA smoothing on quaternion, then renormalize ---
                    q = a * new_pose[3:7] + (1.0 - a) * prev[3:7]
                    q_norm = np.linalg.norm(q)
                    if q_norm > 1e-8:
                        new_pose[3:7] = q / q_norm

                self.current_pose_goal = new_pose
                self.current_hand_goal = action[7:13]

        # Servo needs a continuous stream — re-publish current goal every tick
        # (even if the queue is momentarily empty) to keep tracking alive.
        if self.current_pose_goal is not None:
            self._send_pose(self.current_pose_goal)

        if self.current_hand_goal is not None:
            self._send_hand(self.current_hand_goal)

    def _send_pose(self, pose_7d):
        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = self.robot_base_frame

        target.pose.position.x = float(pose_7d[0])
        target.pose.position.y = float(pose_7d[1])
        target.pose.position.z = float(pose_7d[2])
        target.pose.orientation.x = float(pose_7d[3])
        target.pose.orientation.y = float(pose_7d[4])
        target.pose.orientation.z = float(pose_7d[5])
        target.pose.orientation.w = float(pose_7d[6])

        self.pose_pub.publish(target)

    def _send_hand(self, hand_joints):
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        reversed_joints = hand_joints[::-1]
        # Trailing 0 matches the working forward-position controller convention
        msg.angles = [int((1 - x) * 1000) for x in reversed_joints] + [0]
        self.hand_pub.publish(msg)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_cartesian(self, pose_7d):
        x, y, z = pose_7d[0], pose_7d[1], pose_7d[2]
        if not (workspace['x'][0] <= x <= workspace['x'][1] and
                workspace['y'][0] <= y <= workspace['y'][1] and
                workspace['z'][0] <= z <= workspace['z'][1]):
            return False

        quat_norm = np.linalg.norm(pose_7d[3:7])
        return 0.99 <= quat_norm <= 1.01


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