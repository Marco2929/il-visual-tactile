#!/usr/bin/env python3
"""
Inference Control Node using JointTrajectoryController.
Sends full trajectory chunks — controller handles smoothing, velocity, and acceleration.
"""

import os
import sys
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask

from service_interfaces.msg import SetAngle1
sys.path.insert(0, "/ros2_ws/data/record_utils")
from camera_info import workspace

URDF_PATH = os.environ.get(
    "URDF_PATH",
    "/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf"
)
ROBOT_TIP_FRAME = os.environ.get("ROBOT_TIP_FRAME", "right_inspire_base_link")

UR_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]

HAND_JOINT_NAMES = [
    'right_inspire_right_thumb_1_joint', 'right_inspire_right_thumb_2_joint',
    'right_inspire_right_index_1_joint', 'right_inspire_right_middle_1_joint',
    'right_inspire_right_ring_1_joint', 'right_inspire_right_little_1_joint'
]


class InferenceControlNode(Node):
    def __init__(self):
        super().__init__('inference_control_node')

        # --- Parameters ---
        self.declare_parameter('execution_fps', 20.0)
        self.declare_parameter('max_joint_delta', 0.3)  # rad, IK flip protection
        self.declare_parameter('trajectory_speed_scale', 0.3)  # <1 = slower, >1 = faster
        self.declare_parameter('lead_time', 0.1)
        # Safety thresholds (from joints controller)
        self.declare_parameter('max_entry_jump', 0.3)    # rad, first-point vs current
        self.declare_parameter('max_step_vel', 1.0)       # rad/s between waypoints

        self.execution_fps = self.get_parameter('execution_fps').value
        self.max_joint_delta = self.get_parameter('max_joint_delta').value
        self.speed_scale = self.get_parameter('trajectory_speed_scale').value
        self.dt_per_step = (1.0 / self.execution_fps) / self.speed_scale
        self.lead_time = self.get_parameter('lead_time').value
        self.max_entry_jump = self.get_parameter('max_entry_jump').value
        self.max_step_vel = self.get_parameter('max_step_vel').value

        # --- Kinematics Setup (Pinocchio & Pink) ---
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()

        self.ee_task = FrameTask(
            ROBOT_TIP_FRAME,
            position_cost=1.0,
            orientation_cost=1.0
        )
        self.posture_task = PostureTask(cost=1e-2)

        # --- State ---
        self.current_joint_positions = np.zeros(6)
        self._joint_state_dict = {name: 0.0 for name in UR_JOINT_NAMES}
        self.joint_state_received = False
        self.last_ik_solution = None
        self.policy_active = False
        self._ik_thread = None

        # --- ROS Interfaces ---
        # Trajectory publisher (topic interface)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Hand publisher
        self.hand_pub = self.create_publisher(SetAngle1, '/set_angle_data', 10)

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10)
        self.chunk_sub = self.create_subscription(
            Float64MultiArray, '/predicted_action_chunk', self._chunk_callback, 10)
        self.policy_gate_sub = self.create_subscription(
            Bool, '/policy_gate', self._policy_gate_cb, 10)
        self.policy_done_sub = self.create_subscription(
            Bool, '/policy_done', self._policy_done_cb, 10)

        # Hand timer (sends hand commands at 30Hz from latest goal)
        self.current_hand_goal = None
        self.hand_timer = self.create_timer(1.0 / 30.0, self._hand_loop)

        self.get_logger().info("JointTrajectory Controller Node initialized (INACTIVE).")
        self.get_logger().info(f"Execution FPS: {self.execution_fps}, Speed Scale: {self.speed_scale}")
        self.get_logger().info("Publish /policy_gate True to enable.")

    # =========================================================================
    # ROS Callbacks
    # =========================================================================

    def _policy_gate_cb(self, msg: Bool):
        if msg.data and not self.policy_active:
            self.policy_active = True
            self.last_ik_solution = self.current_joint_positions.copy()
            self.get_logger().info("✅ Policy ACTIVE")
        elif not msg.data and self.policy_active:
            self.policy_active = False
            self.last_ik_solution = None
            self.current_hand_goal = None
            # Send empty trajectory to stop
            self._send_stop()
            self.get_logger().info("⏸️  Policy INACTIVE — Robot stopped.")

    def _policy_done_cb(self, msg: Bool):
        """Auto-stop: progress head signals task completion."""
        if msg.data and self.policy_active:
            self.policy_active = False
            self.last_ik_solution = None
            self.current_hand_goal = None
            self._send_stop()
            self.get_logger().info("🏁 Policy DONE — Progress threshold reached. Robot stopped.")

    def _joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joint_state_dict:
                self._joint_state_dict[name] = pos

        self.current_joint_positions = np.array(
            [self._joint_state_dict[n] for n in UR_JOINT_NAMES])
        self.joint_state_received = True

    def _chunk_callback(self, msg: Float64MultiArray):
        if not self.policy_active:
            return
        if not self.joint_state_received:
            return
        if self._ik_thread is not None and self._ik_thread.is_alive():
            self.get_logger().debug("IK thread busy — dropping chunk.")
            return

        data = np.array(msg.data)
        self._ik_thread = threading.Thread(
            target=self._process_and_send_chunk, args=(data,), daemon=True)
        self._ik_thread.start()

    # =========================================================================
    # Core: Process Chunk → IK → JointTrajectory
    # =========================================================================

    def _process_and_send_chunk(self, data):
        """Convert Cartesian chunk to joint trajectory and publish."""
        action_dim = 13
        chunk_size = len(data) // action_dim
        actions_raw = data.reshape(chunk_size, action_dim)

        # IK reference: last solution or current position
        ik_ref = (self.last_ik_solution.copy()
                  if self.last_ik_solution is not None
                  else self.current_joint_positions.copy())

        joint_waypoints = []
        hand_waypoints = []

        for action in actions_raw:
            pose_7d = action[:7]
            hand = action[7:13]

            if not self._validate_cartesian(pose_7d):
                self.get_logger().debug("Cartesian validation failed — stopping chunk.")
                break

            joints = self._cartesian_to_joints(pose_7d, ik_ref)
            if joints is None:
                self.get_logger().warn(
                    f"IK failed: target={pose_7d[:3].round(3)}, quat_norm={np.linalg.norm(pose_7d[3:7]):.4f}")
                continue

            ik_ref = joints
            joint_waypoints.append(joints)
            hand_waypoints.append(hand)

        if not joint_waypoints:
            self.get_logger().warn("No valid IK solutions in chunk.")
            return

        arm = np.array(joint_waypoints)  # (N, 6)

        # --- SAFETY: entry jump (first waypoint vs current/last pose) ---
        ref = (self.last_ik_solution
               if self.last_ik_solution is not None
               else self.current_joint_positions)
        entry_jump = np.max(np.abs(arm[0] - ref))
        if entry_jump > self.max_entry_jump:
            self.get_logger().warn(
                f"Chunk REJECTED: entry jump {entry_jump:.3f} rad > {self.max_entry_jump} rad.")
            return

        # --- SAFETY: per-step velocity across the chunk ---
        if len(arm) > 1:
            step_deltas = np.abs(np.diff(arm, axis=0))
            max_vel = step_deltas.max() / self.dt_per_step
            if max_vel > self.max_step_vel:
                self.get_logger().warn(
                    f"Chunk REJECTED: peak joint vel {max_vel:.2f} rad/s > {self.max_step_vel} rad/s.")
                return

        # Update last IK solution for next chunk
        self.last_ik_solution = joint_waypoints[-1].copy()

        # Update hand goal (last waypoint)
        self.current_hand_goal = hand_waypoints[-1]

        # Build and send JointTrajectory
        self._publish_trajectory(joint_waypoints)

    def _publish_trajectory(self, joint_waypoints):
        """Build JointTrajectory message and publish."""
        traj = JointTrajectory()
        # stamp=0 → controller starts executing immediately from 'now', rather
        # than trying to align to an absolute wall-clock time (avoids drift/jerk).
        traj.header.stamp = rclpy.time.Time().to_msg()
        traj.joint_names = UR_JOINT_NAMES

        for i, joints in enumerate(joint_waypoints):
            point = JointTrajectoryPoint()
            point.positions = joints.tolist()
            # Time from start: lead_time offset + spacing. lead_time gives the
            # controller room to reach the first point smoothly.
            t = self.lead_time + (i + 1) * self.dt_per_step
            point.time_from_start = Duration(seconds=t).to_msg()
            traj.points.append(point)

        self.traj_pub.publish(traj)
        self.get_logger().debug(
            f"Sent trajectory: {len(joint_waypoints)} points, "
            f"duration: {len(joint_waypoints) * self.dt_per_step:.2f}s")

    def _send_stop(self):
        """Send empty trajectory to stop the robot."""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = UR_JOINT_NAMES
        # Single point at current position = stop
        point = JointTrajectoryPoint()
        point.positions = self.current_joint_positions.tolist()
        point.time_from_start = Duration(seconds=0.1).to_msg()
        traj.points.append(point)
        self.traj_pub.publish(traj)

    # =========================================================================
    # Hand Control
    # =========================================================================

    def _hand_loop(self):
        """Send hand commands at fixed rate."""
        if not self.policy_active or self.current_hand_goal is None:
            return
        self._send_hand(self.current_hand_goal)

    def _send_hand(self, hand_joints):
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        reversed_joints = hand_joints[::-1]
        msg.angles = [int((1 - x) * 1000) for x in reversed_joints]
        self.hand_pub.publish(msg)

    # =========================================================================
    # IK & Validation
    # =========================================================================

    def _validate_cartesian(self, pose_7d):
        x, y, z = pose_7d[0], pose_7d[1], pose_7d[2]
        if not (workspace['x'][0] <= x <= workspace['x'][1] and
                workspace['y'][0] <= y <= workspace['y'][1] and
                workspace['z'][0] <= z <= workspace['z'][1]):
            return False
        quat_norm = np.linalg.norm(pose_7d[3:7])
        return 0.99 <= quat_norm <= 1.01

    def _cartesian_to_joints(self, pose_7d, ref_joints):
        """IK: Cartesian pose → joint positions."""
        pos = pose_7d[:3]
        quat = pin.Quaternion(pose_7d[6], pose_7d[3], pose_7d[4], pose_7d[5])
        target_pose = pin.SE3(quat, pos)

        self.ee_task.set_target(target_pose)

        target_q = np.zeros(self.model.nq)
        target_q[:6] = ref_joints
        self.posture_task.set_target(target_q)

        q = np.zeros(self.model.nq)
        q[:6] = ref_joints

        configuration = pink.Configuration(self.model, self.data, q)

        try:
            for _ in range(10):  # More iterations for convergence
                velocity = pink.solve_ik(
                    configuration,
                    [self.ee_task, self.posture_task],
                    dt=1.0,
                    solver="quadprog",
                    damping=1e-2
                )
                q = configuration.integrate(velocity, dt=1.0)
                configuration = pink.Configuration(self.model, self.data, q)

            joints = q[:6]

            # IK flip protection
            delta = np.max(np.abs(joints - ref_joints))
            if delta > self.max_joint_delta:
                self.get_logger().debug(f"IK flip avoided: delta={delta:.3f} > {self.max_joint_delta}")
                return None

            return joints

        except Exception as e:
            self.get_logger().error(f"IK solver failed: {e}")
            return None


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