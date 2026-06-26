#!/usr/bin/env python3
import os
import sys
import threading
import time
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool

import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask

from service_interfaces.msg import SetAngle1
sys.path.insert(0, "/ros2_ws/data/record_utils")
from camera_info import workspace

URDF_PATH = os.environ.get(
    "FOLDER_NAME",
    "/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf"
)
ROBOT_TIP_FRAME = os.environ.get("ROBOT_TIP_FRAME", "right_inspire_base_link")
UR_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]


class InferenceControlNode(Node):
    def __init__(self):
        super().__init__('inference_control_node')

        # --- 1. Kinematics Setup (Pinocchio & Pink) ---
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()

        self.ee_task = FrameTask(
            ROBOT_TIP_FRAME,
            position_cost=1.0,
            orientation_cost=1.0
        )
        self.posture_task = PostureTask(cost=1e-2)

        # --- 2. State Variables ---
        self.current_joint_positions = np.zeros(6)
        self._joint_state_dict = {name: 0.0 for name in UR_JOINT_NAMES}
        self.joint_state_received = False

        self._queue_lock = threading.Lock()
        self._action_queue = deque()
        self._ik_thread = None

        self.last_commanded_positions = None
        self.current_arm_goal = None
        self.current_hand_goal = None
        self.CHUNK_SIZE = 32

        # --- Policy gate (INACTIVE by default) ---
        self.policy_active = False

        # --- 3. ROS Interfaces ---
        self.arm_cmd_pub = self.create_publisher(
            Float64MultiArray, '/forward_position_controller/commands', 10)
        self.hand_pub = self.create_publisher(
            SetAngle1, '/set_angle_data', 10)

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10)
        self.chunk_sub = self.create_subscription(
            Float64MultiArray, '/predicted_action_chunk', self._chunk_callback, 10)

        # --- Policy gate subscription ---
        self.policy_gate_sub = self.create_subscription(
            Bool, '/policy_gate', self._policy_gate_cb, 10)

        # --- 4. Control Loops ---
        self.queue_timer = self.create_timer(1.0 / 30.0, self._queue_pop_loop)
        self.control_timer = self.create_timer(1.0 / 500.0, self._motor_control_loop)

        self.get_logger().info("JTC Robot Controller Initialized (INACTIVE by default).")
        self.get_logger().info("Publish /policy_gate True to enable.")

    # -------------------------------------------------------------------------
    # ROS Callbacks
    # -------------------------------------------------------------------------

    def _policy_gate_cb(self, msg: Bool):
        """Gate: enable/disable policy execution on the robot."""
        if msg.data and not self.policy_active:
            self.policy_active = True
            self.last_commanded_positions = None  # ← re-init from actual joints on next use
            self.get_logger().info("✅ Policy ACTIVE — Robot will follow policy commands.")
        elif not msg.data and self.policy_active:
            self.policy_active = False
            with self._queue_lock:
                self._action_queue.clear()
            self.current_arm_goal = None
            self.current_hand_goal = None
            self.last_commanded_positions = None  # ← ADD THIS
            self.get_logger().info("⏸️  Policy INACTIVE — Queue cleared, robot stopped.")

    def _joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joint_state_dict:
                self._joint_state_dict[name] = pos

        self.current_joint_positions = np.array(
            [self._joint_state_dict[n] for n in UR_JOINT_NAMES])
        self.joint_state_received = True

    def _chunk_callback(self, msg: Float64MultiArray):
        """Receives action chunks — only processed when policy is active."""
        if not self.policy_active:
            return

        if not self.joint_state_received:
            return

        if self._ik_thread is not None and self._ik_thread.is_alive():
            self.get_logger().warn("IK Thread busy. Dropping chunk.")
            return

        data = np.array(msg.data)
        self._ik_thread = threading.Thread(
            target=self._process_action_chunk, args=(data,), daemon=True)
        self._ik_thread.start()

    # -------------------------------------------------------------------------
    # Core Processing: IK & Temporal Ensembling
    # -------------------------------------------------------------------------

    def _process_action_chunk(self, data):
        action_dim = 13
        chunk_size = len(data) // action_dim
        actions_raw = data.reshape(chunk_size, action_dim)[:self.CHUNK_SIZE]

        processed_actions = []

        ik_ref = (self.last_commanded_positions.copy()
                  if self.last_commanded_positions is not None
                  else self.current_joint_positions.copy())

        for action in actions_raw:
            pose_7d = action[:7]
            hand = action[7:13]

            if not self._validate_cartesian(pose_7d):
                break

            joints = self._cartesian_to_joints(pose_7d, ik_ref)
            if joints is None:
                continue

            ik_ref = joints
            processed_actions.append(np.concatenate([joints, hand]))

        if not processed_actions:
            return

        with self._queue_lock:
            old_queue = list(self._action_queue)
            self._action_queue.clear()

            if not old_queue:
                curr_arm = (self.last_commanded_positions.copy()
                            if self.last_commanded_positions is not None
                            else self.current_joint_positions.copy())
                curr_hand = (self.current_hand_goal
                             if self.current_hand_goal is not None
                             else np.zeros(6))

                stationary_point = np.concatenate([curr_arm, curr_hand])
                old_queue = [stationary_point] * len(processed_actions)

            for i, new_act in enumerate(processed_actions):
                if i < len(old_queue):
                    blend_horizon = 15.0
                    w = max(0.0, 1.0 - (i / blend_horizon))
                    blended_act = (w * old_queue[i]) + ((1.0 - w) * new_act)
                    self._action_queue.append(blended_act)
                else:
                    self._action_queue.append(new_act)

    # -------------------------------------------------------------------------
    # Control Loops
    # -------------------------------------------------------------------------

    def _queue_pop_loop(self):
        with self._queue_lock:
            if not self._action_queue:
                return
            action = self._action_queue.popleft()

        self.current_arm_goal = action[:6]
        self.current_hand_goal = action[6:13]

    def _motor_control_loop(self):
        if not self.policy_active:
            return

        if self.current_arm_goal is None:
            return

        if self.last_commanded_positions is None:
            self.last_commanded_positions = self.current_joint_positions.copy()

        alpha = 0.02
        safe_target = (alpha * self.current_arm_goal) + \
                      ((1.0 - alpha) * self.last_commanded_positions)

        self.last_commanded_positions = safe_target

        cmd = Float64MultiArray(data=safe_target.tolist())
        self.arm_cmd_pub.publish(cmd)

        if self.current_hand_goal is not None:
            self._send_hand(self.current_hand_goal)

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def _validate_cartesian(self, pose_7d):
        x, y, z = pose_7d[0], pose_7d[1], pose_7d[2]
        if not (workspace['x'][0] <= x <= workspace['x'][1] and
                workspace['y'][0] <= y <= workspace['y'][1] and
                workspace['z'][0] <= z <= workspace['z'][1]):
            return False

        quat_norm = np.linalg.norm(pose_7d[3:7])
        return 0.99 <= quat_norm <= 1.01

    def _cartesian_to_joints(self, pose_7d, ref_joints):
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
            for _ in range(3):
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

            if np.max(np.abs(joints - ref_joints)) > 0.3:
                self.get_logger().debug("IK Flip avoided.")
                return None

            return joints

        except Exception as e:
            self.get_logger().error(f"IK Solver failed: {e}")
            return None

    def _send_hand(self, hand_joints):
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        reversed_joints = hand_joints[::-1]
        msg.angles = [int((1 - x) * 1000) for x in reversed_joints]
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