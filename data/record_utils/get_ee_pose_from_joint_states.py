#!/usr/bin/env python3
"""Compute EE pose from live ROS2 joint_states topic."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from data.dataset_builder.processing.proprioception import setup_kinematics, compute_ee_pose

URDF_PATH = "/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf"
END_LINK = "right_inspire_base_link"

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


class EEPoseCalculator(Node):
    def __init__(self):
        super().__init__('ee_pose_calculator')

        self.chain = setup_kinematics(urdf_path=URDF_PATH, end_link=END_LINK)

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.get_logger().info("Waiting for /joint_states messages...")

    def joint_states_callback(self, msg: JointState):
        """Extract joint positions in correct order and compute EE pose."""

        # Build a name → position mapping from the ROS message
        name_to_position = dict(zip(msg.name, msg.position))

        # Check all required joints are present
        missing = [j for j in JOINT_NAMES if j not in name_to_position]
        if missing:
            self.get_logger().warn(f"Missing joints in message: {missing}")
            return

        # Extract positions in the correct order
        joint_states = [name_to_position[name] for name in JOINT_NAMES]

        # Convert to 2D NumPy array (shape: 1 x 6)
        joint_states_np = np.array([joint_states])

        # Compute EE pose (shape: 1 x 7)
        ee_pose = compute_ee_pose(JOINT_NAMES, joint_states_np, self.chain)

        z_point = ee_pose[0, 2]
        self.get_logger().info(
            f"Joint states: {[f'{v:.4f}' for v in joint_states]}\n"
            f"  EE pose: {ee_pose[0]}\n"
            f"  Z point: {z_point:.6f}"
        )


def main():
    rclpy.init()
    node = EEPoseCalculator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()