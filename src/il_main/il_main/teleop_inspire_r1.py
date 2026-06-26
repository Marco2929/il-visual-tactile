#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from r1_msgs.msg import R1GloveState
from rclpy.qos import QoSProfile, ReliabilityPolicy
from service_interfaces.msg import SetAngle1

hand_ranges = [
    {'thumb_1': {'lower': 0, 'upper': 1000}},
    {'thumb_2': {'lower': 0, 'upper': 1000}},
    {'index': {'lower': 0, 'upper': 1000}},
    {'middle': {'lower': 0, 'upper': 1000}},
    {'ring': {'lower': 0, 'upper': 1000}},
    {'pinky': {'lower': 0, 'upper': 1000}}
]

hand_ranges_map = {list(item.keys())[0]: list(item.values())[0] for item in hand_ranges}


def remap_value(normalized_value, target_range):
    """Scales a normalized [0, 1] value to the robot's target joint range."""
    return normalized_value * (target_range['upper'] - target_range['lower']) + target_range['lower']


def teleoperation_mapping(normalized_positions):
    """
    Maps 10-element normalized glove data (0-10000) to robot joint ranges.
    Indices 0-4: Flexion (Thumb to Pinky)
    Indices 5-9: Abduction (Thumb to Pinky)
    """
    hand_mapping = []

    def get_norm(val):
        return max(0.0, min(1.0, val / 10000.0))

    # Thumb mapping
    thumb1_norm = 1.0 - get_norm(normalized_positions[5])
    hand_mapping.append({"thumb_1": remap_value(thumb1_norm, hand_ranges_map["thumb_1"])})

    thumb2_norm = 1.0 - get_norm(normalized_positions[0])
    hand_mapping.append({"thumb_2": remap_value(thumb2_norm, hand_ranges_map["thumb_2"])})

    # Remaining fingers mapping (Indices 1-4 for flexion)
    finger_names = ["index", "middle", "ring", "pinky"]

    for i, name in enumerate(finger_names):
        flex_norm = 1.0 - get_norm(normalized_positions[i + 1])
        hand_mapping.append({name: remap_value(flex_norm, hand_ranges_map[name])})

    return hand_mapping


class Teleoperation(Node):
    def __init__(self):
        super().__init__('teleoperation')

        self.enabled = False

        # --- COMMAND SUBSCRIBER ---
        self.sub_enable = self.create_subscription(
            Bool, '/teleop_enable', self.enable_cb, 10
        )

        # --- GLOVE SUBSCRIBER ---
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.subscription = self.create_subscription(
            R1GloveState,
            '/r1/glove70/rh/glove_states',
            self.listener_callback,
            qos_profile
        )

        # --- PUBLISHER ---
        self.publisher_ = self.create_publisher(SetAngle1, '/set_angle_data', 10)

        self.get_logger().info('Teleoperation node started. Waiting for /teleop_enable...')

    def enable_cb(self, msg: Bool):
        self.enabled = msg.data
        state = "ENABLED" if self.enabled else "DISABLED"
        self.get_logger().info(f'Teleoperation {state}')

    def listener_callback(self, msg: R1GloveState):
        if not self.enabled:
            return

        norm_positions = msg.normalized_finger_positions

        if len(norm_positions) < 10:
            self.get_logger().warn("Expected 10 values in normalized_finger_positions.")
            return

        new_angles = teleoperation_mapping(norm_positions)

        # Extract and reverse sequence to match robot execution order
        final_angles = [int(list(d.values())[0]) for d in reversed(new_angles)]

        self.set_joint_goal_and_execute(final_angles)

    def set_joint_goal_and_execute(self, new_angles):
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        new_angles[-1] = 0
        msg.angles = new_angles

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        teleoperation = Teleoperation()
        rclpy.spin(teleoperation)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if 'Parameter already declared' not in str(e):
            print(f"{e}")
    finally:
        if rclpy.ok():
            teleoperation.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()