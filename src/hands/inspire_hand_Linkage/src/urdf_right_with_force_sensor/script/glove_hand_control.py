#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# --- CRITICAL IMPORTS ---
from service_interfaces.msg import GetAngleAct1  # The Driver's language
from sensor_msgs.msg import JointState           # MoveIt's language

class InspireStatePublisher(Node):
    def __init__(self):
        super().__init__('inspire_hand_state_publisher')

        # NOTE: Ensure this order matches the order inside the 'angles' array from the driver!
        self.joint_names = [
            'inspire_right_little_1_joint',
            'inspire_right_ring_1_joint',
            'inspire_right_middle_1_joint',
            'inspire_right_index_1_joint',
            'inspire_right_thumb_2_joint',
            'inspire_right_thumb_1_joint',
        ]

        self.max_radians = {
            'inspire_right_thumb_1_joint': 1.1641,
            'inspire_right_thumb_2_joint': 0.5864,
            'inspire_right_index_1_joint': 1.4381,
            'inspire_right_middle_1_joint': 1.4381,
            'inspire_right_ring_1_joint': 1.4381,
            'inspire_right_little_1_joint': 1.4381,
        }

        # Publisher to MoveIt
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Subscriber to the Driver (Corrected Type)
        self.sub = self.create_subscription(
            GetAngleAct1, 
            '/angle_data', 
            self.angle_data_callback, 
            10)
        
        self.get_logger().info('Inspire Hand State Publisher started (Listening for GetAngleAct1)...')

    def angle_data_callback(self, msg):
        # 1. Extract the data from the specific field 'angles' found in your interface show
        raw_values = msg.angles 

        # 2. Safety check
        if len(raw_values) < 6:
            self.get_logger().warn(
                f'Insufficient data received, expected 6 values, got {len(raw_values)}')
            return

        positions = []
        try:
            # 3. Iterate and Convert
            for name, val in zip(self.joint_names, raw_values):
                # Conversion: 0-1000 (Int) -> 0.0 - MaxRad (Float)
                # Note: val comes in as int32, must cast to float
                rad = (float(val) / 1000.0) * self.max_radians[name]
                positions.append(rad)
        
        except KeyError as e:
            self.get_logger().error(f'Unknown joint name in max_radians: {e}')
            return
        except Exception as e:
            self.get_logger().error(f'Error during joint value conversion: {e}')
            return

        # 4. Publish Standard JointState
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.header.frame_id = "base_link" # Optional, helps suppress warnings
        js_msg.name = self.joint_names
        js_msg.position = positions
        
        # Empty arrays for velocity/effort to be clean
        js_msg.velocity = []
        js_msg.effort = []
        
        self.pub.publish(js_msg)


def main(args=None):
    rclpy.init(args=args)
    node = InspireStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()