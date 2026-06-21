import rclpy
from rclpy.node import Node
from dg_msgs.msg import ReceivedFingertipSensorData, ReceivedGripperData
import numpy as np
import time


class TesolloMockNode(Node):
    def __init__(self):
        super().__init__('tesollo_mock_publisher')

        self.gripper_pub = self.create_publisher(
            ReceivedGripperData, '/tesollo/gripper_data', 10
        )
        self.sensor_pub = self.create_publisher(
            ReceivedFingertipSensorData, '/tesollo/fingertip_sensor', 10
        )

        # 250 Hz
        self.timer = self.create_timer(0.004, self.publish_mock_data)
        self.start_time = time.time()
        self.get_logger().info('Tesollo Mock started (250Hz)')

    def publish_mock_data(self):
        t = time.time() - self.start_time

        # --- Gripper Data ---
        gripper_msg = ReceivedGripperData()

        # Joint positions (20 DoF, degrees)
        gripper_msg.joint = [
            float(20.0 * np.sin(t + i * 0.1)) for i in range(20)
        ]

        # TCP: 5 fingers × 6 (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)
        tcp_data = []
        for finger in range(5):
            x = 10.0 + 5.0 * np.sin(t + finger)    # mm
            y = -5.0 + 3.0 * np.cos(t + finger)    # mm
            z = 60.0 + 10.0 * np.sin(t * 0.5)      # mm
            rx, ry, rz = 0.0, 0.0, 0.0             # degrees
            tcp_data.extend([x, y, z, rx, ry, rz])
        gripper_msg.tcp = [float(v) for v in tcp_data]

        gripper_msg.current = [0] * 20
        gripper_msg.velocity = [0] * 20
        gripper_msg.temperature = [25.0] * 20
        gripper_msg.moving = 1
        gripper_msg.target_arrived = 0
        gripper_msg.blend_move_state = 0
        gripper_msg.current_blend_index = 0
        gripper_msg.product_id = 1
        gripper_msg.firmware_version = 100

        self.gripper_pub.publish(gripper_msg)

        # --- Fingertip Sensor Data ---
        sensor_msg = ReceivedFingertipSensorData()

        ft_data = []
        for finger_idx in range(5):
            noise = np.random.normal(0, 0.01, 6)

            # Contact spike on index finger every 5 seconds
            contact = 0.0
            if finger_idx == 1 and (int(t) % 5 == 0):
                contact = 5.0 + np.sin(t * 10)

            ft_data.extend([
                noise[0],              # fx
                noise[1],              # fy
                noise[2] + contact,    # fz
                noise[3],              # tx
                noise[4],              # ty
                noise[5],              # tz
            ])

        sensor_msg.force_torque = [float(v) for v in ft_data]
        self.sensor_pub.publish(sensor_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TesolloMockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()