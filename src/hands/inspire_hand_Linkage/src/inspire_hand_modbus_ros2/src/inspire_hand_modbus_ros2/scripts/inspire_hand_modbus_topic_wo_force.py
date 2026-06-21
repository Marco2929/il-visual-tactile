#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import rclpy
from rclpy.node import Node
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
from service_interfaces.msg import GetAngleAct1, SetAngle1, SetSpeed1
from sensor_msgs.msg import JointState

# --- CONFIGURATION ---
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000

# (Index in block read, Finger ID, Name, Max Radians)
JOINT_MAPPING = [
    (0, 1, 'right_inspire_right_little_1_joint', 1.4381),
    (1, 2, 'right_inspire_right_ring_1_joint',   1.4381),
    (2, 3, 'right_inspire_right_middle_1_joint', 1.4381),
    (3, 4, 'right_inspire_right_index_1_joint',  1.4381),
    (4, 5, 'right_inspire_right_thumb_2_joint',  0.5864),
    (5, 6, 'right_inspire_right_thumb_1_joint',  1.1641),
]

# Write addresses (byte-addressed, single writes work with these)
ANGLE_SET_RANGES = {1: 1486, 2: 1488, 3: 1490, 4: 1492, 5: 1494, 6: 1496}
SPEED_SET_RANGES = {1: 1522, 2: 1524, 3: 1526, 4: 1528, 5: 1530, 6: 1532}


class ModbusNode(Node):
    def __init__(self):
        super().__init__('inspire_hand_driver')

        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.angle_publisher = self.create_publisher(GetAngleAct1, 'angle_data', 10)

        self.create_subscription(SetAngle1, 'set_angle_data', self.angle_set_callback, 10)
        self.create_subscription(SetSpeed1, 'set_speed_data', self.speed_set_callback, 10)

        self.modbus_client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT, timeout=0.05)
        self.connect_modbus()

        self.modbus_lock = threading.Lock()
        self.get_logger().info("Inspire Hand Driver Started (Block Read Fixed).")

        self.timer = self.create_timer(0.0166, self.timer_callback)

    def connect_modbus(self):
        if not self.modbus_client.connect():
            self.get_logger().error(f"Failed to connect to Modbus at {MODBUS_IP}:{MODBUS_PORT}")
            return False
        return True

    def write_signed_register(self, address, value):
        if not self.modbus_client.connected:
            return
        value = int(value)
        if value < 0:
            value += 65536
        try:
            self.modbus_client.write_register(address, value)
        except Exception as e:
            self.get_logger().error(f"Write error: {e}")

    def timer_callback(self):
        if not self.modbus_client.connected:
            if not self.connect_modbus():
                return

        # Single block read: 6 registers = all 6 DOFs
        with self.modbus_lock:
            try:
                response = self.modbus_client.read_holding_registers(address=1546, count=6)
                if response.isError():
                    return
            except Exception as e:
                self.get_logger().debug(f"Block read exception: {e}")
                return

        current_time = self.get_clock().now().to_msg()

        angle_msg = GetAngleAct1()
        angle_msg.finger_ids = []
        angle_msg.angles = []
        angle_msg.finger_names = []

        js_names = []
        js_positions = []

        for index, finger_id, name, max_rad in JOINT_MAPPING:
            raw_val = response.registers[index]

            if raw_val > 32767:
                raw_val -= 65536

            angle_msg.finger_ids.append(finger_id)
            angle_msg.angles.append(raw_val)
            angle_msg.finger_names.append(name)

            rads = ((1000.0 - float(raw_val)) / 1000.0) * max_rad
            js_names.append(name)
            js_positions.append(rads)

        js_msg = JointState()
        js_msg.header.stamp = current_time
        js_msg.header.frame_id = "base_link"
        js_msg.name = js_names
        js_msg.position = js_positions

        if angle_msg.finger_ids:
            self.angle_publisher.publish(angle_msg)
            self.joint_state_pub.publish(js_msg)

    def angle_set_callback(self, msg):
        with self.modbus_lock:
            for finger_id, angle in zip(msg.finger_ids, msg.angles):
                addr = ANGLE_SET_RANGES.get(finger_id)
                if addr:
                    self.write_signed_register(addr, angle)

    def speed_set_callback(self, msg):
        with self.modbus_lock:
            for finger_id, speed in zip(msg.finger_ids, msg.speeds):
                addr = SPEED_SET_RANGES.get(finger_id)
                if addr:
                    self.write_signed_register(addr, speed)


def main(args=None):
    rclpy.init(args=args)
    node = ModbusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.modbus_client.connected:
        node.modbus_client.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()