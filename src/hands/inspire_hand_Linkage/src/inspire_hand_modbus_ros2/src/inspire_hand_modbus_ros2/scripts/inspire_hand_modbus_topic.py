#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
# Message Imports
from service_interfaces.msg import GetAngleAct1, SetAngle1, SetSpeed1
from sensor_msgs.msg import JointState

# --- CONFIGURATION ---
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000

# Using the mapping from the WORKING script
# (Finger ID, Register Address, Name, Max Radians)
JOINT_MAPPING = [
    (1, 1546, 'inspire_right_little_1_joint', 1.4381),
    (2, 1548, 'inspire_right_ring_1_joint',   1.4381),
    (3, 1550, 'inspire_right_middle_1_joint', 1.4381),
    (4, 1552, 'inspire_right_index_1_joint',  1.4381),
    (5, 1554, 'inspire_right_thumb_2_joint',  0.5864),
    (6, 1556, 'inspire_right_thumb_1_joint',  1.1641),
]

ANGLE_SET_RANGES = {1: 1486, 2: 1488, 3: 1490, 4: 1492, 5: 1494, 6: 1496}
SPEED_SET_RANGES = {1: 1522, 2: 1524, 3: 1526, 4: 1528, 5: 1530, 6: 1532}

class ModbusNode(Node):
    def __init__(self):
        super().__init__('inspire_hand_driver')

        # --- PUBLISHERS ---
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.angle_publisher = self.create_publisher(GetAngleAct1, 'angle_data', 10)

        # --- SUBSCRIBERS ---
        self.create_subscription(SetAngle1, 'set_angle_data', self.angle_set_callback, 10)
        self.create_subscription(SetSpeed1, 'set_speed_data', self.speed_set_callback, 10)

        # --- MODBUS CONNECTION ---
        self.modbus_client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT, timeout=0.1)
        self.connect_modbus()

        self.get_logger().info("Inspire Hand Driver Started.")

        # --- TIMER ---
        # Increased to 40ms (25Hz) for better stability
        self.timer = self.create_timer(0.04, self.timer_callback)

    def connect_modbus(self):
        if not self.modbus_client.connect():
            self.get_logger().error(f"Failed to connect to Modbus at {MODBUS_IP}:{MODBUS_PORT}")
            return False
        return True

    def read_signed_register(self, address):
        """Reads a single register to ensure stability (logic from working script)."""
        try:
            response = self.modbus_client.read_holding_registers(address=address, count=1)
            
            if response.isError():
                self.get_logger().debug(f"Read error at {address}")
                return None
            
            val = response.registers[0]
            if val > 32767:
                val -= 65536
            return val
            
        except Exception as e:
            self.get_logger().debug(f"Exception reading {address}: {e}")
            return None

    def write_signed_register(self, address, value):
        if not self.modbus_client.connected: return
        
        value = int(value)
        if value < 0: value += 65536
            
        try:
            self.modbus_client.write_register(address, value)
        except Exception as e:
            self.get_logger().error(f"Write error: {e}")

    def timer_callback(self):
        if not self.modbus_client.connected:
            if not self.connect_modbus(): return

        # --- Message Prep ---
        current_time = self.get_clock().now().to_msg()
        
        angle_msg = GetAngleAct1()
        angle_msg.finger_ids = []
        angle_msg.angles = []
        angle_msg.finger_names = []

        js_msg = JointState()
        js_msg.header.stamp = current_time
        js_msg.header.frame_id = "base_link"
        js_msg.name = []
        js_msg.position = []
        
        # --- Sequential Read Loop (Fixes "Weird Values") ---
        for finger_id, address, name, max_rad in JOINT_MAPPING:
            
            # Read one by one
            raw_val = self.read_signed_register(address)
            
            if raw_val is None:
                # If one fails, we skip it to prevent publishing bad data
                continue

            angle_msg.finger_ids.append(finger_id)
            angle_msg.angles.append(raw_val)
            angle_msg.finger_names.append(name)

            # Calculation
            rads = ((1000.0 - float(raw_val)) / 1000.0) * max_rad
            
            js_msg.name.append(name)
            js_msg.position.append(rads)

        # Only publish if we got data
        if angle_msg.finger_ids:
            self.angle_publisher.publish(angle_msg)
            self.joint_state_pub.publish(js_msg)

    # --- CALLBACKS ---
    def angle_set_callback(self, msg):
        for finger_id, angle in zip(msg.finger_ids, msg.angles):
            addr = ANGLE_SET_RANGES.get(finger_id)
            if addr: self.write_signed_register(addr, angle)

    def speed_set_callback(self, msg):
        for finger_id, speed in zip(msg.finger_ids, msg.speeds):
            addr = SPEED_SET_RANGES.get(finger_id)
            if addr: self.write_signed_register(addr, speed)

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