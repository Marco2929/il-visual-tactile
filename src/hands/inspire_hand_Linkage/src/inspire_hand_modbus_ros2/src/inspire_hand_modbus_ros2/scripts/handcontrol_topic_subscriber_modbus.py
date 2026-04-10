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
READ_START_ADDR = 1546
READ_COUNT = 11 

# STANDARD UNIT ID FOR INSPIRE HAND
# ID 0 is broadcast (pymodbus ignores replies to 0). 
# ID 1 is the default for most Inspire hands.
SLAVE_ID = 1

JOINT_MAPPING = [
    (0,  1, 'inspire_right_little_1_joint', 1.4381),
    (2,  2, 'inspire_right_ring_1_joint',   1.4381),
    (4,  3, 'inspire_right_middle_1_joint', 1.4381),
    (6,  4, 'inspire_right_index_1_joint',  1.4381),
    (8,  5, 'inspire_right_thumb_2_joint',  0.5864),
    (10, 6, 'inspire_right_thumb_1_joint',  1.1641),
]

ANGLE_SET_RANGES = {1: 1486, 2: 1488, 3: 1490, 4: 1492, 5: 1494, 6: 1496}
SPEED_SET_RANGES = {1: 1522, 2: 1524, 3: 1526, 4: 1528, 5: 1530, 6: 1532}

class ModbusNode(Node):
    def __init__(self):
        super().__init__('sensor_data_publisher')

        # --- PUBLISHERS ---
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.angle_publisher = self.create_publisher(GetAngleAct1, 'angle_data', 10)

        # --- SUBSCRIBERS ---
        self.create_subscription(SetAngle1, 'set_angle_data', self.angle_set_callback, 10)
        self.create_subscription(SetSpeed1, 'set_speed_data', self.speed_set_callback, 10)

        # --- MODBUS CONNECTION ---
        # Critical: Set a timeout. If 50Hz is target (20ms), timeout must be short.
        # We set 0.1s to allow slight jitter but fail fast if device hangs.
        self.modbus_client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT, timeout=0.1)
        self.connect_modbus()

        self.get_logger().info("Modbus Node Started. 50Hz Timer active.")

        # --- TIMER (Replaces Thread) ---
        # ROS2 timers handle scheduling better than raw Python threads
        self.timer = self.create_timer(0.02, self.timer_callback)

    def connect_modbus(self):
        """Attempts to connect to Modbus. Returns True if successful."""
        if not self.modbus_client.connect():
            self.get_logger().error(f"Failed to connect to Modbus at {MODBUS_IP}:{MODBUS_PORT}")
            return False
        return True

    def to_signed_16bit(self, value):
        if value > 32767:
            return value - 65536
        return value

    def write_signed_register(self, address, value):
        if not self.modbus_client.connected:
            return
        
        value = int(value)
        if value < 0:
            value += 65536
            
        try:
            # write_register is synchronous. If it fails, it raises ModbusException
            # FIX: Added slave=SLAVE_ID to prevent broadcast (ID 0)
            self.modbus_client.write_register(address, value, slave=SLAVE_ID)
        except (ModbusException, OSError) as e:
            self.get_logger().error(f"Write error: {e}")

    def timer_callback(self):
        """Replaces the threaded while loop."""
        if not self.modbus_client.connected:
            # Attempt reconnect periodically, not in a tight loop
            if not self.connect_modbus():
                return

        try:
            # Synchronous read with the timeout defined in __init__
            # FIX: Added slave=SLAVE_ID. Without this, it sends ID 0, receives ID 0, and pymodbus drops it.
            response = self.modbus_client.read_holding_registers(address=READ_START_ADDR, count=READ_COUNT, slave=SLAVE_ID)
            
            if response.isError():
                self.get_logger().warn(f"Modbus read error: {response}")
                return

            registers = response.registers
            
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
            
            for block_idx, finger_id, name, max_rad in JOINT_MAPPING:
                # Safety check for index out of range if partial packet received
                if block_idx >= len(registers):
                    continue

                raw_val = self.to_signed_16bit(registers[block_idx])
                
                angle_msg.finger_ids.append(finger_id)
                angle_msg.angles.append(raw_val)
                angle_msg.finger_names.append(name)

                # Calculation
                rads = ((1000.0 - float(raw_val)) / 1000.0) * max_rad
                
                js_msg.name.append(name)
                js_msg.position.append(rads)

            self.angle_publisher.publish(angle_msg)
            # self.joint_state_pub.publish(js_msg)

        except (ModbusException, OSError) as e:
            self.get_logger().error(f"Read Loop Exception: {e}")
            # Do not immediately close/open. Just mark logged. 
            # Pymodbus handles socket flags internally. 
            # Only force close if persistent failures occur.
            if isinstance(e, OSError):
                 self.modbus_client.close()

    # --- CALLBACKS ---
    def angle_set_callback(self, msg):
        # Callbacks interrupt the main loop (SingleThreadedExecutor)
        # We must ensure this doesn't block for long.
        if not self.modbus_client.connected: return

        for finger_id, angle in zip(msg.finger_ids, msg.angles):
            addr = ANGLE_SET_RANGES.get(finger_id)
            if addr: 
                self.write_signed_register(addr, angle)

    def speed_set_callback(self, msg):
        if not self.modbus_client.connected: return
        
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
    # Clean shutdown
    if node.modbus_client.connected:
        node.modbus_client.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()