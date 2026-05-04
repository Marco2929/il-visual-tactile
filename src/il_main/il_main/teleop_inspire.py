#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from service_interfaces.msg import SetAngle1
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy
from il_main.mapping import teleoperation_mapping

class Teleoperation(Node):
    def __init__(self):
        super().__init__('teleoperation')

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.init = True

        self.subscription = self.create_subscription(
        JointState,
        '/senseglove/glove0/rh/joint_states',
        self.listener_callback,
        qos_profile
        )

        self.publisher_ = self.create_publisher(SetAngle1, '/set_angle_data', 10)
                

    def map_glove_joints(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))

        return [
            {'thumb': {
                'mcp': name_to_pos.get('r_thumb_mcp', 0.0),
                'pip': name_to_pos.get('r_thumb_pip', 0.0),
                'dip': name_to_pos.get('r_thumb_dip', 0.0),
            }},
            {'index': {
                'mcp': name_to_pos.get('r_index_mcp', 0.0),
                'pip': name_to_pos.get('r_index_pip', 0.0),
                'dip': name_to_pos.get('r_index_dip', 0.0),
            }},
            {'middle': {
                'mcp': name_to_pos.get('r_middle_mcp', 0.0),
                'pip': name_to_pos.get('r_middle_pip', 0.0),
                'dip': name_to_pos.get('r_middle_dip', 0.0),
            }},
            {'ring': {
                'mcp': name_to_pos.get('r_ring_mcp', 0.0),
                'pip': name_to_pos.get('r_ring_pip', 0.0),
                'dip': name_to_pos.get('r_ring_dip', 0.0),
            }},
            {'pinky': {
                'mcp': name_to_pos.get('r_pinky_mcp', 0.0),
                'pip': name_to_pos.get('r_pinky_pip', 0.0),
                'dip': name_to_pos.get('r_pinky_dip', 0.0),
            }},
        ]
    

    def listener_callback(self, msg: JointState):
        if self.init:
            self.open_hand()
            self.init = False

        data_glove_joints = self.map_glove_joints(msg)

        new_angles = teleoperation_mapping(data_glove_joints=data_glove_joints)

        final_angles = [int(list(d.values())[0]) for d in reversed(new_angles)]

        self.set_joint_goal_and_execute(final_angles)
    
    def open_hand(self):
        self.set_joint_goal_and_execute([1000] * 6)


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
