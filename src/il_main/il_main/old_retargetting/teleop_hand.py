#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy
from dex_retargeting.retargeting_config import RetargetingConfig
import numpy as np

# Correct Pinocchio import
import pinocchio as pin

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

        # Target Robot Config
        config_path = "/ros2_ws/src/il_main/il/retarget_correct.yml"
        self.retargeting = RetargetingConfig.load_from_file(config_path).build()

        # Human Glove Config
        # Pinocchio requires a compiled .urdf file. A .xacro file will fail to parse.
        human_urdf_path = "/ros2_ws/src/il_main/third_party/senseglove/senseglove_description/urdf/nova2/nova2_right.urdf" 
        
        self.human_model = pin.buildModelFromUrdf(human_urdf_path)
        self.human_data = self.human_model.createData()

    def compute_fk(self, msg: JointState, frame_name: str) -> np.ndarray:
        # Map the arbitrary JointState message to Pinocchio's strict q vector
        q = np.zeros(self.human_model.nq)
        
        for name, position in zip(msg.name, msg.position):
            if self.human_model.existJointName(name):
                joint_id = self.human_model.getJointId(name)
                idx_q = self.human_model.joints[joint_id].idx_q
                
                # Ensure it is a valid joint with degrees of freedom
                if idx_q >= 0:
                    q[idx_q] = position

        # Compute forward kinematics and update the placements of all operational frames
        pin.framesForwardKinematics(self.human_model, self.human_data, q)

        # Extract the 3D translation vector for the requested link
        if self.human_model.existFrame(frame_name):
            frame_id = self.human_model.getFrameId(frame_name)
            return self.human_data.oMf[frame_id].translation
        else:
            self.get_logger().error(f"Frame '{frame_name}' does not exist in the URDF.")
            return np.zeros(3)

    def retarget_glove_to_hand(self, data_glove_joints: JointState):

        # Call compute_fk as a class method
        thumb_tip_pos = self.compute_fk(data_glove_joints, "r_thumb_flexion_three")
        index_tip_pos = self.compute_fk(data_glove_joints, "r_index_flexion_three")
        middle_tip_pos = self.compute_fk(data_glove_joints, "r_middle_flexion_three")
        ring_tip_pos = self.compute_fk(data_glove_joints, "r_ring_flexion_three")
        pinky_tip_pos = self.compute_fk(data_glove_joints, "r_pinky_flexion_three")
        
        human_keypoint_array = np.array([
            thumb_tip_pos,
            index_tip_pos,
            middle_tip_pos,
            ring_tip_pos,
            pinky_tip_pos
        ])
        
        artus_target_joint_angles = self.retargeting.retarget(human_keypoint_array)
        return artus_target_joint_angles

    def listener_callback(self, msg: JointState):
        if self.init:
            self.init = False

        new_angles = self.retarget_glove_to_hand(data_glove_joints=msg)
        
        self.get_logger().info(f"Retargeted Angles: {new_angles}")
        self.set_joint_goal_and_execute(new_angles)

    def set_joint_goal_and_execute(self, new_angles):  
        pass

def main(args=None):
    rclpy.init(args=args)
    teleoperation = None
    
    try:
        teleoperation = Teleoperation()
        rclpy.spin(teleoperation)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if 'Parameter already declared' not in str(e):
            print(f"Error during node execution: {e}")
    finally:
        if teleoperation is not None:
            teleoperation.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()