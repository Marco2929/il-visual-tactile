#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import ServoCommandType
from tf2_ros import TransformException
from rclpy.qos import qos_profile_sensor_data
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from service_interfaces.msg import SetAngle1
import numpy as np
import threading
from trajectory_msgs.msg import JointTrajectory

class InferenceControlNode(Node):
    def __init__(self):
        super().__init__('inference_control_node')

        # --- Config ---
        self.robot_base_frame = 'base_link'
        self.robot_tip_frame = 'inspire_base_link'

        # --- Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        
        self.pub = self.create_publisher(PoseStamped, '/servo_node/pose_target_cmds', 10)
        self.publisher_ = self.create_publisher(SetAngle1, '/set_angle_data', 10)

        self.switch_servo_mode()

        # --- ASYNC EXECUTION QUEUE ---
        self.action_queue = []
        self.queue_lock = threading.Lock()
        
        # Add this dynamic latency tracker
        self.steps_executed = 0

        # Execute actions strictly at 15Hz
        self.control_timer = self.create_timer(1/15, self.timer_callback)

        # Listen for the structured chunk from File 1
        self.chunk_sub = self.create_subscription(
            JointTrajectory, 
            '/predicted_action', 
            self.chunk_callback, 
            qos_profile_sensor_data
        )

        self.get_logger().info("Async Robot Controller Initialized")

    def switch_servo_mode(self):
        client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')
        if client.wait_for_service(timeout_sec=1.0):
            req = ServoCommandType.Request()
            req.command_type = 2
            client.call_async(req)
        else:
            self.get_logger().warn("Servo service not available, skipping switch.")

    def send_hand_joints(self, hand_joints):
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        # Reverse and scale joints based on your original logic
        reversed_joints = hand_joints[::-1]
        msg.angles = [int((1 - x) * 1000) for x in reversed_joints]
        self.publisher_.publish(msg)

    def validate_pose(self, pose_stamped):
        """validate if the target pose is within the workspace"""
        pos_limits = {
            'x': (0.0, 0.8),
            'y': (-0.8, 0.8),
            'z': (0.05, 1.0)
        } 
        
        pos = pose_stamped.pose.position
        orn = pose_stamped.pose.orientation

        if not (pos_limits['x'][0] <= pos.x <= pos_limits['x'][1] and
                pos_limits['y'][0] <= pos.y <= pos_limits['y'][1] and
                pos_limits['z'][0] <= pos.z <= pos_limits['z'][1]):
            self.get_logger().warn(f'Target position out of limits: [{pos.x}, {pos.y}, {pos.z}]')
            return False

        quat_norm = np.sqrt(orn.x**2 + orn.y**2 + orn.z**2 + orn.w**2)
        if not 0.99 <= quat_norm <= 1.01:
            self.get_logger().warn(f'Quaternion not normalized: {quat_norm}')
            return False

        return True

    def create_pose_message(self, array_data):
        """ Creates a PoseStamped from a simple float array [x,y,z,qx,qy,qz,qw] """
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = self.robot_base_frame
        
        target_pose.pose.position.x = float(array_data[0])
        target_pose.pose.position.y = float(array_data[1])
        target_pose.pose.position.z = float(array_data[2])

        target_pose.pose.orientation.x = float(array_data[3])
        target_pose.pose.orientation.y = float(array_data[4])
        target_pose.pose.orientation.z = float(array_data[5])
        target_pose.pose.orientation.w = float(array_data[6])

        return target_pose

    def chunk_callback(self, msg: JointTrajectory):
        new_trajectory = []

        for point in msg.points:
            cartesian_data = point.positions[0:7]
            hand_data = point.positions[7:13]
            target_pose = self.create_pose_message(cartesian_data)
            new_trajectory.append({'pose': target_pose, 'hands': hand_data})

        with self.queue_lock:
            if len(self.action_queue) > 0:
                current_target = self.action_queue[0]['pose'].pose.position
                
                # 1. WINDOWED SPATIAL ALIGNMENT (The version that looked better)
                min_dist = float('inf')
                
                # We know the GPU takes ~0.4s (about 6 steps).
                # The true overlap MUST be in the middle of the new chunk.
                # We start searching at index 3 to prevent "Time-Travel" (pulling backward/up).
                # We stop searching at index 9 to prevent "Shortcutting" to the end of the trajectory.
                search_start = min(3, len(new_trajectory) - 1)
                search_end = min(9, len(new_trajectory))
                
                best_idx = search_start # Fallback
                
                for i in range(search_start, search_end):
                    step = new_trajectory[i]
                    new_pos = step['pose'].pose.position
                    dist = (current_target.x - new_pos.x)**2 + \
                           (current_target.y - new_pos.y)**2 + \
                           (current_target.z - new_pos.z)**2
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                
                new_future = new_trajectory[best_idx:]
                
                # 2. MOMENTUM FADING (Smooths out the AI noise)
                overlap_length = min(len(self.action_queue), len(new_future))
                
                for i in range(overlap_length):
                    old_step = self.action_queue[i]
                    new_step = new_future[i]
                    
                    if overlap_length > 1:
                        w_new = i / (overlap_length - 1)
                    else:
                        w_new = 1.0
                        
                    w_old = 1.0 - w_new

                    # Apply fade to XYZ
                    new_future[i]['pose'].pose.position.x = (old_step['pose'].pose.position.x * w_old) + (new_step['pose'].pose.position.x * w_new)
                    new_future[i]['pose'].pose.position.y = (old_step['pose'].pose.position.y * w_old) + (new_step['pose'].pose.position.y * w_new)
                    new_future[i]['pose'].pose.position.z = (old_step['pose'].pose.position.z * w_old) + (new_step['pose'].pose.position.z * w_new)
                    
                    # Apply fade to Orientations
                    old_orn = old_step['pose'].pose.orientation
                    new_orn = new_future[i]['pose'].pose.orientation
                    
                    qx = (old_orn.x * w_old) + (new_orn.x * w_new)
                    qy = (old_orn.y * w_old) + (new_orn.y * w_new)
                    qz = (old_orn.z * w_old) + (new_orn.z * w_new)
                    qw = (old_orn.w * w_old) + (new_orn.w * w_new)
                    
                    mag = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
                    new_future[i]['pose'].pose.orientation.x = qx / mag
                    new_future[i]['pose'].pose.orientation.y = qy / mag
                    new_future[i]['pose'].pose.orientation.z = qz / mag
                    new_future[i]['pose'].pose.orientation.w = qw / mag
                    
                    # Apply fade to Hands
                    for j in range(6):
                        new_future[i]['hands'][j] = (old_step['hands'][j] * w_old) + (new_step['hands'][j] * w_new)

                self.action_queue = new_future
                self.get_logger().info(f"Spatially aligned at index {best_idx}. Blended {overlap_length} steps.")
            else:
                self.action_queue = new_trajectory
                self.get_logger().info("Queue empty. Accepted full 16-step chunk.")


    def timer_callback(self):
        """ Strictly executes the next available action at 15Hz """
        target_pose = None
        hand_joints = None

        with self.queue_lock:
            if len(self.action_queue) > 0:
                action_data = self.action_queue.pop(0)
                target_pose = action_data['pose']
                hand_joints = action_data['hands']
                
                # Removed self.steps_executed += 1 here
                
        # If queue was empty, skip execution this tick
        if target_pose is not None:
            if not self.validate_pose(target_pose):
                self.get_logger().error("Pose validation FAILED. Dropping action.")
                return
            
            # Publish commands to hardware
            self.pub.publish(target_pose)
            self.send_hand_joints(hand_joints)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceControlNode()
    try: 
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit): 
        pass
    finally:
        if rclpy.ok(): 
            rclpy.shutdown()

if __name__ == '__main__':
    main()