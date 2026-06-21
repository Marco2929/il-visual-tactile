#!/usr/bin/env python3
import rclpy
import os
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, PointStamped, TwistStamped
from moveit_msgs.srv import ServoCommandType
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import sys, termios, tty, threading
import numpy as np
from scipy.spatial.transform import Rotation as R

class ViveFollowNode(Node):
    def __init__(self):
        super().__init__('vive_follow_node')

        # --- Config ---
        self.target_frame = 'LHB-12898A07' 
        self.source_frame = 'LHR-9F6576C3' # virtual_palm_link
        self.robot_base_frame = 'base_link'
        self.robot_tip_frame = os.environ.get("ROBOT_TIP_FRAME", "right_inspire_base_link")     
        self.scale_pos = 1.0

        self.deadband = 0.002

        # --- AXIS FIX MATRIX (Vive -> ROS) ---
        # Row 0 (Robot X) = From Tracker -Z
        # Row 1 (Robot Y) = From Tracker -X
        # Row 2 (Robot Z) = From Tracker +Y
        # self.R_fix = np.array([
        #     # X  Y  Z  W
        #     [ 0,  -1,  0,  0], 
        #     [ 0,  0,  1,  0], 
        #     [-1,  0,  0,  0],  
        #     [ 0,  0,  0,  1]
        # ])

        self.R_fix = np.array([
            [ 0,  1,  0,  0],
            [ 0,  0, -1,  0],
            [-1,  0,  0,  0],
            [ 0,  0,  0,  1]
        ])

        self.R_fix_inv = np.linalg.inv(self.R_fix)

        # --- State ---
        self.robot_enabled = False
        self.running = True
        self.T_tracker_ref = None  
        self.T_robot_ref = None    
        self.last_valid_pos = None
        self.last_valid_rot = None

        # --- Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        
        # Main Control Publisher (Now using TwistStamped for velocity control)
        self.pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        # --- DEBUG PUBLISHERS ---
        self.pub_dbg_raw = self.create_publisher(PoseStamped, '/debug/raw_tracker_delta', 10)
        self.pub_dbg_mapped = self.create_publisher(PoseStamped, '/debug/mapped_robot_delta', 10)
        self.pub_dbg_moved = self.create_publisher(PoseStamped, '/debug/mapped_robot_delta_moved', 10)
        self.pub_dbg_ref = self.create_publisher(PoseStamped, '/debug/robot_ref', 10)
        self.pub_dbg_noise = self.create_publisher(PointStamped, '/debug/noise_magnitude', 10)

        self.timer = self.create_timer(0.01, self.control_loop) # 100 Hz
        self.switch_servo_mode()
        
        self.settings = termios.tcgetattr(sys.stdin)
        self.input_thread = threading.Thread(target=self.keyboard_listener)
        self.input_thread.daemon = True
        self.input_thread.start()

        print("\n" + "="*40)
        print(" AXIS-CORRECTED TELEOP (TWIST MODE)")
        print(" 1. Hold Tracker aligned with Gripper")
        print(" 2. Press [SPACE] to Sync")
        print("="*40 + "\n")

    def switch_servo_mode(self):
        client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')
        if client.wait_for_service(timeout_sec=1.0):
            req = ServoCommandType.Request()
            req.command_type = 1 # 1 = TWIST (Velocity control mode)
            client.call_async(req)

    def keyboard_listener(self):
        while self.running:
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
            except Exception: break
            finally: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            if key == ' ': self.toggle_enable()
            elif key == 'q' or key == '\x03': 
                self.running = False
                rclpy.shutdown()
                sys.exit(0)

    def get_matrix(self, target, source):
        try:
            t = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=0))
            pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            rot = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            T = np.eye(4)
            T[:3, 3] = pos
            T[:3, :3] = R.from_quat(rot).as_matrix()
            return T, np.array(pos), R.from_quat(rot)
        except TransformException:
            return None, None, None

    def toggle_enable(self):
        if self.robot_enabled:
            self.robot_enabled = False
            print("\r[STOP]                                  ")
        else:
            T_t, pos_t, rot_t = self.get_matrix(self.target_frame, self.source_frame)
            T_r, pos_r, rot_r = self.get_matrix(self.robot_base_frame, self.robot_tip_frame)

            if T_t is not None and T_r is not None:
                self.T_tracker_ref = T_t
                self.T_robot_ref = T_r
                self.last_valid_pos = pos_t
                self.last_valid_rot = rot_t
                self.robot_enabled = True
                print(f"\r[GO]   Synced at {pos_t}", end="")

    def publish_debug_pose(self, publisher, T, frame_id):
        """Helper to publish a matrix as a PoseStamped for visualization."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        t_pos = T[:3, 3]
        t_quat = R.from_matrix(T[:3, :3]).as_quat()
        msg.pose.position.x = t_pos[0]
        msg.pose.position.y = t_pos[1]
        msg.pose.position.z = t_pos[2]
        msg.pose.orientation.x = t_quat[0]
        msg.pose.orientation.y = t_quat[1]
        msg.pose.orientation.z = t_quat[2]
        msg.pose.orientation.w = t_quat[3]
        publisher.publish(msg)

    def control_loop(self):
        if not self.running or not self.robot_enabled: return

        T_tracker_curr, curr_pos, curr_rot = self.get_matrix(self.target_frame, self.source_frame)
        if T_tracker_curr is None: return

        # Deadband check
        dist_sq = np.sum((curr_pos - self.last_valid_pos)**2)
        rot_mag = (curr_rot * self.last_valid_rot.inv()).magnitude()
        
        # Publish noise magnitude even if deadband is triggered
        noise_msg = PointStamped()
        noise_msg.header.stamp = self.get_clock().now().to_msg()
        noise_msg.header.frame_id = self.target_frame
        noise_msg.point.x = float(np.sqrt(dist_sq)) # Linear error
        noise_msg.point.y = float(rot_mag)          # Rotational error
        self.pub_dbg_noise.publish(noise_msg)

        if dist_sq < (self.deadband ** 2) and rot_mag < 0.0005: return 
        
        self.last_valid_pos = curr_pos
        self.last_valid_rot = curr_rot
        
        # 1. Calculate Raw Delta (In Tracker Frame)
        T_tracker_ref_inv = np.linalg.inv(self.T_tracker_ref)
        T_delta_raw = T_tracker_ref_inv @ T_tracker_curr

        self.publish_debug_pose(self.pub_dbg_raw, T_delta_raw, self.target_frame)

        # 2. Apply Scaling to Translation
        T_delta_raw[:3, 3] *= self.scale_pos

        # 3. Apply Axis Correction (Basis Change)
        # Math: T_corrected = R_fix * T_raw * R_fix_inv
        T_delta_ros = self.R_fix @ T_delta_raw @ self.R_fix_inv
        
        self.publish_debug_pose(self.pub_dbg_mapped, T_delta_ros, self.robot_base_frame)

        # 4. Calculate Target Pose
        T_target = self.T_robot_ref @ T_delta_ros

        self.publish_debug_pose(self.pub_dbg_ref, self.T_robot_ref, self.robot_base_frame)
        self.publish_debug_pose(self.pub_dbg_moved, T_target, self.robot_base_frame)

        # ==========================================
        # 5. CUSTOM P-CONTROLLER FOR TWIST COMMANDS
        # ==========================================
        
        # Get current robot pose
        T_curr, _, _ = self.get_matrix(self.robot_base_frame, self.robot_tip_frame)
        if T_curr is None: return

        # a) Linear error (Difference in meters in BASE frame)
        err_x = T_target[0, 3] - T_curr[0, 3]
        err_y = T_target[1, 3] - T_curr[1, 3]
        err_z = T_target[2, 3] - T_curr[2, 3]

        # b) Rotational error (Axis-angle in BASE frame)
        R_target = T_target[:3, :3]
        R_curr = T_curr[:3, :3]
        R_error = R_target @ R_curr.T 
        rot_vec_error_base = R.from_matrix(R_error).as_rotvec() 

        # c) Map rotational error to the LOCAL end-effector frame
        # R_curr.T transforms the vector from base_link into the tip frame
        rot_vec_error_local = R_curr.T @ rot_vec_error_base

        # d) Independent Proportional Gains
        Kp_lin = 0.5
        Kp_pitch_yaw = 0.5 # Keep your perfect value
        Kp_roll = 0.5    # Drastically boost ONLY the roll

        # Apply gains. 
        # NOTE: For UR10e tool flanges, local Z (index 2) is typically the roll axis. 
        # If your Inspire hand is mounted sideways, this might be X (0) or Y (1).
        twist_angular_local = np.array([
            rot_vec_error_local[0] * Kp_pitch_yaw,
            rot_vec_error_local[1] * Kp_pitch_yaw,
            rot_vec_error_local[2] * Kp_roll
        ])

        # e) Map the boosted twist back to the BASE frame for MoveIt Servo
        twist_angular_base = R_curr @ twist_angular_local

        # f) Create and publish Twist message
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_base_frame 

        msg.twist.linear.x = err_x * Kp_lin
        msg.twist.linear.y = err_y * Kp_lin
        msg.twist.linear.z = err_z * Kp_lin

        msg.twist.angular.x = twist_angular_base[0]
        msg.twist.angular.y = twist_angular_base[1]
        msg.twist.angular.z = twist_angular_base[2]

        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ViveFollowNode()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit): pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()