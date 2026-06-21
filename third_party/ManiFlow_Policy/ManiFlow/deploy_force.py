#!/usr/bin/env python3
import time
import threading
from queue import Queue

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, 
    ReliabilityPolicy, 
    HistoryPolicy, 
    qos_profile_sensor_data
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, String, Bool
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, SimpleFilter
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray

from model_serve import init_model, normalize
from cond_preprocess import PointCloudProcessor, setup_kinematics, compute_fk
from maniflow.common.palm_transform import PalmFrameTransform


class RobotCameraSync:
    """
    Event-driven synchronizer that bypasses message_filters.
    Syncs based on physical arrival sequence while preserving pristine hardware timestamps.
    """
    def __init__(self, node, callback_function, joint_names):
        self.node = node
        self.callback = callback_function
        self.joint_names = joint_names
        
        # --- Caches ---
        self.latest_joint_msg = None
        self.latest_color_msg = None
        self.latest_joint_state_dict = {name: None for name in self.joint_names}
        
        # --- Diagnostics ---
        self.cnt_joint_in = 0
        self.cnt_color_in = 0
        self.cnt_depth_in = 0
        self.cnt_synced_out = 0

        # --- QoS Definitions ---
        qos_fast = QoSProfile(
            depth=10, 
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST
        )
        qos_slow = QoSProfile(
            depth=5, 
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST
        )

        # TODO: Just included but not used yet!
        self.gripper_sub = self.create_subscription(
            ReceivedGripperData, '/tesollo/gripper_data',
            self._gripper_cb, 10
        )
        self.sensor_sub = self.create_subscription(
            ReceivedFingertipSensorData, '/tesollo/fingertip_sensor',
            self._sensor_cb, 10
        )

        # --- Subscribers ---
        self.sub_joint = node.create_subscription(JointState, '/joint_states', self._joint_cb, qos_fast)
        self.sub_color = node.create_subscription(Image, '/zed/zed_node/rgb/color/rect/image', self._color_cb, qos_slow)
        self.sub_depth = node.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self._depth_cb, qos_slow)

        # self.diag_timer = self.node.create_timer(2.0, self._print_diagnostics)
        self.node.get_logger().info("RobotCameraSync initialized (Event-Driven, Pristine Timestamps).")

    def _print_diagnostics(self):
        self.node.get_logger().info(
            f"[SYNC DEBUG] In -> Joints: {self.cnt_joint_in} | Color: {self.cnt_color_in} | Depth: {self.cnt_depth_in} || "
            f"Out -> Synced: {self.cnt_synced_out}"
        )

    def _joint_cb(self, msg):
        self.cnt_joint_in += 1
        
        # Aggregate the high-frequency fragmented joint messages
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_joint_state_dict:
                self.latest_joint_state_dict[name] = pos
                
        # Only cache it if we have a complete 12-joint state
        if all(val is not None for val in self.latest_joint_state_dict.values()):
            full_msg = JointState()
            full_msg.header = msg.header 
            full_msg.name = list(self.latest_joint_state_dict.keys())
            full_msg.position = list(self.latest_joint_state_dict.values())
            self.latest_joint_msg = full_msg

    def _color_cb(self, msg):
        self.cnt_color_in += 1
        self.latest_color_msg = msg

    def _depth_cb(self, msg):
        self.cnt_depth_in += 1
        
        # Guard clause: Wait until we have at least one valid color and joint message
        if self.latest_joint_msg is None or self.latest_color_msg is None:
            return
            
        self.cnt_synced_out += 1
        
        # Fire callback to the environment
        self.callback(self.latest_joint_msg, self.latest_color_msg, msg)
        
        # Clear image cache to prevent re-using stale color data if depth drops
        self.latest_color_msg = None

class RealDexEnv(Node):
    """Environment management"""
    def __init__(self, cfg, masks_remove):
        super().__init__('flow_policy')

        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.n_obs_steps = cfg.n_obs_steps
        self.n_action_steps = cfg.n_action_steps
        self.masks_remove = masks_remove
        
        self.JOINT_NAMES = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
            'right_inspire_right_thumb_1_joint', 'right_inspire_right_thumb_2_joint',
            'right_inspire_right_index_1_joint', 'right_inspire_right_middle_1_joint',
            'right_inspire_right_ring_1_joint', 'right_inspire_right_little_1_joint'
        ]
        
        # Initialize data cache
        self.point_cloud_array = []
        self.robot_state_array = []
        
        # Initialize point cloud processor
        self.pc_processor = PointCloudProcessor()
        self.bridge = CvBridge()

        # --- SYNCER ---
        self.syncer = RobotCameraSync(self, self.observation_callback, self.JOINT_NAMES)

        # Publish action
        self.action_pub = self.create_publisher(
            Float64MultiArray,
            '/predicted_action_chunk',
            10
        )

        self.status_pub = self.create_publisher(String, '/policy_status', 10)
        
        self.trigger_sub = self.create_subscription(
            Bool, 
            '/policy_trigger', 
            self.trigger_callback, 
            10
        )

        self.palm_transform = PalmFrameTransform("src/moveit_ur_hand_config/urdf/tesollo_hand_ur10e_flange_workcell.urdf")

        # Flags for control flow
        self.enable_inference = threading.Event() 
        self.force_reset = False                  

        self.debug_pub = self.create_publisher(PointCloud2, '/debug_pc', 10)
        self.new_obs_event = threading.Event()
        self.debug_action_pub = self.create_publisher(PoseStamped, '/debug_action_pose', 10)
        self.chain = setup_kinematics()
        self.traj_pub = self.create_publisher(PoseArray, '/predicted_trajectory', 10)

        self.get_logger().info("RealDexEnv initialized. Waiting for trigger on /policy_trigger...")

    def trigger_callback(self, msg):
        if msg.data:
            self.get_logger().info("RECEIVED TRIGGER: Starting/Restarting Policy.")
            self.force_reset = True
            self.enable_inference.set()

            self.new_obs_event.set()

    def publish_status(self, status_text):
        """Helper to publish status strings"""
        msg = String()
        msg.data = status_text
        self.status_pub.publish(msg)

    # def check_policy_finished(self, state_value):
    #     print(f"Policy Success looks like this: {state_value}")
    #     if state_value > self.success_threshold:
    #         self.success_flag = True
    #     else:
    #         self.success_flag = False

    def observation_callback(self, joint_msg, color_msg, depth_msg):
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            
            pos_dict = dict(zip(joint_msg.name, joint_msg.position))
            ordered_state = [pos_dict[name] for name in self.JOINT_NAMES]
            robot_state = np.array(ordered_state, dtype=np.float32)

            cartesian_state_phys = compute_fk(self.chain, robot_state)

            # print(f"[FK DEBUG] cartesian_state: {np.round(cartesian_state_phys, 4)}")

            # valid_depths = depth_img[~np.isnan(depth_img) & ~np.isinf(depth_img)]
            # if len(valid_depths) > 0:
            #     print(f"[DEPTH DEBUG] Shape: {depth_img.shape} | Min: {valid_depths.min():.3f} | Max: {valid_depths.max():.3f} | NaNs: {np.isnan(depth_img).sum()}")
            # else:
            #     print("[DEPTH DEBUG] ALL DEPTH VALUES ARE NaN OR INF!")
            point_cloud = self.pc_processor.process_rgbd(color_img, depth_img)

            # --- Hand Observation ---
            hand_obs = self._compute_hand_obs()

            if point_cloud is not None:
                # print(f"[PC DEBUG] shape: {point_cloud.shape} | "
                #     f"xyz min: {point_cloud[:,:3].min(axis=0)} | "
                #     f"xyz max: {point_cloud[:,:3].max(axis=0)}")

                self.point_cloud_array.append(point_cloud)
                self.robot_state_array.append(cartesian_state_phys)
                
                if len(self.point_cloud_array) > self.n_obs_steps:
                    self.point_cloud_array.pop(0)
                if len(self.robot_state_array) > self.n_obs_steps:
                    self.robot_state_array.pop(0)
                if len(self.hand_obs_array) > self.n_obs_steps:
                    self.hand_obs_array.pop(0)
                
                self.new_obs_event.set()
            else:
                # If point cloud generation fails, print a warning
                print("[OBS ERROR] Point cloud generation returned None!")

        except Exception as e:
            self.get_logger().error(f"Observation processing error: {e}")

    def _compute_hand_obs(self) -> np.ndarray:
        """
        Compute (5, 13) palm-frame hand observation from latest Tesollo data.
        Returns zeros if data not yet available.
        """
        if self.latest_gripper_msg is None or self.latest_sensor_msg is None:
            return np.zeros((5, 13), dtype=np.float32)

        hand_joints = np.array(self.latest_gripper_msg.joint, dtype=np.float32)  # (20,) degrees
        hand_ft = np.array(self.latest_sensor_msg.force_torque, dtype=np.float32)  # (30,)

        return self.palm_transform(hand_joints, hand_ft).astype(np.float32)  # (5, 13)


    def execution_done_callback(self, msg):
        self.action_completed.set()

    def remove_identical_objects(self, depth_img):
        filtered_depth = depth_img.copy()
        for mask in self.masks_remove:
            filtered_depth[mask.astype(bool)] = 0
        return filtered_depth

    def reset(self):
        """Reset environment state"""
        self.point_cloud_array.clear()
        self.robot_state_array.clear()
        self.force_reset = False

        self.get_logger().info("Resetting environment. Waiting for buffers to fill...")
        return self.get_latest_obs()
    
    def get_latest_obs(self):
        # Block until the background callback flags that a NEW frame arrived
        self.new_obs_event.wait()
        self.new_obs_event.clear()

        # Wait until we have enough history to form a complete observation
        while len(self.robot_state_array) < self.n_obs_steps:
            if self.force_reset:
                return None
            time.sleep(0.01)
            
        return self._get_obs()

    def publish_action_chunk(self, full_action_sequence):
        """Publish flat chunk for forward_position controller."""
        msg = Float64MultiArray()
        # Flatten: each action is 13 values (7 pose + 6 hand)
        msg.data = full_action_sequence[:, :13].flatten().tolist()
        self.action_pub.publish(msg)
    
    def _get_obs(self):
        """Build observation dictionary"""
        obs_dict = {
            'point_cloud': torch.from_numpy(
                np.stack(self.point_cloud_array[-self.n_obs_steps:], axis=0)
            ).unsqueeze(0).to(self.device),
            'agent_pos': torch.from_numpy(
                np.stack(self.robot_state_array[-self.n_obs_steps:], axis=0)
            ).unsqueeze(0).to(self.device),
            'hand_obs': torch.from_numpy(
                np.stack(self.hand_obs_array[-self.n_obs_steps:], axis=0)
            ).unsqueeze(0).to(self.device), # Shape: (1, T_obs, 5, 13)
        }
        
        return obs_dict

def run_policy(alg_name, task_name, run_dir, addition_info=1211, masks_remove=None):
    if not rclpy.ok():
        rclpy.init()

    cfg, policy = init_model(task_name=task_name, alg_name=alg_name, addition_info=addition_info, run_dir=run_dir) 

    env = RealDexEnv(cfg, masks_remove=masks_remove)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(env)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            env.publish_status("IDLE")
            print("[INFO] Policy Idle. Waiting for trigger...")

            env.enable_inference.wait()
            env.enable_inference.clear()

            env.publish_status("RUNNING")
            print("[INFO] Policy Triggered! Resetting environment...")
            
            obs = env.reset()

            if obs is None: 
                env.force_reset = False 
                continue

            task_completed = False

            while rclpy.ok() and not task_completed and not env.force_reset:
                with torch.no_grad():
                    all_actions_raw = policy.predict_action(obs)
                    all_actions = normalize(all_actions_raw)

                    # In run_policy, BEFORE publishing, add:
                    print(f"Action[0] pose: {all_actions[0][:7]}")
                    print(f"Action[0] quat norm: {np.linalg.norm(all_actions[0][3:7])}")
                    print(f"Action[0] hand: {all_actions[0][7:13]}")
                    print(f"Action[0] hand range: [{all_actions[0][7:13].min():.3f}, {all_actions[0][7:13].max():.3f}]")

                    # Publish entire chunk asynchronously
                    env.publish_action_chunk(all_actions)

                    # # Check success based on final step in the chunk
                    # final_action_step = all_actions[-1]
                    
                    # if final_action_step[-1] > env.success_threshold:
                    #     task_completed = True
                    #     env.publish_status("SUCCESS")
                    #     break

                    obs = env.get_latest_obs()
            
            if env.force_reset:
                print("[INFO] Restarting loop due to Trigger...")

                continue

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    run_policy(alg_name="maniflow_pointcloud_policy", 
               task_name="test_block_grasping", 
               run_dir="/ros2_ws/data/outputs/test_block_grasping-maniflow_pointcloud_policy-0112_seed0",
               addition_info = 1211,
               masks_remove=None)
