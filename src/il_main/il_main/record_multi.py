#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, SimpleFilter
from sensor_msgs.msg import Image, CameraInfo, JointState
import numpy as np
import cv2
from cv_bridge import CvBridge
import os
import threading
import shutil

class RobotMultiCameraSync:
    def __init__(self, node, callback_function, joint_names, camera_names):
        self.node = node
        self.callback = callback_function
        self.joint_names = joint_names
        self.camera_names = camera_names

        self.latest_joint_state = {name: None for name in self.joint_names}
        self.latest_depth = {cam: None for cam in self.camera_names}

        self.joint_filt = SimpleFilter()
        self.color_filters = {cam: SimpleFilter() for cam in self.camera_names}

        # Debugging Flags
        self.got_joints = False
        self.got_color = {cam: False for cam in self.camera_names}

        for cam in self.camera_names:
            # Depth background buffer
            depth_topic = f"/{cam}/{cam}/aligned_depth_to_color/image_raw"
            self.node.create_subscription(
                Image, depth_topic, 
                lambda msg, c=cam: self._depth_cb(msg, c), 
                qos_profile_sensor_data
            )

            # Color (Sync-Trigger)
            color_topic = f"/{cam}/{cam}/color/image_raw"
            self.node.create_subscription(
                Image, color_topic, 
                lambda msg, c=cam: self._color_cb(msg, c), 
                qos_profile_sensor_data
            )

        # Robot-Joints
        self.node.create_subscription(JointState, "/joint_states", self._joint_cb, qos_profile_sensor_data)

        # Synchronizer dynamically
        sync_inputs = [self.joint_filt] + [self.color_filters[cam] for cam in self.camera_names]
        
        self.sync = ApproximateTimeSynchronizer(
            sync_inputs,
            queue_size=50,
            slop=0.04
        ) 
        self.sync.registerCallback(self._internal_callback)

    def _depth_cb(self, msg, cam_name):
        self.latest_depth[cam_name] = msg

    def _color_cb(self, msg, cam_name):
        if not self.got_color[cam_name]:
            self.node.get_logger().info(f"✅ First color image received: {cam_name}")
            self.got_color[cam_name] = True
        self.color_filters[cam_name].signalMessage(msg)

    def _joint_cb(self, msg):
        if not self.got_joints:
            self.node.get_logger().info("✅ First jointstate received")
            self.got_joints = True

        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_joint_state:
                self.latest_joint_state[name] = pos
        
        if all(val is not None for val in self.latest_joint_state.values()):
            full_msg = JointState()
            full_msg.header = msg.header 
            full_msg.name = list(self.latest_joint_state.keys())
            full_msg.position = list(self.latest_joint_state.values())
            self.joint_filt.signalMessage(full_msg)

    def _internal_callback(self, joint_msg, *color_msgs):

        if any(self.latest_depth[cam] is None for cam in self.camera_names):
            self.node.get_logger().warn("Sync hit but depth data is still missing", throttle_duration_sec=2.0)
            return
            
        color_dict = {cam: msg for cam, msg in zip(self.camera_names, color_msgs)}
        depth_dict = dict(self.latest_depth)

        self.callback(joint_msg, color_dict, depth_dict)


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        default_workdir = os.environ.get('WORKDIR', '/ros2_ws/data') # SET accordingly
        self.root_dir = os.path.join(default_workdir, 'episodes')
        self.target_fps = 15.0

        self.JOINT_NAMES = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', # SET accordingly
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.CAMERA_NAMES = ['cam_left', 'cam_right'] # SET accordingly

        self.syncer = RobotMultiCameraSync(self, self.sync_callback, self.JOINT_NAMES, self.CAMERA_NAMES)

        self.bridge = CvBridge()
        self.is_recording = False
        self.episode_idx = 0
        self.current_frame = 0
        self.current_episode_dir = None

        self.state_arrays = []
        self.action_arrays = []
        self.timestamps = []
        
        self.capture_interval = 1.0 / self.target_fps
        self.last_capture_time = 0.0

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        self.get_logger().info(f"Saving data to: {self.root_dir}")

        # Intrinsics only from the first cam
        primary_cam = self.CAMERA_NAMES[0]
        self.sub_info = self.create_subscription(
            CameraInfo, 
            f'/{primary_cam}/{primary_cam}/aligned_depth_to_color/camera_info', 
            self.camera_info_callback, 
            qos_profile_sensor_data
        )
        self.camera_intrinsics = None

        self.get_logger().info("Ready. Press 's' to START, 'e' to END, 'q' to QUIT.")
        
        self.input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.input_thread.start()

    def sync_callback(self, joint_msg, color_dict, depth_dict):
        if not self.is_recording:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self.last_capture_time) < self.capture_interval:
            return
        self.last_capture_time = now

        try:
            pos_dict = dict(zip(joint_msg.name, joint_msg.position))
            current_state = [pos_dict[name] for name in self.JOINT_NAMES]
            current_state_np = np.array(current_state, dtype=np.float32)
            
            frame_name = f"frame_{self.current_frame:04d}"
            
            for cam in self.CAMERA_NAMES:
                cv_color = self.bridge.imgmsg_to_cv2(color_dict[cam], "bgr8")
                cv_depth = self.bridge.imgmsg_to_cv2(depth_dict[cam], "16UC1")
                
                cv2.imwrite(os.path.join(self.current_episode_dir, f"img_{cam}", f"{frame_name}.png"), cv_color)
                np.save(os.path.join(self.current_episode_dir, f"depth_{cam}", f"{frame_name}.npy"), cv_depth)

            self.state_arrays.append(current_state_np)
            self.action_arrays.append(current_state_np)
            self.timestamps.append(now)

            self.current_frame += 1
            if self.current_frame % 10 == 0:
                print(f"Recorded {self.current_frame} frames...", end='\r')

        except Exception as e:
            self.get_logger().error(f"Save Failed: {e}")

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.get_logger().info("Got Intrinsics.")
            self.camera_intrinsics = {'k': msg.k, 'd': msg.d}
            np.save(os.path.join(self.root_dir, "camera_intrinsics.npy"), self.camera_intrinsics)

    def _create_episode_directory(self):
        while True:
            episode_name = f"episode_{self.episode_idx:03d}"
            episode_dir = os.path.join(self.root_dir, episode_name)
            if not os.path.exists(episode_dir):
                break
            self.episode_idx += 1
            
        for cam in self.CAMERA_NAMES:
            os.makedirs(os.path.join(episode_dir, f"img_{cam}"))
            os.makedirs(os.path.join(episode_dir, f"depth_{cam}"))
            
        return episode_dir

    def start_recording(self):
        if self.is_recording:
            print("Already recording!")
            return
            
        self.current_episode_dir = self._create_episode_directory()
        self.state_arrays = []
        self.action_arrays = []
        self.timestamps = []
        self.current_frame = 0
        self.is_recording = True
        self.get_logger().info(f"Started Episode {self.episode_idx}")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        print("\nStopping...")

        num_frames = self.current_frame
        num_states = len(self.state_arrays)

        if num_frames == 0:
            self.get_logger().warn("Empty episode! Deleting directory.")
            shutil.rmtree(self.current_episode_dir)
            return

        if num_frames != num_states:
            self.get_logger().error(f"Mismatch: {num_frames} Images vs {num_states} States. Truncating.")
            min_len = min(num_frames, num_states)
            self.state_arrays = self.state_arrays[:min_len]
            self.action_arrays = self.action_arrays[:min_len]
            self.timestamps = self.timestamps[:min_len]

        np.save(os.path.join(self.current_episode_dir, "state.npy"), np.array(self.state_arrays))
        np.save(os.path.join(self.current_episode_dir, "action.npy"), np.array(self.action_arrays))
        np.save(os.path.join(self.current_episode_dir, "timestamps.npy"), np.array(self.timestamps))
        
        self.get_logger().info(f"Saved Episode {self.episode_idx} ({len(self.state_arrays)} steps)")
        self.episode_idx += 1

    def _keyboard_listener(self):
        while rclpy.ok():
            try:
                i = input()
                if i == 's': self.start_recording()
                elif i == 'e': self.stop_recording()
                elif i == 'q': 
                    self.stop_recording()
                    os._exit(0)
            except:
                break

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_recording()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()