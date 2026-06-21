#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, SimpleFilter
from sensor_msgs.msg import Image, CameraInfo, JointState
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import cv2
from cv_bridge import CvBridge
import os
import threading
import shutil
import queue

FOLDER_NAME = "test_collection"

class RobotCameraSync:
    def __init__(self, node, callback_function, joint_names):
        self.node = node
        self.callback = callback_function
        self.joint_names = joint_names

        self.arm_joints = {'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'}
        self.hand_joints = {'right_inspire_right_thumb_1_joint', 'right_inspire_right_thumb_2_joint',
                            'right_inspire_right_index_1_joint', 'right_inspire_right_middle_1_joint',
                            'right_inspire_right_ring_1_joint', 'right_inspire_right_little_1_joint'}

        self.latest_joint_state = {name: None for name in self.joint_names}
        self.arm_ready = False
        self.last_hand_update = None
        
        # Cache for BOTH image types to prevent race conditions
        self.latest_depth_msg = None
        self.latest_color_msg = None

        self.sync_count = 0
        self.last_log_time = self.node.get_clock().now()

        qos_joints = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        qos_camera_buffered = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST
        )

        self.node.create_subscription(JointState, "/joint_states", self._joint_cb, qos_joints)
        self.node.create_subscription(Image, "/zed/zed_node/depth/depth_registered", self.depth_cb, qos_camera_buffered)
        self.node.create_subscription(Image, "/zed/zed_node/rgb/color/rect/image", self.color_cb, qos_camera_buffered)

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_joint_state:
                self.latest_joint_state[name] = pos

        if any(n in self.hand_joints for n in msg.name):
            self.last_hand_update = self.node.get_clock().now()

        if not self.arm_ready:
            if all(self.latest_joint_state[n] is not None for n in self.arm_joints):
                self.arm_ready = True
                self.node.get_logger().info("Arm joints ready.")

    def depth_cb(self, msg):
        self.latest_depth_msg = msg
        self._try_sync()

    def color_cb(self, msg):
        self.latest_color_msg = msg
        self._try_sync()

    def _try_sync(self):
        # Only proceed if the arm is ready and we have AT LEAST one of each image type buffered
        if not self.arm_ready or self.latest_color_msg is None or self.latest_depth_msg is None:
            return

        c_t = self.latest_color_msg.header.stamp.sec + (self.latest_color_msg.header.stamp.nanosec * 1e-9)
        d_t = self.latest_depth_msg.header.stamp.sec + (self.latest_depth_msg.header.stamp.nanosec * 1e-9)

        time_diff = abs(c_t - d_t)

        if time_diff < 0.025:
            # Timestamps match! Build state and fire the recording callback
            current_state = np.array(
                [self.latest_joint_state[n] if self.latest_joint_state[n] is not None else 0.0 for n in self.joint_names], 
                dtype=np.float32
            )

            # --- FPS Logging ---
            self.sync_count += 1
            now = self.node.get_clock().now()
            duration_ns = (now - self.last_log_time).nanoseconds
            if duration_ns > 2e9:
                duration_s = duration_ns / 1e9
                fps = self.sync_count / duration_s
                hand_age_ms = 0.0
                if self.last_hand_update is not None:
                    hand_age_ms = (now - self.last_hand_update).nanoseconds / 1e6
                self.node.get_logger().info(f"Sync: {fps:.1f} FPS | Hand Age: {hand_age_ms:.0f}ms")
                self.sync_count = 0
                self.last_log_time = now
            # -------------------

            self.callback(current_state, self.latest_color_msg, self.latest_depth_msg)
            
            # Consume the messages so they aren't reused for the next trigger
            self.latest_color_msg = None
            self.latest_depth_msg = None
            
        else:
            # Mismatch: Log the drift to see if we are rejecting valid frames
            self.node.get_logger().warn(f"Dropped pair! Time diff: {time_diff * 1000:.1f}ms")
            
            if c_t < d_t:
                self.latest_color_msg = None
            else:
                self.latest_depth_msg = None

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        default_workdir = os.environ.get('WORKDIR', '/ros2_ws/data/')
        self.root_dir = os.path.join(default_workdir, f'episodes/{FOLDER_NAME}')

        self.JOINT_NAMES = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
            'right_inspire_right_thumb_1_joint', 'right_inspire_right_thumb_2_joint',
            'right_inspire_right_index_1_joint', 'right_inspire_right_middle_1_joint',
            'right_inspire_right_ring_1_joint', 'right_inspire_right_little_1_joint'
        ]

        self.syncer = RobotCameraSync(self, self.sync_callback, self.JOINT_NAMES)

        self.bridge = CvBridge()
        self.is_recording = False
        self.episode_idx = 0
        self.current_frame = 0
        self.current_episode_dir = None
        

        self.state_arrays = []
        self.action_arrays = []
        self.timestamps = []

        self.save_queue = queue.Queue(maxsize=300)
        self.writer_thread = threading.Thread(target=self._disk_writer_loop, daemon=True)
        self.writer_thread.start()

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        self.get_logger().info(f"Saving data to: {self.root_dir}")

        self.sub_info = self.create_subscription(
            CameraInfo, 
            '/zed/zed_node/depth/camera_info', 
            self.camera_info_callback, 
            qos_profile_sensor_data
        )
        self.camera_intrinsics = None

        self.get_logger().info("Ready. Press 's' to START, 'e' to END, 'q' to QUIT.")
        
        self.input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.input_thread.start()

    def sync_callback(self, current_state, img_msg, depth_msg):
        if not self.is_recording:
            return

        msg_time = img_msg.header.stamp.sec + (img_msg.header.stamp.nanosec * 1e-9)

        try:
            # We already built the current_state array in RobotCameraSync.
            # Push everything to RAM instantly. Zero disk I/O.
            self.save_queue.put((
                self.current_frame, 
                img_msg,
                depth_msg,
                current_state, 
                msg_time
            ))

            self.current_frame += 1

        except Exception as e:
            self.get_logger().error(f"Callback Failed: {e}")

    def _disk_writer_loop(self):
        while rclpy.ok():
            try:
                # Blocks until data is available
                frame_idx, color, depth, state, timestamp = self.save_queue.get(timeout=1.0)
                
                if self.current_episode_dir is None:
                    self.save_queue.task_done() 
                    continue # Safety check

                frame_name = f"frame_{frame_idx:04d}"

                cv_color = self.bridge.imgmsg_to_cv2(color, "bgr8")
                cv_depth_f32 = self.bridge.imgmsg_to_cv2(depth, "32FC1")
                cv_depth_scaled = cv_depth_f32 * 1000.0
                cv_depth_scaled = np.nan_to_num(cv_depth_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                cv_depth = cv_depth_scaled.astype(np.uint16)
                
                # Slow disk operations happen safely here
                cv2.imwrite(os.path.join(self.current_episode_dir, "img", f"{frame_name}.png"), cv_color)
                np.save(os.path.join(self.current_episode_dir, "depth", f"{frame_name}.npy"), cv_depth)
                
                self.state_arrays.append(state)
                self.action_arrays.append(state)
                self.timestamps.append(timestamp)
                
                self.save_queue.task_done()

                if frame_idx % 10 == 0:
                    print(f"Saved {frame_idx} frames to disk...", end='\r')

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Disk Write Failed: {e}")


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
            
        os.makedirs(os.path.join(episode_dir, "img"))
        os.makedirs(os.path.join(episode_dir, "depth"))
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

        # Stop adding new frames to the queue
        self.is_recording = False
        print("\nStopping and draining write queue. Please wait...")

        # Block the main thread until the background thread finishes writing all images
        self.save_queue.join()
        
        print("Queue drained. Finalizing arrays...")

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
            except:
                break

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    
    # Use a MultiThreadedExecutor to prevent the GIL from starving ROS callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.is_recording:
            node.stop_recording()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()