#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, SimpleFilter
from sensor_msgs.msg import Image, CameraInfo, JointState
import numpy as np
import cv2
from cv_bridge import CvBridge
import os
import threading
import shutil
import queue

FOLDER_NAME = "test_collection"



class RobotCameraSync:
    """
    Synchronizes high-frequency JointStates (550Hz) with ZED 2i images (60Hz)
    """
    def __init__(self, node, callback_function, joint_names):
        self.node = node
        self.callback = callback_function
        self.joint_names = joint_names

        # Buffer for aggregating fragmented joint state packets
        self.latest_joint_state = {name: None for name in self.joint_names}

        # Message filters to interface with the Synchronizer
        self.joint_filt = SimpleFilter()
        self.color_filt = SimpleFilter()
        self.depth_filt = SimpleFilter()

        qos_joints = QoSProfile(
            depth=300, 
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST
        )

        # Camera (high bandwidth, use best effort / sensor data)
        qos_camera = qos_profile_sensor_data

        # Passthrough Publisher to monitor speed externally
        self.synced_joint_pub = self.node.create_publisher(JointState, '~/synced_joint_states', 10)

        # Standard ROS Subscribers using ZED 2i topics
        self.node.create_subscription(JointState, "/joint_states", self.joint_cb, qos_joints)
        self.node.create_subscription(Image, "/zed/zed_node/rgb/color/rect/image", self.color_cb, qos_camera)
        self.node.create_subscription(Image, "/zed/zed_node/depth/depth_registered", self.depth_cb, qos_camera)

        # Synchronizer
        # Reduced slop to 0.01s (10ms) because ZED hardware sync is highly accurate
        self.sync = ApproximateTimeSynchronizer(
            [self.joint_filt, self.color_filt, self.depth_filt],
            queue_size=300,
            slop=0.025
        ) 
        self.sync.registerCallback(self.callback)
        self.node.get_logger().info("ZED Synchronizer Node active. Hardware timestamps enabled.")

    def joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_joint_state:
                self.latest_joint_state[name] = pos
        
        if all(val is not None for val in self.latest_joint_state.values()):
            full_msg = JointState()
            full_msg.header = msg.header 
            full_msg.name = list(self.latest_joint_state.keys())
            full_msg.position = list(self.latest_joint_state.values())
            self.joint_filt.signalMessage(full_msg)

    def color_cb(self, msg):
        """Pass RGB image to synchronizer using original hardware timestamp."""
        self.color_filt.signalMessage(msg)

    def depth_cb(self, msg):
        """Pass depth image to synchronizer using original hardware timestamp."""
        self.depth_filt.signalMessage(msg)

    def sync_cb(self, joint_msg, color_msg, depth_msg):
        """Callback triggered when all three topics are synchronized."""
        self.sync_count += 1
        
        # Performance logging every 2 seconds
        now = self.node.get_clock().now()
        duration_ns = (now - self.last_log_time).nanoseconds
        if duration_ns > 2e9:  # 2 seconds
            duration_s = duration_ns / 1e9
            fps = self.sync_count / duration_s
            
            # Calculate time difference (jitter) between Joint and Camera timestamps
            # Converting header stamps to total nanoseconds for comparison
            j_ts = joint_msg.header.stamp.sec * 1e9 + joint_msg.header.stamp.nanosec
            c_ts = color_msg.header.stamp.sec * 1e9 + color_msg.header.stamp.nanosec
            diff_ms = abs(j_ts - c_ts) / 1e6
            
            self.node.get_logger().info(
                f"Sync: {fps:.1f} FPS | Joint-Camera Delta: {diff_ms:.2f}ms"
            )
            self.sync_count = 0
            self.last_log_time = now

        # Execute the main recording/processing callback
        self.callback(joint_msg, color_msg, depth_msg)


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        default_workdir = os.environ.get('WORKDIR', '/ros2_ws/data/')
        self.root_dir = os.path.join(default_workdir, f'episodes/{FOLDER_NAME}')

        self.JOINT_NAMES = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
            'inspire_right_thumb_1_joint', 'inspire_right_thumb_2_joint',
            'inspire_right_index_1_joint', 'inspire_right_middle_1_joint',
            'inspire_right_ring_1_joint', 'inspire_right_little_1_joint'
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

    def sync_callback(self, joint_msg, img_msg, depth_msg):
        if not self.is_recording:
            return

        msg_time = img_msg.header.stamp.sec + (img_msg.header.stamp.nanosec * 1e-9)

        try:
            pos_dict = dict(zip(joint_msg.name, joint_msg.position))
            current_state = np.array([pos_dict[name] for name in self.JOINT_NAMES], dtype=np.float32)
            
            cv_color = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # Push everything to RAM instantly. Zero disk I/O.
            self.save_queue.put((
                self.current_frame, 
                cv_color, 
                cv_depth, 
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
                    continue # Safety check

                frame_name = f"frame_{frame_idx:04d}"
                
                # Slow disk operations happen safely here
                cv2.imwrite(os.path.join(self.current_episode_dir, "img", f"{frame_name}.png"), color)
                np.save(os.path.join(self.current_episode_dir, "depth", f"{frame_name}.npy"), depth)
                
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_recording()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()