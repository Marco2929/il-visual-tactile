#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import subprocess
import signal
import os
import shutil

class BagRecorderService(Node):
    def __init__(self):
        super().__init__('bag_recorder_service')
        
        # Initialize the service
        self.srv = self.create_service(SetBool, '/record_episode', self.service_callback)
        
        self.recording_process = None
        self.current_bag_path = None
        self.timeout_timer = None
        
        self.CAMERA_NAMES = ['cam_wrist', 'cam_top', 'cam_right']
        self.MAX_RECORD_TIME_SEC = 300.0  # 5 minutes
        
        # Path configuration
        default_workdir = os.environ.get('WORKDIR', '/media/robot-cell-1/Crucial X9/data')
        self.root_dir = os.path.join(default_workdir, 'episodes')
        self.episode_idx = 0
        
        # Ensure root directory exists
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        self.get_logger().info(f"Bag Recorder Service ready. Saving data to: {self.root_dir}")
        self.get_logger().info(f"Configured to record cameras: {self.CAMERA_NAMES}")

    def _get_next_episode_path(self):
        """Scans the directory and returns the next available sequential episode path."""
        while True:
            episode_name = f"episode_{self.episode_idx:03d}"
            episode_dir = os.path.join(self.root_dir, episode_name)
            if not os.path.exists(episode_dir):
                return episode_dir
            self.episode_idx += 1

    def service_callback(self, request, response):
        if request.data:  # Start request
            if self.recording_process is None:
                bag_path = self.start_recording()
                response.success = True
                response.message = f"Recording started successfully at {bag_path}"
            else:
                response.success = False
                response.message = "Failed: Already recording. Stop current recording first."
                self.get_logger().warn(response.message)
                
        else:  # Stop request
            if self.recording_process is not None:
                self.stop_recording(delete_episode=False)
                response.success = True
                response.message = "Recording stopped and saved safely."
            else:
                response.success = False
                response.message = "Failed: No active recording to stop."
                self.get_logger().warn(response.message)
                
        return response

    def start_recording(self):
        self.current_bag_path = self._get_next_episode_path()

        command = [
            "ros2", "bag", "record",
            "-o", self.current_bag_path,
            "/joint_states"
        ]

        for cam in self.CAMERA_NAMES:
            command.extend([
                f"/{cam}/{cam}/color/image_raw",
                f"/{cam}/{cam}/aligned_depth_to_color/image_raw",
                f"/{cam}/{cam}/aligned_depth_to_color/camera_info"
            ])

        self.get_logger().info(f"Starting recording process for: {self.current_bag_path}")
        
        # preexec_fn=os.setsid is required to cleanly kill the process group later
        self.recording_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        
        # Start the safety timeout timer
        self.timeout_timer = self.create_timer(self.MAX_RECORD_TIME_SEC, self.timeout_callback)
        
        return self.current_bag_path

    def timeout_callback(self):
        """Triggered if the recording exceeds the maximum allowed time."""
        self.get_logger().error(f"Recording exceeded {self.MAX_RECORD_TIME_SEC} seconds. Terminating and deleting data to prevent storage overflow.")
        if self.recording_process is not None:
            self.stop_recording(delete_episode=True)

    def stop_recording(self, delete_episode=False):
        # Cancel the timer if it is running
        if self.timeout_timer is not None:
            self.timeout_timer.cancel()
            self.timeout_timer = None

        self.get_logger().info("Stopping recording... sending SIGINT to rosbag2.")
        
        # Kill the entire process group to ensure SQLite threads write the metadata.yaml
        os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
        self.recording_process.wait()
        
        self.recording_process = None
        self.get_logger().info("Process terminated and metadata written.")
        
        if delete_episode and self.current_bag_path and os.path.exists(self.current_bag_path):
            try:
                shutil.rmtree(self.current_bag_path)
                self.get_logger().info(f"Safely deleted abandoned episode directory: {self.current_bag_path}")
                # We intentionally do not increment self.episode_idx here, 
                # so the next recording attempt reuses the deleted episode number.
            except Exception as e:
                self.get_logger().error(f"Failed to delete directory {self.current_bag_path}: {e}")
        else:
            self.episode_idx += 1 # Increment index for the next call
            
        self.current_bag_path = None

def main(args=None):
    rclpy.init(args=args)
    node = BagRecorderService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.recording_process is not None:
            # On keyboard interrupt, we preserve the data rather than deleting it
            node.stop_recording(delete_episode=False)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()