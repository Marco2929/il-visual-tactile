#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Joy
import pyzed.sl as sl
import numpy as np
import json
import os
import threading
import shutil

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    'right_inspire_right_thumb_1_joint', 'right_inspire_right_thumb_2_joint',
    'right_inspire_right_index_1_joint', 'right_inspire_right_middle_1_joint',
    'right_inspire_right_ring_1_joint', 'right_inspire_right_little_1_joint'
]

ARM_JOINTS = {
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
}

TRACKER_BUTTON_INDEX = 3  # The button you pressed on the Vive Tracker
CONFIRMATION_TIMEOUT_SEC = 2.0  # Time window to confirm success


class SVORecorder(Node):
    def __init__(self):
        super().__init__('svo_recorder')

        default_workdir = os.environ.get('WORKDIR', '/root/ros2_ws/data/')
        self.root_dir = os.path.join(default_workdir, f'episodes/{FOLDER_NAME}')
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self.episode_idx = 0
        self.is_recording = False
        self.current_episode_dir = None
        self.running = True

        # Confirmation state
        self.awaiting_confirmation = False
        self.pending_episode_dir = None
        self.pending_episode_idx = None
        self.confirmation_timer = None

        # Joint state tracking
        self.latest_joint_state = {name: None for name in JOINT_NAMES}
        self.arm_ready = False

        # Queue for buffering joint states between camera grabs
        self.pending_states_queue = []
        self.pending_lock = threading.Lock()

        # Tracker button edge detection
        self.prev_button_state = 0

        # ZED camera
        self.cam = sl.Camera()
        self.cam_lock = threading.Lock()
        self._init_camera()

        # ROS joint subscriber
        qos_joints = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, qos_joints)

        # Vive Tracker button subscriber
        self.create_subscription(Joy, "libsurvive/joy", self._joy_cb, 10)

        self.get_logger().info(f"Saving data to: {self.root_dir}")
        self.get_logger().info("Ready. Press tracker button to START/STOP recording.")
        self.get_logger().info(f"After stopping, press again within {CONFIRMATION_TIMEOUT_SEC}s to mark SUCCESS.")

        self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.grab_thread.start()

    def _init_camera(self):
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 60
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.depth_minimum_distance = 200
        init.depth_maximum_distance = 2000
        init.sdk_verbose = 1
        status = self.cam.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"ZED open failed: {status}")
            raise RuntimeError(f"ZED open failed: {status}")

        calib = self.cam.get_camera_information().camera_configuration.calibration_parameters
        left = calib.left_cam
        self.get_logger().info(f"Intrinsics at HD720: fx={left.fx}, fy={left.fy}, cx={left.cx}, cy={left.cy}")
        self.get_logger().info(f"Image size: {left.image_size.width}x{left.image_size.height}")

        intrinsics = {
            "width": left.image_size.width,
            "height": left.image_size.height,
            "fx": left.fx,
            "fy": left.fy,
            "cx": left.cx,
            "cy": left.cy
        }
        intrinsics_path = os.path.join(self.root_dir, "camera_intrinsics.json")
        with open(intrinsics_path, 'w') as f:
            json.dump(intrinsics, f, indent=2)
        self.get_logger().info(f"Saved intrinsics to {intrinsics_path}")
        self.get_logger().info("ZED camera opened.")

    def _joy_cb(self, msg: Joy):
        """Detect rising edge on tracker button to toggle recording."""
        if len(msg.buttons) <= TRACKER_BUTTON_INDEX:
            return

        current = msg.buttons[TRACKER_BUTTON_INDEX]

        # Rising edge: button was 0, now is 1
        if current == 1 and self.prev_button_state == 0:
            if self.awaiting_confirmation:
                # Second press: confirm success
                self._confirm_success()
            elif self.is_recording:
                # First press while recording: stop and await confirmation
                self.stop_recording()
            else:
                # Not recording, not awaiting: start new recording
                self.start_recording()

        self.prev_button_state = current

    def _confirm_success(self):
        """User pressed button again after stopping -> mark as success."""
        if self.confirmation_timer is not None:
            self.confirmation_timer.cancel()
            self.confirmation_timer = None

        self.awaiting_confirmation = False

        if self.pending_episode_dir and os.path.exists(self.pending_episode_dir):
            new_name = f"episode_{self.pending_episode_idx:03d}_success"
            new_path = os.path.join(self.root_dir, new_name)
            os.rename(self.pending_episode_dir, new_path)
            self.get_logger().info(f"✅ Episode {self.pending_episode_idx} marked as SUCCESS -> {new_name}")

        self.pending_episode_dir = None
        self.pending_episode_idx = None

    def _confirmation_timeout(self):
        """Timeout expired without second press -> mark as failed."""
        self.awaiting_confirmation = False

        if self.pending_episode_dir and os.path.exists(self.pending_episode_dir):
            new_name = f"episode_{self.pending_episode_idx:03d}_failed"
            new_path = os.path.join(self.root_dir, new_name)
            os.rename(self.pending_episode_dir, new_path)
            self.get_logger().info(f"❌ Episode {self.pending_episode_idx} marked as FAILED (timeout) -> {new_name}")

        self.pending_episode_dir = None
        self.pending_episode_idx = None
        self.confirmation_timer = None

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_joint_state:
                self.latest_joint_state[name] = pos

        if not self.arm_ready:
            if all(self.latest_joint_state[n] is not None for n in ARM_JOINTS):
                self.arm_ready = True
                self.get_logger().info("Arm joints ready.")

        if self.is_recording and self.arm_ready:
            ros_ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

            positions = [
                self.latest_joint_state[n] if self.latest_joint_state[n] is not None else 0.0
                for n in JOINT_NAMES
            ]

            with self.pending_lock:
                self.pending_states_queue.append({
                    "timestamp_ns": ros_ts_ns,
                    "names": JOINT_NAMES,
                    "positions": positions
                })

    def _create_episode_directory(self):
        while True:
            episode_name = f"episode_{self.episode_idx:03d}"
            episode_dir = os.path.join(self.root_dir, episode_name)
            # Also check for existing success/failed variants
            success_dir = os.path.join(self.root_dir, f"episode_{self.episode_idx:03d}_success")
            failed_dir = os.path.join(self.root_dir, f"episode_{self.episode_idx:03d}_failed")
            if not os.path.exists(episode_dir) and not os.path.exists(success_dir) and not os.path.exists(failed_dir):
                break
            self.episode_idx += 1
        os.makedirs(episode_dir)
        return episode_dir

    def start_recording(self):
        if self.is_recording:
            self.get_logger().warn("Already recording!")
            return
        if not self.arm_ready:
            self.get_logger().warn("Arm joints not ready yet, wait...")
            return

        # If there's a pending confirmation that hasn't been resolved, mark it as failed
        if self.awaiting_confirmation:
            self._confirmation_timeout()

        self.current_episode_dir = self._create_episode_directory()
        svo_path = os.path.join(self.current_episode_dir, "recording.svo2")

        recording_param = sl.RecordingParameters(
            svo_path,
            sl.SVO_COMPRESSION_MODE.H265
        )
        with self.cam_lock:
            err = self.cam.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to start SVO recording: {err}")
            shutil.rmtree(self.current_episode_dir)
            return

        self.is_recording = True
        self.get_logger().info(f"▶ RECORDING Episode {self.episode_idx} -> {svo_path}")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        with self.cam_lock:
            self.cam.disable_recording()

        self.get_logger().info(f"⏹ STOPPED Episode {self.episode_idx}. SVO saved.")
        self.get_logger().info(f"   Press button again within {CONFIRMATION_TIMEOUT_SEC}s to confirm SUCCESS...")

        # Enter confirmation-awaiting state
        self.awaiting_confirmation = True
        self.pending_episode_dir = self.current_episode_dir
        self.pending_episode_idx = self.episode_idx
        self.current_episode_dir = None

        # Start timeout timer using ROS2 timer (one-shot via threading.Timer)
        self.confirmation_timer = threading.Timer(CONFIRMATION_TIMEOUT_SEC, self._confirmation_timeout)
        self.confirmation_timer.daemon = True
        self.confirmation_timer.start()

        self.episode_idx += 1

    def _grab_loop(self):
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 100
        runtime.remove_saturated_areas = True
        runtime.texture_confidence_threshold = 100

        while rclpy.ok() and self.running:
            with self.cam_lock:
                if self.cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    self.get_logger().warn("cam.grab() failed", throttle_duration_sec=5.0)
                    continue

                if self.is_recording:
                    with self.pending_lock:
                        queued_states = self.pending_states_queue
                        self.pending_states_queue = []

                    if queued_states:
                        data = sl.SVOData()
                        data.key = "joint_states"
                        data.set_string_content(json.dumps(queued_states))
                        data.timestamp_ns = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                        err = self.cam.ingest_data_into_svo(data)
                        if err != sl.ERROR_CODE.SUCCESS:
                            self.get_logger().warn(f"SVO ingest failed: {err}")


def main(args=None):
    rclpy.init(args=args)
    node = SVORecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.is_recording:
            node.stop_recording()
        # If still awaiting confirmation at shutdown, mark as failed
        if node.awaiting_confirmation:
            node._confirmation_timeout()
        node.running = False
        node.grab_thread.join(timeout=1.0)
        with node.cam_lock:
            node.cam.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()