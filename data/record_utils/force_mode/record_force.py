#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from dg_msgs.msg import ReceivedGripperData, ReceivedFingertipSensorData
import pyzed.sl as sl
import numpy as np
import json
import os
import threading
import shutil

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

UR_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


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

        # Joint state tracking (UR arm only)
        self.latest_arm_state = {name: None for name in UR_JOINT_NAMES}
        self.arm_ready = False

        # Tesollo hand state tracking (raw data)
        self.latest_hand_joints = None   # (20,) float32, degrees
        self.latest_hand_ft = None       # (30,) float32, tip-local F/T

        # Queue for buffering states between camera grabs
        self.pending_states_queue = []
        self.pending_lock = threading.Lock()

        # ZED camera
        self.cam = sl.Camera()
        self.cam_lock = threading.Lock()
        self._init_camera()

        # --- ROS Subscribers ---
        qos_joints = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, qos_joints)

        self.create_subscription(
            ReceivedGripperData, '/tesollo/gripper_data',
            self._gripper_cb, 10
        )
        self.create_subscription(
            ReceivedFingertipSensorData, '/tesollo/fingertip_sensor',
            self._sensor_cb, 10
        )

        self.get_logger().info(f"Saving data to: {self.root_dir}")
        self.get_logger().info("Ready. Press 's' to START, 'e' to END, 'q' to QUIT.")

        self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.grab_thread.start()

        self.input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.input_thread.start()

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

    def _joint_cb(self, msg):
        """UR arm joint states."""
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_arm_state:
                self.latest_arm_state[name] = pos

        if not self.arm_ready:
            if all(self.latest_arm_state[n] is not None for n in UR_JOINT_NAMES):
                self.arm_ready = True
                self.get_logger().info("Arm joints ready.")

        if self.is_recording and self.arm_ready:
            ros_ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

            arm_positions = [
                self.latest_arm_state[n] if self.latest_arm_state[n] is not None else 0.0
                for n in UR_JOINT_NAMES
            ]

            # Build combined state snapshot
            state_entry = {
                "timestamp_ns": ros_ts_ns,
                "arm_joints": {
                    "names": UR_JOINT_NAMES,
                    "positions": arm_positions,
                },
                "hand_joints": self.latest_hand_joints.tolist() if self.latest_hand_joints is not None else [0.0] * 20,
                "hand_ft": self.latest_hand_ft.tolist() if self.latest_hand_ft is not None else [0.0] * 30,
            }

            with self.pending_lock:
                self.pending_states_queue.append(state_entry)

    def _gripper_cb(self, msg: ReceivedGripperData):
        """Tesollo hand joint positions (20 DoF, degrees)."""
        self.latest_hand_joints = np.array(msg.joint, dtype=np.float32)

    def _sensor_cb(self, msg: ReceivedFingertipSensorData):
        """Tesollo fingertip F/T sensor (5×6 = 30 values)."""
        self.latest_hand_ft = np.array(msg.force_torque, dtype=np.float32)

    def _create_episode_directory(self):
        while True:
            episode_name = f"episode_{self.episode_idx:03d}"
            episode_dir = os.path.join(self.root_dir, episode_name)
            if not os.path.exists(episode_dir):
                break
            self.episode_idx += 1
        os.makedirs(episode_dir)
        return episode_dir

    def start_recording(self):
        if self.is_recording:
            print("Already recording!")
            return
        if not self.arm_ready:
            print("Arm joints not ready yet, wait...")
            return

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
        self.get_logger().info(f"Started Episode {self.episode_idx} -> {svo_path}")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        with self.cam_lock:
            self.cam.disable_recording()

        self.get_logger().info(f"Stopped Episode {self.episode_idx}. SVO saved.")
        self.episode_idx += 1
        self.current_episode_dir = None

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
                        data.key = "robot_state"
                        data.set_string_content(json.dumps(queued_states))
                        data.timestamp_ns = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)

                        err = self.cam.ingest_data_into_svo(data)
                        if err != sl.ERROR_CODE.SUCCESS:
                            self.get_logger().warn(f"SVO ingest failed: {err}")

    def _keyboard_listener(self):
        while rclpy.ok():
            try:
                i = input()
                if i == 's':
                    self.start_recording()
                elif i == 'e':
                    self.stop_recording()
                elif i == 'q':
                    self.stop_recording()
            except:
                break


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
        node.running = False
        node.grab_thread.join(timeout=1.0)
        with node.cam_lock:
            node.cam.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()