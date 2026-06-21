#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Joy, Image, CameraInfo
from std_msgs.msg import Bool
from service_interfaces.msg import GetForceAct1
import pyzed.sl as sl
import numpy as np
import json
import os
import threading
import shutil
import subprocess
import signal
import time

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

REALSENSE_COLOR_TOPIC = "/cam_right/cam_right/color/image_raw"
REALSENSE_DEPTH_TOPIC = "/cam_right/cam_right/aligned_depth_to_color/image_raw"

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

FORCE_NAMES = [
    'right_inspire_right_little_1_joint',
    'right_inspire_right_ring_1_joint',
    'right_inspire_right_middle_1_joint',
    'right_inspire_right_index_1_joint',
    'right_inspire_right_thumb_2_joint',
    'right_inspire_right_thumb_1_joint'
]

TRACKER_BUTTON_INDEX = 3
CONFIRMATION_TIMEOUT_SEC = 2.0

class State:
    IDLE = 0
    SETUP = 1
    RECORDING = 2
    CONFIRMING = 3

class SVORecorder(Node):
    def __init__(self):
        super().__init__('svo_recorder')

        default_workdir = os.environ.get('WORKDIR', '/root/ros2_ws/data/')
        self.root_dir = os.path.join(default_workdir, f'episodes/{FOLDER_NAME}')
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self.episode_idx = 0
        self.state = State.IDLE
        self.running = True

        self.current_episode_dir = None
        self.pending_episode_dir = None
        self.pending_episode_idx = None
        self.confirmation_timer = None

        self.latest_joint_state = {name: None for name in JOINT_NAMES}
        self.latest_force_values = {name: 0 for name in FORCE_NAMES}
        self.arm_ready = False

        self.pending_states_queue = []
        self.pending_lock = threading.Lock()

        self.prev_button_state = 0
        self.bag_process = None

        # --- RealSense heartbeat ---
        self._rs_last_seen = 0.0

        # --- TELEOP PUBLISHER ---
        self.teleop_pub = self.create_publisher(Bool, '/teleop_enable', 10)

        self.cam = sl.Camera()
        self.cam_lock = threading.Lock()
        self._init_camera()

        qos_joints = QoSProfile(
            depth=100, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, qos_joints)

        qos_force = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(GetForceAct1, "force_data", self._force_cb, qos_force)

        qos_joy = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(Joy, "libsurvive/joy", self._joy_cb, qos_joy)

        # --- RealSense heartbeat subscription ---
        qos_rs = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(Image, REALSENSE_COLOR_TOPIC, self._rs_heartbeat, qos_rs)
        self._rs_intrinsics_saved = False
        self.create_subscription(
            CameraInfo,
            "/cam_right/cam_right/color/camera_info",
            self._rs_camera_info_cb,
            qos_rs
        )

        self.get_logger().info(f"Saving data to: {self.root_dir}")
        self.get_logger().info("Ready. Tracker Button Workflow:")
        self.get_logger().info("1. Click -> SETUP (Move robot to start position)")
        self.get_logger().info("2. Click -> RECORDING (Start task)")
        self.get_logger().info("3. Click -> STOP (Halt robot)")
        self.get_logger().info("4. Click (within 2s) -> CONFIRM SUCCESS")

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
        if self.cam.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED open failed")

        calib = self.cam.get_camera_information().camera_configuration.calibration_parameters
        intrinsics = {
            "width": calib.left_cam.image_size.width, "height": calib.left_cam.image_size.height,
            "fx": calib.left_cam.fx, "fy": calib.left_cam.fy, "cx": calib.left_cam.cx, "cy": calib.left_cam.cy
        }
        with open(os.path.join(self.root_dir, "camera_intrinsics.json"), 'w') as f:
            json.dump(intrinsics, f, indent=2)

    # --- RealSense Health ---
    def _rs_heartbeat(self, msg):
        self._rs_last_seen = time.time()

    def _is_realsense_live(self) -> bool:
        return (time.time() - self._rs_last_seen) < 2.0

    # --- Bag Recording Helpers ---
    def _start_bag_recording(self, episode_dir):
        bag_path = os.path.join(episode_dir, "realsense_bag")
        yaml_path = os.path.join(episode_dir, "qos_overrides.yaml")
        with open(yaml_path, 'w') as f:
            for t in [REALSENSE_COLOR_TOPIC, REALSENSE_DEPTH_TOPIC]:
                f.write(f"{t}:\n  reliability: best_effort\n  history: keep_last\n  depth: 10\n")
        
        cmd = [
        "ros2", "bag", "record",
        REALSENSE_COLOR_TOPIC, REALSENSE_DEPTH_TOPIC,
        "-o", bag_path,
        "--qos-profile-overrides-path", yaml_path,
        "--compression-mode", "message",
        "--compression-format", "zstd",
        ]
        self.bag_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid
        )

    def _stop_bag_recording(self):
        if self.bag_process is None: return
        try:
            os.killpg(os.getpgid(self.bag_process.pid), signal.SIGINT)
            self.bag_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.bag_process.pid), signal.SIGKILL)
            self.bag_process.wait()
        except Exception: pass
        finally:
            self.bag_process = None

    # --- State Machine & Button Logic ---
    def _joy_cb(self, msg: Joy):
        if len(msg.buttons) <= TRACKER_BUTTON_INDEX: return
        current = msg.buttons[TRACKER_BUTTON_INDEX]
        
        if current == 1 and self.prev_button_state == 0:
            if self.state == State.IDLE:
                if not self.arm_ready:
                    self.get_logger().warn("Arm joints not ready!")
                elif not self._is_realsense_live():
                    self.get_logger().error("❌ RealSense not publishing! Start camera first.")
                else:
                    self.state = State.SETUP
                    self.teleop_pub.publish(Bool(data=True))
                    self.get_logger().info("[SETUP] Teleop ACTIVE. Move to start position.")

            elif self.state == State.SETUP:
                self._start_recording_internal()

            elif self.state == State.RECORDING:
                self._stop_recording_internal()

            elif self.state == State.CONFIRMING:
                self._confirm_success()

        self.prev_button_state = current

    def _create_episode_directory(self):
        while True:
            ep_dir = os.path.join(self.root_dir, f"episode_{self.episode_idx:03d}")
            if not any(os.path.exists(d) for d in [ep_dir, ep_dir + "_success", ep_dir + "_failed"]):
                break
            self.episode_idx += 1
        os.makedirs(ep_dir)
        return ep_dir

    def _start_recording_internal(self):
        self.current_episode_dir = self._create_episode_directory()
        svo_path = os.path.join(self.current_episode_dir, "recording.svo2")

        self._start_bag_recording(self.current_episode_dir)

        recording_param = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H265)
        with self.cam_lock:
            err = self.cam.enable_recording(recording_param)
            
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("ZED Recording failed!")
            self._stop_bag_recording()
            shutil.rmtree(self.current_episode_dir)
            self.state = State.IDLE
            self.teleop_pub.publish(Bool(data=False))
            return

        self.state = State.RECORDING
        self.get_logger().info(f"[RECORDING] Episode {self.episode_idx} started.")

    def _stop_recording_internal(self):
        self.teleop_pub.publish(Bool(data=False))
        self.state = State.CONFIRMING

        with self.cam_lock: self.cam.disable_recording()
        self._stop_bag_recording()

        self.get_logger().info(f"[STOPPED] Episode {self.episode_idx}. Press within {CONFIRMATION_TIMEOUT_SEC}s for SUCCESS.")

        self.pending_episode_dir = self.current_episode_dir
        self.pending_episode_idx = self.episode_idx
        self.current_episode_dir = None

        self.confirmation_timer = threading.Timer(CONFIRMATION_TIMEOUT_SEC, self._confirmation_timeout)
        self.confirmation_timer.daemon = True
        self.confirmation_timer.start()
        self.episode_idx += 1

    def _confirm_success(self):
        if self.confirmation_timer: self.confirmation_timer.cancel()
        if self.pending_episode_dir and os.path.exists(self.pending_episode_dir):
            os.rename(self.pending_episode_dir, os.path.join(self.root_dir, f"episode_{self.pending_episode_idx:03d}_success"))
            self.get_logger().info(f"✅ Episode {self.pending_episode_idx} -> SUCCESS")
        
        self.pending_episode_dir = None
        self.state = State.IDLE

    def _confirmation_timeout(self):
        if self.pending_episode_dir and os.path.exists(self.pending_episode_dir):
            os.rename(self.pending_episode_dir, os.path.join(self.root_dir, f"episode_{self.pending_episode_idx:03d}_failed"))
            self.get_logger().info(f"❌ Episode {self.pending_episode_idx} -> FAILED (timeout)")
        
        self.pending_episode_dir = None
        self.state = State.IDLE

    # --- Sensor Callbacks ---
    def _force_cb(self, msg: GetForceAct1):
        for name, force in zip(msg.finger_names, msg.force_values):
            if name in self.latest_force_values:
                self.latest_force_values[name] = force

    def _rs_camera_info_cb(self, msg: CameraInfo):
        """Save RealSense intrinsics once."""
        if self._rs_intrinsics_saved:
            return

        intrinsics = {
            "width": msg.width,
            "height": msg.height,
            "fx": msg.k[0],
            "fy": msg.k[4],
            "cx": msg.k[2],
            "cy": msg.k[5],
            "distortion_model": msg.distortion_model,
            "distortion_coeffs": list(msg.d)
        }

        path = os.path.join(self.root_dir, "wrist_camera_intrinsics.json")
        with open(path, 'w') as f:
            json.dump(intrinsics, f, indent=2)

        self._rs_intrinsics_saved = True
        self.get_logger().info(f"Wrist camera (RealSense) intrinsics saved: {path}")

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_joint_state:
                self.latest_joint_state[name] = pos

        if not self.arm_ready and all(self.latest_joint_state[n] is not None for n in ARM_JOINTS):
            self.arm_ready = True
            self.get_logger().info("Arm joints ready.")

        if self.state == State.RECORDING and self.arm_ready:
            ros_ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            positions = [self.latest_joint_state[n] if self.latest_joint_state[n] is not None else 0.0 for n in JOINT_NAMES]
            forces = [self.latest_force_values[n] for n in FORCE_NAMES]
            with self.pending_lock:
                self.pending_states_queue.append({
                    "timestamp_ns": ros_ts_ns,
                    "names": JOINT_NAMES,
                    "positions": positions,
                    "force_names": FORCE_NAMES,
                    "force_values": forces
                })

    def _grab_loop(self):
        runtime = sl.RuntimeParameters(confidence_threshold=100, remove_saturated_areas=True, texture_confidence_threshold=100)
        while rclpy.ok() and self.running:
            with self.cam_lock:
                if self.cam.grab(runtime) != sl.ERROR_CODE.SUCCESS: continue

                if self.state == State.RECORDING:
                    with self.pending_lock:
                        queued_states = self.pending_states_queue
                        self.pending_states_queue = []

                    if queued_states:
                        data = sl.SVOData()
                        data.key = "joint_states"
                        data.set_string_content(json.dumps(queued_states))
                        data.timestamp_ns = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                        self.cam.ingest_data_into_svo(data)

def main(args=None):
    rclpy.init(args=args)
    node = SVORecorder()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.teleop_pub.publish(Bool(data=False))
        if node.state == State.RECORDING: node._stop_recording_internal()
        if node.state == State.CONFIRMING: node._confirmation_timeout()
        node.running = False
        node.grab_thread.join(timeout=1.0)
        with node.cam_lock: node.cam.close()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()