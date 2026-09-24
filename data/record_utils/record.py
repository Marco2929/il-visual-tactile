#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Joy, Image, CameraInfo
from std_msgs.msg import Bool
from dg_msgs.msg import ReceivedGripperData, ReceivedFingertipSensorData
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

# --- Wrist camera configuration -------------------------------------------
# The wrist camera is interchangeable: Intel RealSense D435i or Orbbec Gemini 305.
# Select with WRIST_CAMERA=realsense|orbbec|auto (default: auto).
#
# Note on depth: RealSense publishes a separate aligned topic, while the Orbbec
# wrapper aligns in-place on /depth/image_raw when launched with
# depth_registration:=true. Both therefore yield depth registered to color.
WRIST_CAMERA_PRESETS = {
    "realsense": {
        "color": "/cam_right/cam_right/color/image_raw",
        "depth": "/cam_right/cam_right/aligned_depth_to_color/image_raw",
        "info":  "/cam_right/cam_right/color/camera_info",
    },
    "orbbec": {
        "color": "/cam_gemini/color/image_raw",
        "depth": "/cam_gemini/depth/image_raw",
        "info":  "/cam_gemini/color/camera_info",
    },
}

WRIST_CAMERA = os.environ.get("WRIST_CAMERA", "auto").lower()

# Directory name inside each episode holding the wrist camera bag.
# Kept as "realsense_bag" by default for compatibility with the existing
# extractors (record_extractor/extract_episodes.py, clean_episodes.py).
WRIST_BAG_NAME = os.environ.get("WRIST_BAG_NAME", "realsense_bag")

# --- Robot configuration ---------------------------------------------------
# UR arm joints come from /joint_states. The Tesollo DG-5F hand publishes its
# own topics (raw joints + fingertip F/T), so it is NOT part of JointState.
UR_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Tesollo DG-5F: 20 joint values (degrees) and 5x6 fingertip force/torque.
HAND_NUM_JOINTS = 20
HAND_NUM_FT = 30

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

        # UR arm state
        self.latest_arm_state = {name: None for name in UR_JOINT_NAMES}
        self.arm_ready = False

        # Tesollo hand state (raw, as published by the driver)
        self.latest_hand_joints = None   # (20,) float32, degrees
        self.latest_hand_ft = None       # (30,) float32, tip-local F/T

        self.pending_states_queue = []
        self.pending_lock = threading.Lock()

        self.prev_button_state = 0
        self.bag_process = None

        # --- Wrist camera heartbeat ---
        self._wrist_last_seen = 0.0

        # --- TELEOP PUBLISHER ---
        self.teleop_pub = self.create_publisher(Bool, '/teleop_enable', 10)

        self.cam = sl.Camera()
        self.cam_lock = threading.Lock()
        self._init_camera()

        qos_joints = QoSProfile(
            depth=100, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST
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

        qos_joy = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(Joy, "libsurvive/joy", self._joy_cb, qos_joy)

        # --- Wrist camera subscriptions ---
        self.wrist_model, self.wrist_topics = self._resolve_wrist_camera()
        self.get_logger().info(
            f"Wrist camera: {self.wrist_model} "
            f"(color={self.wrist_topics['color']}, depth={self.wrist_topics['depth']})"
        )

        qos_rs = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST
        )
        self.create_subscription(Image, self.wrist_topics["color"], self._wrist_heartbeat, qos_rs)
        self._wrist_intrinsics_saved = False
        self.create_subscription(
            CameraInfo,
            self.wrist_topics["info"],
            self._wrist_camera_info_cb,
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

    # --- Wrist camera discovery & health ---
    def _resolve_wrist_camera(self):
        """Pick the wrist camera topics, either from WRIST_CAMERA or by probing the graph.

        Explicit presets are trusted immediately. In "auto" mode we wait briefly for
        ROS graph discovery (subscribing before discovery completes yields no data)
        and select the preset whose color topic is actually advertised.
        """
        if WRIST_CAMERA in WRIST_CAMERA_PRESETS:
            topics = dict(WRIST_CAMERA_PRESETS[WRIST_CAMERA])
            model = WRIST_CAMERA
        else:
            if WRIST_CAMERA != "auto":
                self.get_logger().warn(
                    f"Unknown WRIST_CAMERA='{WRIST_CAMERA}', falling back to auto-detection."
                )
            model, topics = self._probe_wrist_camera()

        # Per-topic overrides always win, so odd namespaces stay usable.
        for key, env in (("color", "WRIST_COLOR_TOPIC"),
                         ("depth", "WRIST_DEPTH_TOPIC"),
                         ("info", "WRIST_INFO_TOPIC")):
            override = os.environ.get(env)
            if override:
                topics[key] = override
                model = f"{model}+override"
        return model, topics

    def _probe_wrist_camera(self, timeout_sec: float = 5.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            advertised = {name for name, _ in self.get_topic_names_and_types()}
            for model, topics in WRIST_CAMERA_PRESETS.items():
                if topics["color"] in advertised:
                    return model, dict(topics)
            time.sleep(0.25)

        fallback = "realsense"
        self.get_logger().warn(
            f"No wrist camera detected within {timeout_sec:.0f}s; "
            f"assuming '{fallback}'. Set WRIST_CAMERA to override."
        )
        return fallback, dict(WRIST_CAMERA_PRESETS[fallback])

    def _wrist_heartbeat(self, msg):
        self._wrist_last_seen = time.time()

    def _is_wrist_camera_live(self) -> bool:
        return (time.time() - self._wrist_last_seen) < 2.0

    # --- Bag Recording Helpers ---
    def _start_bag_recording(self, episode_dir):
        bag_path = os.path.join(episode_dir, WRIST_BAG_NAME)
        yaml_path = os.path.join(episode_dir, "qos_overrides.yaml")
        color_topic = self.wrist_topics["color"]
        depth_topic = self.wrist_topics["depth"]

        with open(yaml_path, 'w') as f:
            for t in [color_topic, depth_topic]:
                f.write(f"{t}:\n  reliability: best_effort\n  history: keep_last\n  depth: 10\n")
        
        # Record the topic layout so extraction does not have to guess which
        # camera produced this episode.
        with open(os.path.join(episode_dir, "wrist_camera_topics.json"), 'w') as f:
            json.dump({
                "model": self.wrist_model,
                "bag": WRIST_BAG_NAME,
                "color_topic": color_topic,
                "depth_topic": depth_topic,
                "info_topic": self.wrist_topics["info"],
            }, f, indent=2)

        cmd = [
        "ros2", "bag", "record",
        color_topic, depth_topic,
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
                elif not self._is_wrist_camera_live():
                    self.get_logger().error(
                        f"❌ Wrist camera ({self.wrist_model}) not publishing on "
                        f"{self.wrist_topics['color']}! Start camera first."
                    )
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
    def _gripper_cb(self, msg: ReceivedGripperData):
        """Tesollo hand joint positions (20 DoF, degrees)."""
        self.latest_hand_joints = np.array(msg.joint, dtype=np.float32)

    def _sensor_cb(self, msg: ReceivedFingertipSensorData):
        """Tesollo fingertip F/T sensor (5x6 = 30 values)."""
        self.latest_hand_ft = np.array(msg.force_torque, dtype=np.float32)

    def _wrist_camera_info_cb(self, msg: CameraInfo):
        """Save wrist camera intrinsics once."""
        if self._wrist_intrinsics_saved:
            return

        intrinsics = {
            "camera_model": self.wrist_model,
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

        self._wrist_intrinsics_saved = True
        self.get_logger().info(f"Wrist camera ({self.wrist_model}) intrinsics saved: {path}")

    def _joint_cb(self, msg):
        """UR arm joint states; drives the recording tick."""
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_arm_state:
                self.latest_arm_state[name] = pos

        if not self.arm_ready and all(self.latest_arm_state[n] is not None for n in UR_JOINT_NAMES):
            self.arm_ready = True
            self.get_logger().info("Arm joints ready.")

        if self.state == State.RECORDING and self.arm_ready:
            ros_ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

            arm_positions = [
                self.latest_arm_state[n] if self.latest_arm_state[n] is not None else 0.0
                for n in UR_JOINT_NAMES
            ]

            # Schema must match extract_svo_force.py (SVO key "robot_state").
            state_entry = {
                "timestamp_ns": ros_ts_ns,
                "arm_joints": {
                    "names": UR_JOINT_NAMES,
                    "positions": arm_positions,
                },
                "hand_joints": (self.latest_hand_joints.tolist()
                                if self.latest_hand_joints is not None
                                else [0.0] * HAND_NUM_JOINTS),
                "hand_ft": (self.latest_hand_ft.tolist()
                            if self.latest_hand_ft is not None
                            else [0.0] * HAND_NUM_FT),
            }

            with self.pending_lock:
                self.pending_states_queue.append(state_entry)

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
                        data.key = "robot_state"
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