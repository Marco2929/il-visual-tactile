#!/usr/bin/env python3
"""ZED SVO + wrist-camera episode recorder (UR arm + Tesollo DG-5F hand)."""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from enum import IntEnum

import numpy as np
import pyzed.sl as sl
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState, Joy
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory

# The recording plan (slot bookkeeping, labels.json, operator prompts) lives
# in record_utils/plan/ and is optional: with PLAN_ENABLED=0 attach() returns
# None and this file behaves exactly as before. Running record.py directly
# does not put the parent of record_utils/ on sys.path, so both spellings are
# tried rather than assuming one entry point.
try:
    from record_utils.plan import attach as attach_plan
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from record_utils.plan import attach as attach_plan

env = os.environ.get

# --- Paths -----------------------------------------------------------------
# Episodes go to data/episodes/<EPISODES_SUBDIR>/<FOLDER_NAME>. "external" is the
# bind mount of the larger disk and matches the defaults of
# record_utils/camera_info.py and dataset_builder/cli.py, so extractors find the
# episodes without extra env vars. Set EPISODES_SUBDIR="" to skip the subdir.
WORKDIR = env("WORKDIR", "/root/ros2_ws/data/")
FOLDER_NAME = env("FOLDER_NAME", "test_collection")
EPISODES_SUBDIR = env("EPISODES_SUBDIR", "external")

# --- Wrist camera (Intel RealSense D435i or Orbbec Gemini 305) -------------
# RealSense publishes a separate aligned depth topic; the Orbbec wrapper aligns
# in-place on /depth/image_raw (depth_registration:=true). Both are registered
# to color. Select with WRIST_CAMERA=realsense|orbbec|auto.
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
WRIST_CAMERA = env("WRIST_CAMERA", "auto").lower()
WRIST_TOPIC_OVERRIDE_ENVS = {
    "color": "WRIST_COLOR_TOPIC",
    "depth": "WRIST_DEPTH_TOPIC",
    "info": "WRIST_INFO_TOPIC",
}
# Bag directory inside each episode; "realsense_bag" keeps the existing
# extractors (record_extractor/extract_episodes.py, clean_episodes.py) working.
WRIST_BAG_NAME = env("WRIST_BAG_NAME", "realsense_bag")
WRIST_LIVE_TIMEOUT_SEC = 2.0

# --- Robot configuration ---------------------------------------------------
# UR arm joints come from /joint_states; the Tesollo DG-5F publishes its own
# topics (joints + fingertip F/T) and is NOT part of that JointState.
UR_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# --- Arm action (teleoperation command) ------------------------------------
# /joint_states is where the arm IS. Deriving the action from it (FK, then
# next_state - current) does not recover what the arm was TOLD to do: that
# difference is the servo's scaling and singularity damping, the cartesian
# controller's compliance, and one to several ticks of tracking lag. Trained
# on measured deltas, the policy learns to reproduce the LAG as if it were
# intent - and at deploy time the executor adds its own lag on top.
#
# teleop_ur.py publishes the commanded tool0 pose in base_link on
# /teleop/target_pose. It is an ABSOLUTE pose, the same quantity (same frame,
# same link) that control_ur_tesollo_cartesian.py sends to /target_frame at
# deploy time, so recording it closes the train/deploy gap for the arm exactly
# as hand_action already does for the hand.
#
# Absolute, not delta, on purpose: deltas are only meaningful at the rate they
# were produced, so any resampling in the dataset builder silently rescales
# every commanded velocity. The builder can always difference an absolute
# pose; it cannot undo a mis-scaled delta.
ARM_ACTION_TOPIC = env("ARM_ACTION_TOPIC", "/teleop/target_pose")
ARM_ACTION_DIM = 7           # xyz + quat xyzw, tool0 in base_link
ARM_ACTION_FRAME = env("ARM_ACTION_FRAME", "base_link")
# The teleop node only publishes while enabled, so staleness is measured from
# the moment teleop went active, not from node start.
ARM_ACTION_LIVE_TIMEOUT_SEC = 1.0
HAND_NUM_JOINTS = 20
HAND_NUM_FT = 30
HAND_NUM_FINGERS = 5

# The old dg_msgs stack is gone; the SenseGlove stack exposes the same data via
# ros2_control: joints -> /<HAND_NS>/joint_states, F/T -> fingertip_{1..5}_broadcaster.
HAND_NS = env("HAND_NS", "dg5f_right")

# Index by name: JointState ordering is not guaranteed stable across driver
# versions and a silent permutation would be invisible in the recorded arrays.
HAND_JOINT_NAMES = [f"rj_dg_{finger}_{joint}"
                    for finger in range(1, 6)
                    for joint in range(1, 5)]

# --- Hand action (teleoperation command) -----------------------------------
# dg5f_interaction/dg5f_robot_hand_mapper_node retargets the SenseGlove state
# (/r1/glove<id>/rh/glove_states) and publishes the command that actually moves
# the hand:
#
#   /<HAND_NS>/<HAND_CONTROLLER>/joint_trajectory
#
# This is the same topic control_ur_tesollo_cartesian.py publishes to at deploy
# time, so recording it gives the commanded hand pose instead of only the
# measured one. Note the asymmetry it captures: hand_joints is where the
# fingers ARE, hand_action is where they were TOLD to go.
#
# The topic carries RADIANS. hand_joints is converted to degrees on the way in
# (see below), and the deploy node converts degrees back to radians, so the
# action is converted with the SAME flag - otherwise state and action would
# differ by a factor of ~57.
HAND_CONTROLLER = env("HAND_CONTROLLER", "dg5f_right_controller")
HAND_TRAJECTORY_TOPIC = env(
    "HAND_TRAJECTORY_TOPIC", f"/{HAND_NS}/{HAND_CONTROLLER}/joint_trajectory"
)

# joint_states is in RADIANS, but the schema and everything downstream
# (extract_svo_force.py, merge_episodes_force.py, trained policies) expects the
# DEGREES the old driver published. HAND_JOINTS_IN_DEGREES=0 records radians -
# only with retraining, it changes the action scale by ~57x.
HAND_JOINTS_IN_DEGREES = env("HAND_JOINTS_IN_DEGREES", "1") != "0"

FINGERTIP_WRENCH_TOPIC = env("FINGERTIP_WRENCH_TOPIC", "auto")
WRENCH_SOURCES = ("wrench_filtered", "wrench")  # preference order
# A source counts as live if it produced a full 5-finger set this recently.
FT_LIVE_TIMEOUT_SEC = 1.0

TRACKER_BUTTON_INDEX = 3
CONFIRMATION_TIMEOUT_SEC = 2.0

# Failed episodes are discarded by default: they are never extracted, and the
# SVO plus bag of a single episode is substantial. Note that "failed" is the
# TIMEOUT branch - missing the confirmation window by a moment deletes an
# otherwise good episode. Set KEEP_FAILED_EPISODES=1 to keep them as
# episode_XXX_failed instead.
KEEP_FAILED_EPISODES = env("KEEP_FAILED_EPISODES", "0") != "0"


class State(IntEnum):
    IDLE = 0
    SETUP = 1
    RECORDING = 2
    CONFIRMING = 3


def _qos(depth, reliable=True):
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


class SVORecorder(Node):
    def __init__(self):
        super().__init__("svo_recorder")

        self.root_dir = os.path.join(WORKDIR, "episodes", EPISODES_SUBDIR, FOLDER_NAME)
        os.makedirs(self.root_dir, exist_ok=True)

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
        self._hand_joint_warned = False

        # Tesollo hand action (teleop command), same units as latest_hand_joints
        self.latest_hand_action = None   # (20,) float32, degrees
        self._hand_action_warned = False
        self._hand_action_frames_total = 0
        self._hand_action_frames_zero = 0

        # UR arm action (teleop command): absolute tool0 pose in base_link
        self.latest_arm_action = None    # (7,) float32, xyz + quat xyzw
        self._arm_action_last_seen = 0.0
        self._arm_action_frame_warned = False
        # Last frame_id actually received, recorded in the episode metadata so
        # the frame remains auditable now that it no longer blocks recording.
        self._arm_action_frame_seen = None
        self._arm_action_frames_total = 0
        self._arm_action_frames_zero = 0

        self.pending_states_queue = []
        self.pending_lock = threading.Lock()

        self.prev_button_state = 0
        self.bag_process = None
        self._wrist_last_seen = 0.0
        self._wrist_intrinsics_saved = False

        self.teleop_pub = self.create_publisher(Bool, "/teleop_enable", 10)

        # Attached before the camera so a misconfigured plan (bad PLAN_BLOCK /
        # PLAN_HEIGHT) is reported before the ZED is opened and held.
        self.plan = attach_plan(self, root_dir=self.root_dir,
                                collection=FOLDER_NAME)

        self.cam = sl.Camera()
        self.cam_lock = threading.Lock()
        self._init_camera()

        self._init_robot_subs()
        self._init_fingertip_subs()
        self.create_subscription(Joy, "libsurvive/joy", self._joy_cb, _qos(10, reliable=False))
        self._init_wrist_subs()

        self._log_usage()

        self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.grab_thread.start()

    # --- Setup -------------------------------------------------------------
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

        left = self.cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        _write_json(os.path.join(self.root_dir, "camera_intrinsics.json"), {
            "width": left.image_size.width, "height": left.image_size.height,
            "fx": left.fx, "fy": left.fy, "cx": left.cx, "cy": left.cy,
        })

    def _init_robot_subs(self):
        qos = _qos(100)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, qos)
        self.create_subscription(JointState, f"/{HAND_NS}/joint_states", self._gripper_cb, qos)
        self.create_subscription(
            JointTrajectory, HAND_TRAJECTORY_TOPIC, self._hand_action_cb, qos
        )
        self.create_subscription(
            PoseStamped, ARM_ACTION_TOPIC, self._arm_action_cb, qos
        )
        self.get_logger().info(f"Hand action source: {HAND_TRAJECTORY_TOPIC}")
        self.get_logger().info(f"Arm action source:  {ARM_ACTION_TOPIC}")

    def _init_fingertip_subs(self):
        """One WrenchStamped per finger, assembled into the old flat 30-vector."""
        if FINGERTIP_WRENCH_TOPIC in WRENCH_SOURCES:
            self._ft_sources = (FINGERTIP_WRENCH_TOPIC,)
        else:
            if FINGERTIP_WRENCH_TOPIC != "auto":
                self.get_logger().warn(
                    f"Unknown FINGERTIP_WRENCH_TOPIC='{FINGERTIP_WRENCH_TOPIC}', using auto-selection."
                )
            self._ft_sources = WRENCH_SOURCES

        # Per-source staging, so a dead topic can never contribute to hand_ft.
        self._hand_ft_parts = {s: [None] * HAND_NUM_FINGERS for s in self._ft_sources}
        self._ft_last_full = {s: 0.0 for s in self._ft_sources}
        self._ft_source_in_use = None
        self._ft_frames_total = 0
        self._ft_frames_zero = 0

        qos = _qos(10)
        for source in self._ft_sources:
            for idx in range(HAND_NUM_FINGERS):
                self.create_subscription(
                    WrenchStamped,
                    f"/{HAND_NS}/fingertip_{idx + 1}_broadcaster/{source}",
                    lambda msg, i=idx, s=source: self._fingertip_cb(msg, i, s),
                    qos,
                )
        self.get_logger().info(
            f"Fingertip F/T sources: {', '.join(self._ft_sources)} "
            f"(under /{HAND_NS}/fingertip_*_broadcaster/)"
        )

    def _init_wrist_subs(self):
        self.wrist_model, self.wrist_topics = self._resolve_wrist_camera()
        self.get_logger().info(
            f"Wrist camera: {self.wrist_model} "
            f"(color={self.wrist_topics['color']}, depth={self.wrist_topics['depth']})"
        )
        qos = _qos(1, reliable=False)
        self.create_subscription(Image, self.wrist_topics["color"], self._wrist_heartbeat, qos)
        self.create_subscription(CameraInfo, self.wrist_topics["info"], self._wrist_camera_info_cb, qos)

    def _log_usage(self):
        log = self.get_logger()
        log.info(f"Saving data to: {self.root_dir}")
        log.info("Ready. Tracker Button Workflow:")
        log.info("1. Click -> SETUP (Move robot to start position)")
        log.info("2. Click -> RECORDING (Start task)")
        log.info("3. Click -> STOP (Halt robot)")
        log.info(f"4. Click (within {CONFIRMATION_TIMEOUT_SEC:.0f}s) -> CONFIRM SUCCESS")
        log.info("   No click within the window: episode is "
                 + ("kept as _failed." if KEEP_FAILED_EPISODES
                    else "DELETED (KEEP_FAILED_EPISODES=1 to keep)."))

    # --- Wrist camera discovery & health -----------------------------------
    def _resolve_wrist_camera(self):
        """Explicit presets are trusted; "auto" probes the ROS graph first."""
        if WRIST_CAMERA in WRIST_CAMERA_PRESETS:
            model, topics = WRIST_CAMERA, dict(WRIST_CAMERA_PRESETS[WRIST_CAMERA])
        else:
            if WRIST_CAMERA != "auto":
                self.get_logger().warn(
                    f"Unknown WRIST_CAMERA='{WRIST_CAMERA}', falling back to auto-detection."
                )
            model, topics = self._probe_wrist_camera()

        # Per-topic overrides always win, so odd namespaces stay usable.
        for key, env_name in WRIST_TOPIC_OVERRIDE_ENVS.items():
            override = env(env_name)
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
        return (time.time() - self._wrist_last_seen) < WRIST_LIVE_TIMEOUT_SEC

    def _wrist_camera_info_cb(self, msg: CameraInfo):
        if self._wrist_intrinsics_saved:
            return
        path = os.path.join(self.root_dir, "wrist_camera_intrinsics.json")
        _write_json(path, {
            "camera_model": self.wrist_model,
            "width": msg.width,
            "height": msg.height,
            "fx": msg.k[0], "fy": msg.k[4], "cx": msg.k[2], "cy": msg.k[5],
            "distortion_model": msg.distortion_model,
            "distortion_coeffs": list(msg.d),
        })
        self._wrist_intrinsics_saved = True
        self.get_logger().info(f"Wrist camera ({self.wrist_model}) intrinsics saved: {path}")

    # --- Bag recording -----------------------------------------------------
    def _start_bag_recording(self, episode_dir):
        bag_path = os.path.join(episode_dir, WRIST_BAG_NAME)
        yaml_path = os.path.join(episode_dir, "qos_overrides.yaml")
        color_topic = self.wrist_topics["color"]
        depth_topic = self.wrist_topics["depth"]

        with open(yaml_path, "w") as f:
            for topic in (color_topic, depth_topic):
                f.write(f"{topic}:\n  reliability: best_effort\n  history: keep_last\n  depth: 10\n")

        # Record the topic layout so extraction need not guess the camera.
        _write_json(os.path.join(episode_dir, "wrist_camera_topics.json"), {
            "model": self.wrist_model,
            "bag": WRIST_BAG_NAME,
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "info_topic": self.wrist_topics["info"],
        })

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
        if self.bag_process is None:
            return
        try:
            os.killpg(os.getpgid(self.bag_process.pid), signal.SIGINT)
            self.bag_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.bag_process.pid), signal.SIGKILL)
            self.bag_process.wait()
        except Exception:
            pass
        finally:
            self.bag_process = None

    # --- State machine -----------------------------------------------------
    def _joy_cb(self, msg: Joy):
        if len(msg.buttons) <= TRACKER_BUTTON_INDEX:
            return
        current = msg.buttons[TRACKER_BUTTON_INDEX]

        if current == 1 and self.prev_button_state == 0:
            handler = {
                State.IDLE: self._enter_setup,
                State.SETUP: self._start_recording_internal,
                State.RECORDING: self._stop_recording_internal,
                State.CONFIRMING: self._confirm_success,
            }.get(self.state)
            if handler:
                handler()

        self.prev_button_state = current

    def _preflight_problem(self):
        """(log_level, message) of the first blocking issue, else None."""
        if not self.arm_ready:
            return "warn", "Arm joints not ready!"
        if not self._is_wrist_camera_live():
            return "error", (f"❌ Wrist camera ({self.wrist_model}) not publishing on "
                             f"{self.wrist_topics['color']}! Start camera first.")
        if not self._is_hand_ft_live():
            # Zeros here are indistinguishable from "the hand touched nothing".
            return "error", (f"❌ No fingertip F/T on /{HAND_NS}/fingertip_*_broadcaster/"
                             f"[{'|'.join(self._ft_sources)}]! Refusing to record zeroed "
                             f"force data. Start the hand driver with "
                             f"fingertip_sensor:=true, then retry.")
        if self.latest_hand_joints is None:
            return "error", (f"❌ No hand joints on /{HAND_NS}/joint_states! "
                             f"Refusing to record zeroed hand state.")
        if self.latest_hand_action is None:
            # Zeros are a valid-looking hand pose, so an absent mapper would
            # otherwise be invisible until training.
            return "error", (f"❌ No hand action on {HAND_TRAJECTORY_TOPIC}! "
                             f"Refusing to record zeroed hand actions. Start "
                             f"the robot-hand mapper (start.py tesollo ...), "
                             f"then retry.")
        return None

    def _enter_setup(self):
        problem = self._preflight_problem()
        if problem:
            level, message = problem
            getattr(self.get_logger(), level)(message)
            return
        self.state = State.SETUP
        self.teleop_pub.publish(Bool(data=True))
        self.get_logger().info("[SETUP] Teleop ACTIVE. Move to start position.")
        # Announced here, while the operator is still free to arrange the box:
        # by the time recording starts the setup is already fixed.
        if self.plan:
            self.plan.announce()

    def _create_episode_directory(self):
        while True:
            ep_dir = os.path.join(self.root_dir, f"episode_{self.episode_idx:03d}")
            if not any(os.path.exists(d) for d in (ep_dir, ep_dir + "_success", ep_dir + "_failed")):
                break
            self.episode_idx += 1
        os.makedirs(ep_dir)
        return ep_dir

    def _start_recording_internal(self):
        # The arm action cannot be part of _preflight_problem: teleop_ur.py only
        # publishes while teleop is enabled, i.e. not before SETUP. By now the
        # operator has moved to the start pose with teleop active, so the topic
        # must be live - and this is the last moment refusing is still free.
        if not self._is_arm_action_live():
            self.get_logger().error(
                f"❌ No arm action on {ARM_ACTION_TOPIC} after SETUP! "
                f"Refusing to record: without it the arm action can only be "
                f"re-derived from measured joint states, which bakes the "
                f"controller's tracking lag into the dataset. Start the teleop "
                f"node (il_main/teleop_ur.py), move the tracker once, then "
                f"retry."
            )
            return

        self._ft_frames_total = 0
        self._ft_frames_zero = 0
        self._hand_action_frames_total = 0
        self._hand_action_frames_zero = 0
        self._arm_action_frames_total = 0
        self._arm_action_frames_zero = 0
        self.current_episode_dir = self._create_episode_directory()
        svo_path = os.path.join(self.current_episode_dir, "recording.svo2")

        self._start_bag_recording(self.current_episode_dir)

        params = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H265)
        with self.cam_lock:
            err = self.cam.enable_recording(params)

        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("ZED Recording failed!")
            self._stop_bag_recording()
            shutil.rmtree(self.current_episode_dir)
            self.state = State.IDLE
            self.teleop_pub.publish(Bool(data=False))
            return

        # Labels are written at START, not at finalisation: an interrupted
        # take still leaves a directory on disk, and which plan slot it
        # belonged to is only known here.
        if self.plan:
            self.plan.on_episode_start(self.current_episode_dir, self.episode_idx)

        self.state = State.RECORDING
        self.get_logger().info(f"[RECORDING] Episode {self.episode_idx} started.")

    def _stop_recording_internal(self):
        self.teleop_pub.publish(Bool(data=False))
        self.state = State.CONFIRMING

        with self.cam_lock:
            self.cam.disable_recording()
        self._stop_bag_recording()
        self._log_force_coverage()
        self._log_hand_action_coverage()
        self._log_arm_action_coverage()

        self.get_logger().info(
            f"[STOPPED] Episode {self.episode_idx}. "
            f"Press within {CONFIRMATION_TIMEOUT_SEC}s for SUCCESS."
        )

        self.pending_episode_dir = self.current_episode_dir
        self.pending_episode_idx = self.episode_idx
        self.current_episode_dir = None

        self.confirmation_timer = threading.Timer(CONFIRMATION_TIMEOUT_SEC, self._confirmation_timeout)
        self.confirmation_timer.daemon = True
        self.confirmation_timer.start()
        self.episode_idx += 1

    def _log_force_coverage(self):
        if not self._ft_frames_total:
            return
        bad = self._ft_frames_zero
        msg = (f"Force coverage: {self._ft_frames_total - bad}/{self._ft_frames_total} "
               f"frames with live F/T (source='{self._ft_source_in_use}')")
        if bad:
            self.get_logger().warn(f"{msg} - {100.0 * bad / self._ft_frames_total:.1f}% zero-filled!")
        else:
            self.get_logger().info(msg)

    def _log_hand_action_coverage(self):
        if not self._hand_action_frames_total:
            return
        total = self._hand_action_frames_total
        bad = self._hand_action_frames_zero
        msg = f"Hand action coverage: {total - bad}/{total} frames commanded"
        if bad:
            self.get_logger().warn(
                f"{msg} - {100.0 * bad / total:.1f}% zero-filled! "
                f"Is the robot-hand mapper running?"
            )
        else:
            self.get_logger().info(msg)

    def _log_arm_action_coverage(self):
        if not self._arm_action_frames_total:
            return
        total = self._arm_action_frames_total
        bad = self._arm_action_frames_zero
        msg = f"Arm action coverage: {total - bad}/{total} frames commanded"
        if bad:
            self.get_logger().warn(
                f"{msg} - {100.0 * bad / total:.1f}% held/zero-filled! "
                f"Is {ARM_ACTION_TOPIC} publishing?"
            )
        else:
            self.get_logger().info(msg)

    def _confirm_success(self):
        if self.confirmation_timer:
            self.confirmation_timer.cancel()
        self._finalize_episode("success", "✅ Episode {} -> SUCCESS")

    def _confirmation_timeout(self):
        if KEEP_FAILED_EPISODES:
            self._finalize_episode("failed", "❌ Episode {} -> FAILED (timeout)")
        else:
            self._discard_episode()

    def _finalize_episode(self, suffix, log_fmt):
        if self.pending_episode_dir and os.path.exists(self.pending_episode_dir):
            final_dir = os.path.join(
                self.root_dir,
                f"episode_{self.pending_episode_idx:03d}_{suffix}")
            os.rename(self.pending_episode_dir, final_dir)
            self.get_logger().info(log_fmt.format(self.pending_episode_idx))
            # After the rename, so the plan sees the final directory name -
            # that name is what decides whether the slot counts as done.
            if self.plan:
                self.plan.on_episode_final(final_dir)
        self.pending_episode_dir = None
        self.state = State.IDLE

    def _discard_episode(self):
        """Delete the pending episode and reuse its index for the next take.

        Reusing the index keeps the numbering gap-free, so the episode count
        always matches the highest index - otherwise a discarded take would
        leave a hole and make the recording plan hard to follow.
        """
        ep_dir = self.pending_episode_dir
        idx = self.pending_episode_idx
        self.pending_episode_dir = None

        if ep_dir and os.path.exists(ep_dir):
            try:
                shutil.rmtree(ep_dir)
                self.get_logger().info(
                    f"🗑  Episode {idx} -> DISCARDED (timeout). "
                    f"Index {idx} will be reused."
                )
                if idx is not None:
                    self.episode_idx = idx
                if self.plan:
                    self.plan.on_episode_discarded(ep_dir)
            except OSError as exc:
                self.get_logger().error(
                    f"Could not delete {ep_dir}: {exc}. Left on disk."
                )
        self.state = State.IDLE

    # --- Sensor callbacks --------------------------------------------------
    def _gripper_cb(self, msg: JointState):
        """Tesollo hand joints (20 DoF), radians -> degrees for schema compat."""
        by_name = dict(zip(msg.name, msg.position))
        missing = [n for n in HAND_JOINT_NAMES if n not in by_name]
        if missing:
            # Log once; at 100+ Hz a per-message warning would flood.
            if not self._hand_joint_warned:
                self._hand_joint_warned = True
                self.get_logger().warn(
                    f"/{HAND_NS}/joint_states missing {len(missing)} expected "
                    f"joints (e.g. {missing[:3]}); recording zeros for those. "
                    f"Received: {sorted(by_name)[:5]}..."
                )
            return

        positions = np.array([by_name[n] for n in HAND_JOINT_NAMES], dtype=np.float32)
        self.latest_hand_joints = np.degrees(positions) if HAND_JOINTS_IN_DEGREES else positions

    def _hand_action_cb(self, msg: JointTrajectory):
        """Commanded hand pose from the robot-hand mapper (radians -> degrees).

        Indexed by name for the same reason as _gripper_cb: the publisher's
        joint order is not guaranteed, and a silent permutation would be
        invisible in the recorded arrays.
        """
        if not msg.points:
            return

        by_name = dict(zip(msg.joint_names, msg.points[0].positions))
        missing = [n for n in HAND_JOINT_NAMES if n not in by_name]
        if missing:
            if not self._hand_action_warned:
                self._hand_action_warned = True
                self.get_logger().warn(
                    f"{HAND_TRAJECTORY_TOPIC} missing {len(missing)} expected "
                    f"joints (e.g. {missing[:3]}); ignoring these messages. "
                    f"Received: {sorted(by_name)[:5]}..."
                )
            return

        positions = np.array([by_name[n] for n in HAND_JOINT_NAMES], dtype=np.float32)
        self.latest_hand_action = (np.degrees(positions) if HAND_JOINTS_IN_DEGREES
                                   else positions)

    def _arm_action_cb(self, msg: PoseStamped):
        """Commanded tool0 pose in base_link from the teleop node.

        Stored verbatim: no FK, no frame conversion, no differencing. The
        quantity on this topic is already the one the deploy-time executor
        consumes on /target_frame, and every transformation applied here would
        have to be inverted there.
        """
        # The frame is REPORTED but no longer a reason to drop the message.
        # Dropping made the recorder unstartable whenever the teleop node
        # labelled its poses differently than expected: the messages were
        # discarded, latest_arm_action stayed None, and the preflight check
        # reported "no arm action" while the topic was visibly publishing -
        # with the actual cause logged once, far earlier in the scrollback.
        # A mislabelled frame is now recorded and flagged; an unstartable
        # recorder loses the take outright, which is the worse failure.
        # The frame is written into the episode metadata (see _metadata), so
        # a wrong one stays detectable after the fact instead of being
        # silently accepted here.
        frame = msg.header.frame_id.lstrip("/")
        if frame and frame != ARM_ACTION_FRAME:
            if not self._arm_action_frame_warned:
                self._arm_action_frame_warned = True
                self.get_logger().warn(
                    f"{ARM_ACTION_TOPIC} is in frame '{frame}', expected "
                    f"'{ARM_ACTION_FRAME}'. Recording anyway. Verify the "
                    f"teleop frame before training: an arm action in the "
                    f"wrong frame trains silently and fails at deploy. Set "
                    f"ARM_ACTION_FRAME to silence this."
                )
        self._arm_action_frame_seen = frame or None

        p, q = msg.pose.position, msg.pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if not (0.9 <= norm <= 1.1):
            self.get_logger().warn(
                f"{ARM_ACTION_TOPIC} quaternion norm {norm:.3f} is not unit; "
                f"dropping this message.", throttle_duration_sec=2.0)
            return
        quat /= norm

        # Hemisphere continuity: q and -q are the same rotation, but a sign
        # flip between consecutive frames looks like a 360 deg snap to anything
        # that interpolates or differences the recorded stream.
        if (self.latest_arm_action is not None
                and float(np.dot(quat, self.latest_arm_action[3:7])) < 0.0):
            quat = -quat

        self.latest_arm_action = np.array(
            [p.x, p.y, p.z, quat[0], quat[1], quat[2], quat[3]],
            dtype=np.float32)
        self._arm_action_last_seen = time.time()

    def _is_arm_action_live(self) -> bool:
        return (self.latest_arm_action is not None
                and (time.time() - self._arm_action_last_seen)
                < ARM_ACTION_LIVE_TIMEOUT_SEC)

    def _fingertip_cb(self, msg: WrenchStamped, finger_idx: int, source: str):
        """[fx, fy, fz, tx, ty, tz] per finger, fingers 1-5 (old F/T ordering)."""
        w = msg.wrench
        parts = self._hand_ft_parts[source]
        parts[finger_idx] = (w.force.x, w.force.y, w.force.z,
                             w.torque.x, w.torque.y, w.torque.z)

        # Publish only once every finger reported, never mixing fresh readings
        # with uninitialised zeros.
        if any(p is None for p in parts):
            return
        self._ft_last_full[source] = time.time()

        # Highest-preference source that actually delivers wins: a silent topic
        # cannot shadow a live one, but is picked up as soon as it starts.
        if source != self._preferred_ft_source():
            return
        if self._ft_source_in_use != source:
            self._ft_source_in_use = source
            self.get_logger().info(f"Fingertip F/T now sourced from '{source}'.")

        self.latest_hand_ft = np.array([v for part in parts for v in part], dtype=np.float32)

    def _preferred_ft_source(self):
        now = time.time()
        for source in self._ft_sources:
            if (now - self._ft_last_full[source]) < FT_LIVE_TIMEOUT_SEC:
                return source
        return None

    def _is_hand_ft_live(self) -> bool:
        return self._preferred_ft_source() is not None

    def _joint_cb(self, msg: JointState):
        """UR arm joint states; drives the recording tick."""
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest_arm_state:
                self.latest_arm_state[name] = pos

        if not self.arm_ready and all(self.latest_arm_state[n] is not None for n in UR_JOINT_NAMES):
            self.arm_ready = True
            self.get_logger().info("Arm joints ready.")

        if self.state != State.RECORDING or not self.arm_ready:
            return

        # Schema must match extract_svo_force.py (SVO key "robot_state").
        state_entry = {
            "timestamp_ns": msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
            "arm_joints": {
                "names": UR_JOINT_NAMES,
                "positions": [self.latest_arm_state[n] if self.latest_arm_state[n] is not None
                              else 0.0 for n in UR_JOINT_NAMES],
            },
            "hand_joints": (self.latest_hand_joints.tolist()
                            if self.latest_hand_joints is not None else [0.0] * HAND_NUM_JOINTS),
            "hand_ft": (self.latest_hand_ft.tolist()
                        if self.latest_hand_ft is not None else [0.0] * HAND_NUM_FT),
            # Commanded tool0 pose in base_link. Held at the last command when
            # the operator is still (teleop_ur.py has a deadband and does not
            # republish), which is correct: "stay here" IS the command. An
            # all-zero entry means the topic never published at all - the
            # identity quaternion below keeps it an invalid pose rather than a
            # plausible one at the origin.
            "arm_action": (self.latest_arm_action.tolist()
                           if self.latest_arm_action is not None
                           else [0.0] * ARM_ACTION_DIM),
            # Commanded hand pose. Zero-filled when the mapper is not running,
            # which the coverage counter below reports.
            "hand_action": (self.latest_hand_action.tolist()
                            if self.latest_hand_action is not None
                            else [0.0] * HAND_NUM_JOINTS),
        }

        self._arm_action_frames_total += 1
        if self.latest_arm_action is None:
            self._arm_action_frames_zero += 1
            self.get_logger().warn(
                f"No arm action on {ARM_ACTION_TOPIC} - recording zeros "
                f"for this frame.",
                throttle_duration_sec=2.0,
            )

        self._hand_action_frames_total += 1
        if self.latest_hand_action is None:
            self._hand_action_frames_zero += 1
            self.get_logger().warn(
                f"No hand action on {HAND_TRAJECTORY_TOPIC} - recording zeros "
                f"for this frame.",
                throttle_duration_sec=2.0,
            )

        # Track force coverage so dropouts show up in the log instead of hiding
        # as plausible-looking zeros in the dataset.
        self._ft_frames_total += 1
        if self.latest_hand_ft is None or not self._is_hand_ft_live():
            self._ft_frames_zero += 1
            self.get_logger().warn(
                "Fingertip F/T stale - recording zeros for this frame.",
                throttle_duration_sec=2.0,
            )

        with self.pending_lock:
            self.pending_states_queue.append(state_entry)

    # --- ZED grab loop -----------------------------------------------------
    def _grab_loop(self):
        runtime = sl.RuntimeParameters(
            confidence_threshold=100, remove_saturated_areas=True, texture_confidence_threshold=100
        )
        while self.running and rclpy.ok():
            with self.cam_lock:
                # self.running can clear while we waited for the lock, and the
                # camera may already be closing.
                if not self.running:
                    return
                if self.cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                if self.state != State.RECORDING:
                    continue

                with self.pending_lock:
                    queued_states = self.pending_states_queue
                    self.pending_states_queue = []

                if queued_states:
                    data = sl.SVOData()
                    data.key = "robot_state"
                    data.set_string_content(json.dumps(queued_states))
                    data.timestamp_ns = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                    self.cam.ingest_data_into_svo(data)


def _shutdown(node):
    """Release the ZED and the bag process, whatever else fails on the way.

    Each step is guarded separately: after Ctrl+C the rclpy context is already
    invalid, so ROS calls here raise. Previously publish() was the first
    statement, its exception propagated out of the finally block and cam.close()
    never ran - leaving the camera locked by a dead process, so the *next*
    launch failed with "camera could not be initialized".
    """
    log = node.get_logger()

    # Stop the robot first: teleop must not stay enabled after we exit.
    if rclpy.ok():
        try:
            node.teleop_pub.publish(Bool(data=False))
        except Exception as exc:
            log.warn(f"Could not publish teleop disable: {exc}")

    # Close out an in-flight episode so SVO and bag are not left truncated.
    try:
        if node.state == State.RECORDING:
            node._stop_recording_internal()
        if node.state == State.CONFIRMING:
            node._confirmation_timeout()
    except Exception as exc:
        log.warn(f"Episode finalisation failed: {exc}")

    # After finalisation, so the closing progress report counts the episode
    # that was still running when Ctrl+C arrived.
    try:
        if node.plan:
            node.plan.shutdown()
    except Exception as exc:
        log.warn(f"Plan shutdown failed: {exc}")

    # The grab thread touches self.cam, so it must be gone before close().
    # The old 1s join could return with the thread still inside cam.grab(),
    # which then segfaulted on the closed camera.
    node.running = False
    node.grab_thread.join(timeout=5.0)
    if node.grab_thread.is_alive():
        log.error("Grab thread still running; skipping cam.close() to avoid a "
                  "segfault. The camera is released when this process exits.")
    else:
        try:
            with node.cam_lock:
                node.cam.close()
        except Exception as exc:
            log.warn(f"ZED close failed: {exc}")

    # Never leave an orphaned `ros2 bag record` behind.
    try:
        node._stop_bag_recording()
    except Exception:
        pass

    try:
        node.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = SVORecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(node)


if __name__ == "__main__":
    main()