#!/usr/bin/env python3
# data/record_extractor/extract_episodes.py
"""
Extract episodes from SVO + ROS bags.

Generates: front_rgb/, front_depth/, front_pc/, wrist_rgb/, state.npy,
force.npy, frame_timestamps_ns.npy, sync_gaps.npy.

Front camera resolutions (ZED, sl.RESOLUTION.HD720):
  * front_pc/ is built from the NATIVE 1280x720 image + depth with the native
    intrinsics, BEFORE any resize.
  * front_rgb/ and front_depth/ are stored at 640x360 - exactly half of HD720,
    so a single isotropic factor 0.5 relates the stored images to the native
    intrinsics in both axes. No anisotropic squash to 4:3 any more.
  * front_depth/ is a 16-bit PNG in MILLIMETERS, pixel-aligned with front_rgb.
    0 means "no measurement" (NaN/Inf from the ZED, or out of range).

The wrist camera contributes RGB only; wrist point clouds are not generated.

The wrist bag starts a little after the ZED (the recorder does not block on
it, so start/stop stays instant). Leading frames with no wrist counterpart are
detected and trimmed; see extract_wrist_camera() and trim_episode_start().

Quality assessment lives in qa_report.py and runs inline, writing qa_report.*
per episode plus qa_summary.* for the collection. Disable with --skip_qa.
"""

import os
import sys
import cv2
import json
import glob
import shutil
import argparse
import numpy as np
import pyzed.sl as sl
from termcolor import cprint
from tqdm import tqdm

for p in ("/root/ros2_ws/data/recording/record_utils", "/root/ros2_ws/data/record_utils"):
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from point_cloud_utils import generate_point_cloud_from_depth

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    ROSBAGS_AVAILABLE = True
except ImportError:
    ROSBAGS_AVAILABLE = False

from camera_info import (
    camera_intrinsics as FRONT_INTRINSICS,
    camera_extrinsics as FRONT_EXTRINSICS,
    workspace as WORKSPACE,
    FRONT_IMAGE_SIZE,
)

# All quality assessment lives in qa_report.py to keep this file focused on
# extraction. It runs inline on the arrays already in memory, so it adds no
# extra SVO or bag reads.
import qa_report

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

# Recorded schema (record_utils/record.py, SVO key "robot_state").
ARM_DIM = 6
HAND_JOINT_DIM = 20

# Front RGB/depth storage size, (W, H). Half of HD720, derived in camera_info
# next to the halved intrinsics so the two cannot drift apart.
FRONT_SAVE_W, FRONT_SAVE_H = FRONT_IMAGE_SIZE

# uint16 PNG in mm tops out here. The ZED is capped at depth_maximum_distance
# (2.0 m) anyway, so this only guards against a stray huge value wrapping
# around into a small, plausible-looking depth.
DEPTH_UINT16_MAX = 65535
HAND_FT_DIM = 30            # 5 fingertips x [fx, fy, fz, tx, ty, tz]
STATE_DIM = ARM_DIM + HAND_JOINT_DIM

# Commanded hand pose from the teleoperation mapper, same units and joint order
# as the hand block of state.npy (see record_utils/record.py "hand_action").
# Written to hand_action.npy so the dataset builder can use the COMMANDED hand
# pose instead of deriving the action from the measured next state.
HAND_ACTION_DIM = HAND_JOINT_DIM

# Commanded tool0 pose in base_link from the teleop node (record.py
# "arm_action"): xyz + quat xyzw. Written to arm_action.npy so the dataset
# builder can use the COMMANDED arm pose instead of FK of the measured next
# state, which carries the controller's tracking lag.
ARM_ACTION_DIM = 7


def _fixed_len(values, expected, field, warn_state):
    """Coerce a recorded vector to `expected` length.

    A short/long vector means the recorder and extractor disagree on the
    schema. Truncating silently would scramble the per-finger F/T layout, so
    warn once and zero-pad instead of guessing.
    """
    arr = np.asarray(values, dtype=np.float32).ravel()
    if arr.shape[0] == expected:
        return arr

    if field not in warn_state:
        warn_state.add(field)
        cprint(f"  [QA] '{field}' has {arr.shape[0]} values, expected "
               f"{expected}; padding/truncating.", "yellow")

    out = np.zeros(expected, dtype=np.float32)
    n = min(expected, arr.shape[0])
    out[:n] = arr[:n]
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Extract episodes from SVO + ROS bags")
    default_data_dir = (
        f'/root/ros2_ws/data/recording/episodes/external/{FOLDER_NAME}'
        if os.path.exists(f'/root/ros2_ws/data/recording/episodes/external/{FOLDER_NAME}')
        else f'/root/ros2_ws/data/episodes/external/{FOLDER_NAME}'
    )
    parser.add_argument('--data_dir', type=str,
                        default=default_data_dir)
    parser.add_argument('--target_fps', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=1500,
                        help='Max SVO frames (60fps). Episodes above are deleted.')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Points per point cloud')
    parser.add_argument('--skip_point_clouds', action='store_true',
                        help='Skip front_pc/ generation (only RGB + depth + state)')
    parser.add_argument('--wrist_color_topic', type=str,
                        default='/cam_gemini/color/image_raw')
    parser.add_argument('--include_failed', type=bool,
                        default=False)
    parser.add_argument('--wrist_max_sync_gap_ms', type=float, default=40.0,
                        help='Report INTERIOR wrist frames whose nearest match '
                             'is farther than this (ms), i.e. dropped messages '
                             'mid-episode. <=0 disables the check. The startup '
                             'ramp is detected separately and exactly.')
    parser.add_argument('--trim_stale_start', action='store_true', default=True,
                        help='Drop leading frames recorded before the wrist bag '
                             'started, so frame 0 is the first instant both '
                             'cameras saw. On by default.')
    parser.add_argument('--no_trim_stale_start', dest='trim_stale_start',
                        action='store_false',
                        help='Keep the stale startup frames (not recommended).')
    parser.add_argument('--skip_qa', action='store_true',
                        help='Skip the inline quality assessment entirely.')
    parser.add_argument('--no_qa_plots', action='store_true',
                        help='Run QA but skip its plots (faster).')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Wrist Camera (ROS Bag)
# ═══════════════════════════════════════════════════════════════

def decode_image_msg(msg, conn):
    """Decode sensor_msgs/Image or CompressedImage → numpy array."""
    if 'CompressedImage' in conn.msgtype:
        img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    encoding = msg.encoding
    h, w = msg.height, msg.width

    dtype_map = {
        '16UC1': (np.uint16, (h, w)),
        'mono16': (np.uint16, (h, w)),
        '32FC1': (np.float32, (h, w)),
        'mono8': (np.uint8, (h, w)),
        'rgb8': (np.uint8, (h, w, 3)),
        'bgr8': (np.uint8, (h, w, 3)),
        'bgra8': (np.uint8, (h, w, 4)),
        'rgba8': (np.uint8, (h, w, 4)),
    }

    if encoding in dtype_map:
        dtype, shape = dtype_map[encoding]
        img = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(shape)
    else:
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, -1)

    if encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == 'bgra8':
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif encoding == 'rgba8':
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img


def get_header_stamp_ns(msg, fallback_ts):
    """Extract header timestamp in nanoseconds."""
    try:
        stamp_ns = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        if stamp_ns > 0:
            return stamp_ns
    except Exception:
        pass
    return fallback_ts


def _wrist_result(ok, **kw):
    """Uniform return payload for extract_wrist_camera()."""
    base = {"ok": ok, "available": False, "n_msgs": 0, "period_ms": None,
            "lead_ms": None, "n_stale_start": 0, "trimmed_frames": 0,
            "gaps_ms": []}
    base.update(kw)
    return base


def extract_wrist_camera(episode_dir, output_timestamps_ns, args):
    """Extract wrist RGB from rosbag, synced to ZED timestamps.

    Returns a metrics dict (see _wrist_result) so the QA module can report on
    the sync without re-reading the bag.
    """
    if not ROSBAGS_AVAILABLE:
        cprint("  [Error] 'rosbags' not installed.", "red")
        return _wrist_result(False)

    bag_path = os.path.join(episode_dir, 'realsense_bag')
    if not os.path.exists(bag_path):
        cprint("  [Info] No realsense_bag found, skipping wrist.", "white")
        return _wrist_result(False)

    wrist_rgb_dir = os.path.join(episode_dir, 'wrist_rgb')
    os.makedirs(wrist_rgb_dir, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_JAZZY)

    # Read all messages
    color_entries = []

    with Reader(bag_path) as reader:
        color_connections = [c for c in reader.connections if c.topic == args.wrist_color_topic]

        if not color_connections:
            available = sorted(set(c.topic for c in reader.connections))
            cprint(f"  [Error] Topic '{args.wrist_color_topic}' not found.", "red")
            cprint(f"  Available: {available}", "yellow")
            return _wrist_result(False)

        for connection, timestamp, rawdata in reader.messages(connections=color_connections):
            color_entries.append((timestamp, rawdata, connection))

    color_entries.sort(key=lambda x: x[0])

    # Max tolerated ZED-to-wrist gap. One wrist frame period (~33 ms at 30 fps)
    # is the natural bound; beyond that the nearest match is a different
    # moment in time, not the same instant.
    max_gap_ms = args.wrist_max_sync_gap_ms

    if not color_entries:
        cprint("  [Error] No wrist color messages found.", "red")
        return _wrist_result(False)

    # Header timestamps
    color_timestamps_ns = []
    for ts, rawdata, conn in color_entries:
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
        color_timestamps_ns.append(get_header_stamp_ns(msg, ts))
    color_timestamps_ns = np.array(color_timestamps_ns, dtype=np.int64)

    # Sync to ZED timestamps
    output_ts = np.array(output_timestamps_ns, dtype=np.int64)
    sync_gaps_ms = []
    stale_frames = []

    # --- Leading frames the wrist camera never saw -------------------
    # The bag starts slightly after the ZED by design (the recorder does not
    # block on it, so start/stop stays instant). Any ZED frame stamped before
    # the FIRST wrist message therefore has no counterpart, and np.argmin
    # silently clamps it onto message #0 -> identical duplicate images.
    #
    # This is a geometric fact about the timestamps, NOT a tunable. A 40 ms
    # threshold missed the last frame of every ramp: with a 50 ms ZED period
    # the final stale gap is always < 50 ms (34.4 ms in episode_001) while
    # being just as stale as the 134 ms one.
    first_wrist_ts = int(color_timestamps_ns[0])
    n_stale_start = int(np.searchsorted(output_ts, first_wrist_ts, side='left'))

    if n_stale_start >= len(output_ts):
        cprint(f"  [Error] All {len(output_ts)} ZED frames precede the first "
               f"wrist message - bag and SVO do not overlap. Refusing to "
               f"trim.", "red")
        return _wrist_result(False, n_stale_start=n_stale_start)

    for idx, target_ts in enumerate(tqdm(output_ts, desc="  Syncing wrist", unit="frame")):
        # Color
        closest_idx = np.argmin(np.abs(color_timestamps_ns - target_ts))
        gap_ms = abs(int(color_timestamps_ns[closest_idx]) - int(target_ts)) / 1e6
        sync_gaps_ms.append(gap_ms)

        # argmin always returns something, so a ZED frame recorded before the
        # bag started clamps onto the first wrist message. Those frames are
        # stale duplicates, not real matches - record them for the report.
        if max_gap_ms > 0 and gap_ms > max_gap_ms:
            stale_frames.append((idx, gap_ms))

        _, rawdata, conn = color_entries[closest_idx]
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
        color_img = decode_image_msg(msg, conn)
        cv2.imwrite(os.path.join(wrist_rgb_dir, f"frame_{idx:04d}.png"), color_img)

    # Report sync quality over frames that actually have a wrist match, so the
    # startup ramp does not skew mean/max.
    qa_report.sync_report(episode_dir, sync_gaps_ms[n_stale_start:],
                          label="ZED-to-Wrist")

    if n_stale_start > 0:
        lead_ms = (first_wrist_ts - int(output_ts[0])) / 1e6
        cprint(f"  [QA] Wrist bag began {lead_ms:.1f}ms after the ZED -> "
               f"{n_stale_start} leading frame(s) without a wrist match:",
               "yellow")
        for idx in range(n_stale_start):
            cprint(f"         frame_{idx:04d}  gap {sync_gaps_ms[idx]:7.1f} ms",
                   "yellow")

    # Interior spikes are a DIFFERENT failure (wrist messages dropped
    # mid-episode). Never trimmed: removing them would splice non-adjacent
    # moments into a pretend-continuous trajectory.
    interior = [(i, g) for i, g in stale_frames if i >= n_stale_start]
    if interior:
        cprint(f"  [QA] {len(interior)} INTERIOR frame(s) exceed "
               f"{max_gap_ms:.0f}ms (dropped wrist messages, NOT trimmed):",
               "red")
        for idx, gap_ms in interior[:8]:
            cprint(f"         frame_{idx:04d}  gap {gap_ms:7.1f} ms", "red")

    # Superseded by actual trimming - leave no unread marker behind.
    legacy = os.path.join(episode_dir, 'wrist_stale_frames.npy')
    if os.path.exists(legacy):
        os.remove(legacy)

    period_ms = (float(np.median(np.diff(color_timestamps_ns)) / 1e6)
                 if len(color_timestamps_ns) > 1 else None)

    return _wrist_result(
        True,
        available=True,
        n_msgs=int(len(color_timestamps_ns)),
        period_ms=period_ms,
        lead_ms=float((first_wrist_ts - int(output_ts[0])) / 1e6),
        n_stale_start=n_stale_start,
        gaps_ms=sync_gaps_ms,
    )


# ═══════════════════════════════════════════════════════════════
# Startup Trim
# ═══════════════════════════════════════════════════════════════

FRAME_DIRS = (
    ('front_rgb', '*.png'),
    ('front_depth', '*.png'),
    ('front_pc', '*.npy'),
    ('wrist_rgb', '*.png'),
)

# Per-frame arrays. sync_gaps.npy is deliberately absent: it has one entry per
# robot_state update, not per output frame, so slicing it would corrupt it.
FRAME_ARRAYS = ('state.npy', 'force.npy', 'hand_action.npy',
                'arm_action.npy', 'frame_timestamps_ns.npy')


def trim_episode_start(episode_dir, n_drop):
    """Drop the first `n_drop` output frames from every aligned artifact.

    Only ever a contiguous run starting at frame 0 (the startup ramp), so the
    remaining frames stay temporally adjacent. Files are renumbered from 0 to
    keep counts consistent for verify_episode() and the dataset builder.
    """
    if n_drop <= 0:
        return True

    marker = os.path.join(episode_dir, 'trim_applied.json')
    if os.path.exists(marker):
        # Guard against double-trimming on a re-run.
        cprint(f"  [Trim] Already trimmed, skipping.", "white")
        return True

    # Slice arrays first; if anything is inconsistent we bail before renaming.
    for name in FRAME_ARRAYS:
        path = os.path.join(episode_dir, name)
        if not os.path.exists(path):
            continue
        arr = np.load(path)
        if len(arr) <= n_drop:
            cprint(f"  [Trim] {name} has {len(arr)} frames, cannot drop "
                   f"{n_drop}. Aborting trim.", "red")
            return False

    for name in FRAME_ARRAYS:
        path = os.path.join(episode_dir, name)
        if os.path.exists(path):
            np.save(path, np.load(path)[n_drop:])

    # Renumber frame files. Ascending order is safe: the target index is always
    # lower than the source, so no rename can clobber a pending file.
    for dirname, pattern in FRAME_DIRS:
        d = os.path.join(episode_dir, dirname)
        if not os.path.isdir(d):
            continue
        files = sorted(glob.glob(os.path.join(d, pattern)))
        if len(files) <= n_drop:
            continue
        ext = os.path.splitext(files[0])[1]
        for p in files[:n_drop]:
            os.remove(p)
        for new_idx, src in enumerate(files[n_drop:]):
            dst = os.path.join(d, f"frame_{new_idx:04d}{ext}")
            if src != dst:
                os.replace(src, dst)

    with open(marker, 'w') as f:
        json.dump({'frames_dropped': int(n_drop),
                   'reason': 'wrist bag started after ZED'}, f, indent=2)

    cprint(f"  [Trim] Dropped {n_drop} stale leading frame(s); "
           f"frame 0 is now the first instant both cameras saw.", "green")
    return True


# ═══════════════════════════════════════════════════════════════
# ZED SVO Extraction (Front Camera)
# ═══════════════════════════════════════════════════════════════

def sanitize_depth_m(depth_m):
    """Replace the ZED's invalid markers with the sentinel 0.

    retrieve_measure(MEASURE.DEPTH) returns float32 meters and encodes failures
    as NaN (no measurement) and +/-Inf (too close / too far). These MUST go
    before anything else touches the array:
      * a single NaN poisons a whole pixel neighbourhood on any averaging
        resize,
      * NaN/Inf survive into the later normalization and wipe out the entire
        tensor.
    0 is the sentinel because it is also the natural "no data" value of the
    16-bit PNG we store, so the convention holds end to end.
    """
    out = np.asarray(depth_m, dtype=np.float32).copy()
    out[~np.isfinite(out)] = 0.0
    out[out < 0.0] = 0.0
    return out


def save_depth_png_mm(path, depth_m):
    """Store a float32 meter depth map as a 16-bit millimeter PNG.

    ~8x smaller than float32 NPY, lossless at mm resolution, unambiguous out to
    ~65 m. Clipping happens BEFORE the uint16 cast: otherwise an out-of-range
    value wraps around and becomes an innocent-looking wrong depth.
    """
    depth_mm = np.rint(np.asarray(depth_m, dtype=np.float32) * 1000.0)
    depth_mm = np.clip(depth_mm, 0, DEPTH_UINT16_MAX)
    cv2.imwrite(path, depth_mm.astype(np.uint16))


def extract_episode(episode_dir, args, source_fps = 60):
    """
    Extract single episode from SVO.
    Saves: front_rgb/, front_depth/, front_pc/, state.npy, force.npy.
    """
    
    skip_factor = source_fps // args.target_fps

    svo_path = os.path.join(episode_dir, "recording.svo2")
    if not os.path.exists(svo_path):
        cprint(f"  No recording.svo2 found, skipping.", "yellow")
        return False

    # Init ZED
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER 
            
    init.depth_minimum_distance = 0.2
    init.depth_maximum_distance = 2.0
    init.sdk_verbose = 0

    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        cprint(f"  Failed to open SVO.", "red")
        return False

    num_frames = cam.get_svo_number_of_frames()

    # Delete if too long
    if num_frames > args.max_frames:
        cam.close()
        cprint(f"  [DELETE] {num_frames} frames > {args.max_frames} limit.", "red")
        shutil.rmtree(episode_dir)
        return False

    # Output dirs
    front_rgb_dir = os.path.join(episode_dir, 'front_rgb')
    front_depth_dir = os.path.join(episode_dir, 'front_depth')
    front_pc_dir = os.path.join(episode_dir, 'front_pc')
    os.makedirs(front_rgb_dir, exist_ok=True)
    os.makedirs(front_depth_dir, exist_ok=True)
    if not args.skip_point_clouds:
        os.makedirs(front_pc_dir, exist_ok=True)

    # ZED mats
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 100
    runtime.texture_confidence_threshold = 100
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    # Buffers
    episode_states = []
    episode_forces = []
    episode_hand_actions = []
    episode_arm_actions = []
    sync_gaps_ms = []
    output_timestamps_ns = []
    last_known_positions = np.zeros(STATE_DIM, dtype=np.float32)
    last_known_forces = np.zeros(HAND_FT_DIM, dtype=np.float32)
    last_known_hand_action = np.zeros(HAND_ACTION_DIM, dtype=np.float32)
    last_known_arm_action = np.zeros(ARM_ACTION_DIM, dtype=np.float32)
    dropped_state_count = 0
    # Leading output frames written before the FIRST robot_state arrived. The
    # SVO starts recording a moment before record.py enters RECORDING, so the
    # first few frames have no state to attach and keep the last_known_*
    # zeros: an arm at the origin with a zero-norm quaternion, zero force and
    # zero hand pose. Those are not missing values, they are plausible-looking
    # measurements, and the step from them to the first real pose appears as a
    # physically impossible jump (~2 rad in one frame).
    #
    # Counted separately from dropped_state_count because only a contiguous
    # run at the START may be trimmed: a hole in the middle has to stay, since
    # dropping it would splice non-adjacent frames into neighbours.
    n_stale_state_start = 0
    seen_any_state = False
    # Counted like hand_action: an episode recorded without the teleop node
    # publishing its commanded pose yields an all-zero arm_action.npy, which
    # loads and trains fine but carries no command signal.
    arm_action_seen_count = 0
    # Counted like force: an episode recorded without the hand mapper running
    # yields an all-zero hand_action.npy, which trains without error but
    # carries no command signal.
    hand_action_seen_count = 0
    # Force is tracked separately from state: hand_ft can be missing or
    # all-zero even when arm/hand joints update fine (e.g. the fingertip
    # broadcasters were not publishing during recording).
    force_seen_count = 0
    schema_warned = set()
    last_ts = sl.Timestamp()
    last_ts.set_nanoseconds(0)

    expected_output = num_frames // skip_factor
    cprint(f"\n--- {os.path.basename(episode_dir)} "
           f"({num_frames}@60fps → {expected_output}@{args.target_fps}fps) ---", "cyan")

    output_idx = 0

    with tqdm(total=num_frames, desc="  Extracting", unit="frame") as pbar:
        for frame_idx in range(num_frames):
            err = cam.grab(runtime)

            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                pbar.update(1)
                continue

            current_ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            cam_ts_ns = current_ts.get_nanoseconds()

            # Read joint states from SVO custom data
            custom_data = {}
            state_updated = False
            res = cam.retrieve_svo_data("robot_state", custom_data, last_ts, current_ts)

            if res == sl.ERROR_CODE.SUCCESS and len(custom_data) > 0:
                latest_key = sorted(custom_data.keys(),
                                    key=lambda t: t.get_nanoseconds())[-1]
                svo_content = custom_data[latest_key]
                json_string = (svo_content.get_content_as_string()
                               if hasattr(svo_content, 'get_content_as_string')
                               else svo_content.get_string_content())
                try:
                    batch = json.loads(json_string)
                    if len(batch) > 0:
                        chosen = batch[-1]
                        arm_positions = _fixed_len(
                            chosen['arm_joints']['positions'], ARM_DIM,
                            'arm_joints', schema_warned)
                        hand_joints = _fixed_len(
                            chosen.get('hand_joints', []), HAND_JOINT_DIM,
                            'hand_joints', schema_warned)
                        last_known_positions = np.concatenate(
                            [arm_positions, hand_joints])

                        # Carry the previous reading forward when hand_ft is
                        # absent, so a gap does not inject a zero spike.
                        # Carry forward like hand_ft: a missing key must not
                        # inject a zero spike into the commanded trajectory.
                        if 'hand_action' in chosen:
                            last_known_hand_action = _fixed_len(
                                chosen['hand_action'], HAND_ACTION_DIM,
                                'hand_action', schema_warned)
                            if np.any(last_known_hand_action):
                                hand_action_seen_count += 1

                        if 'arm_action' in chosen:
                            last_known_arm_action = _fixed_len(
                                chosen['arm_action'], ARM_ACTION_DIM,
                                'arm_action', schema_warned)
                            if np.any(last_known_arm_action):
                                arm_action_seen_count += 1

                        if 'hand_ft' in chosen:
                            last_known_forces = _fixed_len(
                                chosen['hand_ft'], HAND_FT_DIM,
                                'hand_ft', schema_warned)
                            if np.any(last_known_forces):
                                force_seen_count += 1

                        state_updated = True
                        gap_ms = abs(cam_ts_ns - chosen['timestamp_ns']) / 1e6
                        sync_gaps_ms.append(gap_ms)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    cprint(f"  [QA] Malformed robot_state at frame "
                           f"{frame_idx}: {e}", "yellow")

            last_ts = current_ts

            # Output frame (at target FPS)
            if frame_idx % skip_factor == 0:
                # --- Order of operations is load-bearing -----------------
                # 1. RGB + depth from the SAME grab(), both referred to the
                #    LEFT view. retrieve_measure(DEPTH) is already registered
                #    to the left image, so no extra alignment step is needed
                #    and the two are pixel-coincident by construction.
                # 2. Normalize invalid depth to the 0 sentinel.
                # 3. Point cloud from the NATIVE-resolution pair.
                # 4. Only then resize for storage.
                cam.retrieve_image(image_mat, sl.VIEW.LEFT)
                color_bgr = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)

                cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth_m = sanitize_depth_m(depth_mat.get_data())

                # Point cloud: native 1280x720 + native intrinsics, strictly
                # BEFORE any resize. The stored images live at 640x360, related
                # to the native frame by the isotropic factor 0.5 (see
                # camera_info: camera_intrinsics vs. camera_intrinsics_half), so
                # a pixel from front_depth/ unprojected with the halved
                # intrinsics lands on the same 3D point as this cloud.
                if not args.skip_point_clouds and FRONT_INTRINSICS is not None:
                    pc = generate_point_cloud_from_depth(
                        color_bgr, depth_m,
                        intrinsics=FRONT_INTRINSICS,
                        extrinsic_matrix=FRONT_EXTRINSICS,
                        workspace_bounds=WORKSPACE,
                        include_color=True,
                        num_points=args.num_points,
                    )
                    np.save(
                        os.path.join(front_pc_dir, f"frame_{output_idx:04d}.npy"), pc
                    )

                # Downscale RGB (area-averaging is correct for colour) & save.
                color_resized = cv2.resize(
                    color_bgr, (FRONT_SAVE_W, FRONT_SAVE_H),
                    interpolation=cv2.INTER_AREA)
                cv2.imwrite(
                    os.path.join(front_rgb_dir, f"frame_{output_idx:04d}.png"),
                    color_resized
                )

                # Downscale depth with NEAREST ONLY. Any averaging kernel
                # invents phantom distances across depth edges (blending
                # foreground into background) and averages a 0 sentinel into a
                # plausible-looking real depth, which destroys the validity
                # mask that processing/depth.py relies on.
                depth_resized = cv2.resize(
                    depth_m, (FRONT_SAVE_W, FRONT_SAVE_H),
                    interpolation=cv2.INTER_NEAREST)
                save_depth_png_mm(
                    os.path.join(front_depth_dir, f"frame_{output_idx:04d}.png"),
                    depth_resized
                )

                # State + Force
                episode_states.append(last_known_positions.copy())
                episode_forces.append(last_known_forces.copy())
                episode_hand_actions.append(last_known_hand_action.copy())
                episode_arm_actions.append(last_known_arm_action.copy())
                output_timestamps_ns.append(cam_ts_ns)

                if not state_updated:
                    dropped_state_count += 1
                if not seen_any_state:
                    # Still inside the startup ramp: no state has EVER
                    # arrived, so this frame holds zeros rather than
                    # carried-forward measurements.
                    if state_updated:
                        seen_any_state = True
                    else:
                        n_stale_state_start += 1

                output_idx += 1

            pbar.update(1)

    cam.close()

    # Save state
    states_array = np.stack(episode_states, axis=0)
    if (len(states_array) > 1
            and np.all(states_array[0] == 0)
            and not np.all(states_array[1] == 0)):
        states_array[0] = states_array[1]
        cprint("  [QA] Back-filled frame 0.", "cyan")
    np.save(os.path.join(episode_dir, 'state.npy'), states_array)

    # Save force
    forces_array = np.stack(episode_forces, axis=0)
    if (len(forces_array) > 1
            and np.all(forces_array[0] == 0)
            and not np.all(forces_array[1] == 0)):
        forces_array[0] = forces_array[1]
    np.save(os.path.join(episode_dir, 'force.npy'), forces_array)

    # Save commanded hand pose
    hand_actions_array = np.stack(episode_hand_actions, axis=0)
    if (len(hand_actions_array) > 1
            and np.all(hand_actions_array[0] == 0)
            and not np.all(hand_actions_array[1] == 0)):
        hand_actions_array[0] = hand_actions_array[1]
    np.save(os.path.join(episode_dir, 'hand_action.npy'), hand_actions_array)

    # Save commanded arm pose
    arm_actions_array = np.stack(episode_arm_actions, axis=0)
    if (len(arm_actions_array) > 1
            and np.all(arm_actions_array[0] == 0)
            and not np.all(arm_actions_array[1] == 0)):
        arm_actions_array[0] = arm_actions_array[1]
    np.save(os.path.join(episode_dir, 'arm_action.npy'), arm_actions_array)

    # Save timestamps
    np.save(os.path.join(episode_dir, 'frame_timestamps_ns.npy'),
            np.array(output_timestamps_ns, dtype=np.int64))
    np.save(os.path.join(episode_dir, 'sync_gaps.npy'),
            np.array(sync_gaps_ms, dtype=np.float32))

    # Sync report
    qa_report.sync_report(episode_dir, sync_gaps_ms, label="ZED-to-ROS")

    if dropped_state_count > 0:
        cprint(f"  [QA] Missed states: {dropped_state_count}/{output_idx} frames.", "yellow")

    # Force QA. An all-zero force.npy loads and trains without error but
    # carries no tactile signal, so surface it loudly here rather than letting
    # it reach the dataset silently.
    nonzero_frames = int(np.count_nonzero(np.any(forces_array != 0, axis=1)))
    if nonzero_frames == 0:
        cprint("  [QA] force.npy is ALL ZEROS - no fingertip F/T was recorded "
               "for this episode (check the hand driver's fingertip "
               "broadcasters).", "red")
    else:
        pct = 100.0 * nonzero_frames / len(forces_array)
        colour = "green" if pct > 95 else "yellow"
        cprint(f"  [QA] Force coverage: {nonzero_frames}/{len(forces_array)} "
               f"frames nonzero ({pct:.1f}%), "
               f"|F/T| max {np.abs(forces_array).max():.2f}", colour)
        if force_seen_count == 0:
            cprint("  [QA] ...but no fresh hand_ft arrived; values are "
                   "carried-forward only.", "yellow")

    # Hand action QA. Mirrors the force check: all-zero means the episode was
    # recorded without the robot-hand mapper, so the commanded hand pose is
    # absent even though the file loads fine.
    ha_nonzero = int(np.count_nonzero(np.any(hand_actions_array != 0, axis=1)))
    if ha_nonzero == 0:
        cprint("  [QA] hand_action.npy is ALL ZEROS - no hand command was "
               "recorded for this episode (check that the robot-hand mapper "
               "was running).", "red")
    else:
        pct = 100.0 * ha_nonzero / len(hand_actions_array)
        colour = "green" if pct > 95 else "yellow"
        cprint(f"  [QA] Hand action coverage: {ha_nonzero}/"
               f"{len(hand_actions_array)} frames nonzero ({pct:.1f}%)", colour)
        if hand_action_seen_count == 0:
            cprint("  [QA] ...but no fresh hand_action arrived; values are "
                   "carried-forward only.", "yellow")

    # Arm action QA. Mirrors the hand_action check. The quaternion block is
    # checked separately from the position: a dropped teleop topic leaves an
    # all-zero quaternion, which is not a rotation at all and would otherwise
    # reach the policy as a valid-looking target.
    aa_nonzero = int(np.count_nonzero(np.any(arm_actions_array != 0, axis=1)))
    if aa_nonzero == 0:
        cprint("  [QA] arm_action.npy is ALL ZEROS - no commanded arm pose was "
               "recorded for this episode (check that teleop_ur.py was "
               "publishing /teleop/target_pose).", "red")
    else:
        pct = 100.0 * aa_nonzero / len(arm_actions_array)
        quat_norms = np.linalg.norm(arm_actions_array[:, 3:7], axis=1)
        bad_quat = int(np.count_nonzero((quat_norms < 0.9) | (quat_norms > 1.1)))
        colour = "green" if pct > 95 and bad_quat == 0 else "yellow"
        cprint(f"  [QA] Arm action coverage: {aa_nonzero}/"
               f"{len(arm_actions_array)} frames nonzero ({pct:.1f}%)", colour)
        if bad_quat:
            cprint(f"  [QA] {bad_quat} arm_action frame(s) have a non-unit "
                   f"quaternion.", "red")
        if arm_action_seen_count == 0:
            cprint("  [QA] ...but no fresh arm_action arrived; values are "
                   "carried-forward only.", "yellow")

    # Wrist camera
    wrist = extract_wrist_camera(episode_dir, output_timestamps_ns, args)

    if n_stale_state_start > 0:
        cprint(f"  [QA] First {n_stale_state_start} frame(s) precede the first "
               f"robot_state (zero-filled arm/hand/force).", "yellow")

    # Trim whichever startup ramp is longer. Both have the same cause - the ZED
    # starts before the other sources - and both are a contiguous run from
    # frame 0, so a single trim clears both: cutting only the wrist ramp would
    # leave zero-filled arm/force frames behind, and cutting them one after the
    # other would renumber the frame files twice.
    n_stale = max(wrist["n_stale_start"], n_stale_state_start)
    if wrist["ok"] and args.trim_stale_start and n_stale > 0:
        if trim_episode_start(episode_dir, n_stale):
            output_idx -= n_stale
            # Keep the QA view consistent with what is now on disk.
            states_array = states_array[n_stale:]
            forces_array = forces_array[n_stale:]
            hand_actions_array = hand_actions_array[n_stale:]
            arm_actions_array = arm_actions_array[n_stale:]
            output_timestamps_ns = output_timestamps_ns[n_stale:]
            wrist["gaps_ms"] = wrist["gaps_ms"][n_stale:]
            wrist["trimmed_frames"] = n_stale
            wrist["n_stale_start"] = 0
            # The ramp is off the disk, so the counters must not keep
            # reporting it - otherwise QA flags frames that no longer exist.
            dropped_state_count = max(0, dropped_state_count
                                      - n_stale_state_start)
            # The first state after the ramp carries the whole startup offset
            # as its gap (the ZED frame it matched is older than the state by
            # the length of the ramp). That one entry is what produced the
            # "max gap 134ms" warning while every surviving frame was within
            # 2ms. sync_gaps.npy is NOT per-frame, so it cannot be sliced by
            # frame index like the arrays above - but the leading entries do
            # belong to the dropped frames, so dropping exactly as many leaves
            # the distribution describing what is actually on disk.
            n_gap_drop = min(n_stale, len(sync_gaps_ms))
            sync_gaps_ms = sync_gaps_ms[n_gap_drop:]
            np.save(os.path.join(episode_dir, 'sync_gaps.npy'),
                    np.array(sync_gaps_ms, dtype=np.float32))
            qa_report.sync_report(episode_dir, sync_gaps_ms,
                                  label="ZED-to-ROS (nach Trim)")
            n_stale_state_start = 0

    cprint(f"  [Done] {output_idx} frames.", "green")

    if not args.skip_qa:
        qa_report.assess_episode(
            episode_dir,
            state=states_array,
            force=forces_array,
            timestamps=np.array(output_timestamps_ns, dtype=np.int64),
            state_gaps_ms=np.array(sync_gaps_ms, dtype=np.float32),
            wrist=wrist,
            make_plots=not args.no_qa_plots,
        )
    return True


# ═══════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════

def verify_episode(episode_dir, check_wrist=True, check_pc=True):
    """Verify all expected files exist with consistent frame counts."""
    state_path = os.path.join(episode_dir, 'state.npy')
    if not os.path.exists(state_path):
        return False

    states = np.load(state_path)
    n = len(states)
    if n == 0 or np.isnan(states).any() or np.isinf(states).any():
        return False

    # Force. Required, and validated on width/finiteness too - a truncated or
    # NaN-filled force array would otherwise fail much later, during training.
    force_path = os.path.join(episode_dir, 'force.npy')
    if not os.path.exists(force_path):
        return False
    forces = np.load(force_path)
    if len(forces) != n:
        return False
    if forces.ndim != 2 or forces.shape[1] != HAND_FT_DIM:
        return False
    if np.isnan(forces).any() or np.isinf(forces).any():
        return False

    # Commanded hand pose. Tolerated when absent so episodes recorded before
    # hand_action existed still verify, but validated strictly once present:
    # a wrong width or a length mismatch would misalign action and state.
    hand_action_path = os.path.join(episode_dir, 'hand_action.npy')
    if os.path.exists(hand_action_path):
        hand_actions = np.load(hand_action_path)
        if len(hand_actions) != n:
            return False
        if hand_actions.ndim != 2 or hand_actions.shape[1] != HAND_ACTION_DIM:
            return False
        if np.isnan(hand_actions).any() or np.isinf(hand_actions).any():
            return False

    # Commanded arm pose. Same contract as hand_action: optional for older
    # episodes, strictly validated once present.
    arm_action_path = os.path.join(episode_dir, 'arm_action.npy')
    if os.path.exists(arm_action_path):
        arm_actions = np.load(arm_action_path)
        if len(arm_actions) != n:
            return False
        if arm_actions.ndim != 2 or arm_actions.shape[1] != ARM_ACTION_DIM:
            return False
        if np.isnan(arm_actions).any() or np.isinf(arm_actions).any():
            return False

    # Front RGB
    front_rgb_dir = os.path.join(episode_dir, 'front_rgb')
    if not os.path.exists(front_rgb_dir):
        return False
    if len(glob.glob(os.path.join(front_rgb_dir, '*.png'))) != n:
        return False

    # Front depth must be frame-for-frame with front_rgb, otherwise the
    # depth/RGB pairing is off by an unknown offset that nothing downstream
    # can detect.
    front_depth_dir = os.path.join(episode_dir, 'front_depth')
    if os.path.exists(front_depth_dir):
        if len(glob.glob(os.path.join(front_depth_dir, '*.png'))) != n:
            return False

    # Front PC
    if check_pc:
        front_pc_dir = os.path.join(episode_dir, 'front_pc')
        if os.path.exists(front_pc_dir):
            if len(glob.glob(os.path.join(front_pc_dir, '*.npy'))) != n:
                return False

    # Wrist
    if check_wrist:
        wrist_rgb_dir = os.path.join(episode_dir, 'wrist_rgb')
        if os.path.exists(wrist_rgb_dir):
            if len(glob.glob(os.path.join(wrist_rgb_dir, '*.png'))) != n:
                return False

    return True


def is_already_extracted(episode_dir):
    """Check if the FRONT/state part of an episode was already extracted."""
    return verify_episode(episode_dir, check_wrist=False, check_pc=False)


def needs_wrist_extraction(episode_dir):
    """
    True if this episode still needs wrist_rgb/ extracted.

    Requires a realsense_bag (otherwise there is nothing to extract) and
    frame_timestamps_ns.npy (needed to sync without reopening the SVO).
    """
    if not os.path.exists(os.path.join(episode_dir, 'realsense_bag')):
        return False
    if not os.path.exists(os.path.join(episode_dir, 'frame_timestamps_ns.npy')):
        return False

    state_path = os.path.join(episode_dir, 'state.npy')
    if not os.path.exists(state_path):
        return False
    n = len(np.load(state_path))
    if n == 0:
        return False

    wrist_rgb_dir = os.path.join(episode_dir, 'wrist_rgb')
    if not os.path.exists(wrist_rgb_dir):
        return True
    return len(glob.glob(os.path.join(wrist_rgb_dir, '*.png'))) != n


def extract_wrist_only(episode_dir, args):
    """
    Extract ONLY the wrist camera for an already-extracted episode.

    Reuses the stored frame_timestamps_ns.npy, so the SVO is never opened and
    the max_frames deletion path in extract_episode() is never reached.
    Nothing is ever deleted here.
    """
    ts_path = os.path.join(episode_dir, 'frame_timestamps_ns.npy')
    output_timestamps_ns = np.load(ts_path).tolist()

    cprint(f"\n--- {os.path.basename(episode_dir)} (wrist only, "
           f"{len(output_timestamps_ns)} frames) ---", "cyan")

    wrist = extract_wrist_camera(episode_dir, output_timestamps_ns, args)

    n_stale = wrist["n_stale_start"]
    if wrist["ok"] and args.trim_stale_start and n_stale > 0:
        if trim_episode_start(episode_dir, n_stale):
            wrist["gaps_ms"] = wrist["gaps_ms"][n_stale:]
            wrist["trimmed_frames"] = n_stale
            wrist["n_stale_start"] = 0

    if not args.skip_qa:
        # Arrays are reloaded from disk here: trimming just rewrote them.
        qa_report.assess_episode(episode_dir, wrist=wrist,
                                 make_plots=not args.no_qa_plots)
    return wrist["ok"]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if not ROSBAGS_AVAILABLE:
        cprint("[Warning] 'rosbags' not installed. Wrist extraction disabled.", "yellow")

    if FRONT_INTRINSICS is None and not args.skip_point_clouds:
        cprint("[Warning] Camera intrinsics not found. Point clouds disabled.", "yellow")
        args.skip_point_clouds = True

    if args.include_failed:
        episodes = sorted([
            d for d in os.listdir(args.data_dir)
            if d.startswith('episode_') 
            and os.path.isdir(os.path.join(args.data_dir, d))
        ])
    else:
        episodes = sorted([
            d for d in os.listdir(args.data_dir)
            if d.startswith('episode_') 
            and 'failed' not in d 
            and os.path.isdir(os.path.join(args.data_dir, d))
        ])

    if not episodes:
        cprint(f"No episodes found in {args.data_dir}", "red")
        return

    cprint(f"Found {len(episodes)} episodes in {args.data_dir}", "green", attrs=["bold"])

    success_count = 0
    deleted_count = 0
    wrist_backfilled = 0
    qa_results = []

    for ep in episodes:
        episode_dir = os.path.join(args.data_dir, ep)

        if is_already_extracted(episode_dir):
            # Front/state already done. Backfill wrist_rgb/ if it is missing,
            # without touching (or deleting) any existing data.
            if needs_wrist_extraction(episode_dir):
                cprint(f"  {ep}: front OK, wrist_rgb missing → extracting wrist.",
                       "cyan")
                if extract_wrist_only(episode_dir, args):
                    wrist_backfilled += 1
                else:
                    cprint(f"  [Warn] Wrist extraction failed for {ep}.", "yellow")
            else:
                cprint(f"  Skipping {ep}: already extracted.", "yellow")
                # Still fold it into the summary, re-using the cached report
                # when one exists so skipped episodes are not invisible.
                if not args.skip_qa:
                    cached = os.path.join(episode_dir, 'qa_report.json')
                    if os.path.exists(cached):
                        try:
                            with open(cached) as fh:
                                qa_results.append(json.load(fh))
                        except (json.JSONDecodeError, OSError):
                            pass
                    else:
                        qa_results.append(qa_report.assess_episode(
                            episode_dir, make_plots=not args.no_qa_plots,
                            verbose=False))
            success_count += 1
            continue

        if not os.path.exists(episode_dir):
            deleted_count += 1
            continue

        if extract_episode(episode_dir, args):
            success_count += 1
            if not args.skip_qa:
                cached = os.path.join(episode_dir, 'qa_report.json')
                if os.path.exists(cached):
                    try:
                        with open(cached) as fh:
                            qa_results.append(json.load(fh))
                    except (json.JSONDecodeError, OSError):
                        pass
        elif not os.path.exists(episode_dir):
            deleted_count += 1

    cprint(f"\n{'='*50}", "green")
    cprint(f"Done: {success_count}/{len(episodes)} extracted.", "green", attrs=['bold'])
    if wrist_backfilled > 0:
        cprint(f"Wrist backfilled: {wrist_backfilled} episodes.", "green")
    if deleted_count > 0:
        cprint(f"Deleted: {deleted_count} episodes (>{args.max_frames} frames).", "red")

    if qa_results:
        counts = qa_report.write_summary(
            args.data_dir, qa_results, make_plots=not args.no_qa_plots)
        cprint(f"QA: PASS {counts['PASS']}  WARN {counts['WARN']}  "
               f"FAIL {counts['FAIL']}  -> qa_summary.txt",
               "red" if counts["FAIL"] else
               ("yellow" if counts["WARN"] else "green"), attrs=['bold'])


if __name__ == '__main__':
    main()