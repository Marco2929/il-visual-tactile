#!/usr/bin/env python3
# scripts/extract_episodes.py
"""
Extract episodes from SVO + ROS bags.
Generates: front_rgb/, front_pc/, wrist_rgb/, wrist_pc/, state.npy, force.npy.
Depth is NOT saved — point clouds are generated inline.
"""

import os
import cv2
import json
import glob
import shutil
import argparse
import numpy as np
import pyzed.sl as sl
import matplotlib.pyplot as plt
from pathlib import Path
from termcolor import cprint
from tqdm import tqdm
import sys
sys.path.insert(0, "/root/ros2_ws/data/record_utils")

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
)

from camera_info import (
    wrist_camera_intrinsics as WRIST_INTRINSICS,
    wrist_camera_extrinsics as WRIST_EXTRINSICS,
)

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract episodes from SVO + ROS bags")
    parser.add_argument('--data_dir', type=str,
                        default=f'/root/ros2_ws/data/episodes/external/{FOLDER_NAME}')
    parser.add_argument('--target_fps', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=1500,
                        help='Max SVO frames (60fps). Episodes above are deleted.')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Points per point cloud')
    parser.add_argument('--skip_point_clouds', action='store_true',
                        help='Skip ALL point cloud generation (only RGB + state)')
    parser.add_argument('--skip_wrist_pc', action='store_true', default=True,
                        help='Skip only the WRIST point clouds (front_pc is kept). '
                             'On by default; use --keep_wrist_pc to generate them.')
    parser.add_argument('--keep_wrist_pc', dest='skip_wrist_pc',
                        action='store_false',
                        help='Generate wrist point clouds into wrist_pc/')
    parser.add_argument('--wrist_color_topic', type=str,
                        default='/cam_right/cam_right/color/image_raw')
    parser.add_argument('--wrist_depth_topic', type=str,
                        default='/cam_right/cam_right/aligned_depth_to_color/image_raw')
    parser.add_argument('--include_failed', type=bool,
                        default=False)
    # --- Wrist depth range cap (camera frame, meters) ---
    parser.add_argument('--wrist_max_depth', type=float, default=0.5,
                        help='Cap wrist depth (m) before PC gen. '
                             'Points farther than this are dropped. '
                             '<=0 disables the cap.')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Sync Report
# ═══════════════════════════════════════════════════════════════

def generate_sync_report(episode_dir, sync_gaps_ms, label="Camera-to-ROS"):
    """Terminal stats + histogram."""
    if not sync_gaps_ms:
        return

    gaps = np.array(sync_gaps_ms)
    avg_gap = np.mean(gaps)
    max_gap = np.max(gaps)
    std_gap = np.std(gaps)

    cprint(f"\n--- {label} Sync Profile ---", "cyan")
    cprint(f"  Avg: {avg_gap:.2f}ms | Max: {max_gap:.2f}ms | Jitter: {std_gap:.2f}ms",
           "green" if max_gap < 10 else "yellow")

    plt.figure(figsize=(8, 4))
    plt.hist(gaps, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_gap, color='red', linestyle='dashed', label=f'Mean: {avg_gap:.2f}ms')
    plt.title(f"{label} Sync Delay")
    plt.xlabel("Gap (ms)")
    plt.ylabel("Frames")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    safe_label = label.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(os.path.join(episode_dir, f'sync_histogram_{safe_label}.png'))
    plt.close()


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


def extract_wrist_camera(episode_dir, output_timestamps_ns, args):
    """Extract wrist RGB + point cloud from rosbag, synced to ZED timestamps."""
    if not ROSBAGS_AVAILABLE:
        cprint("  [Error] 'rosbags' not installed.", "red")
        return False

    bag_path = os.path.join(episode_dir, 'realsense_bag')
    if not os.path.exists(bag_path):
        cprint("  [Info] No realsense_bag found, skipping wrist.", "white")
        return False

    # Wrist point clouds are skipped if either the global or the wrist-only
    # flag is set.
    build_wrist_pc = not args.skip_point_clouds and not args.skip_wrist_pc

    wrist_rgb_dir = os.path.join(episode_dir, 'wrist_rgb')
    wrist_pc_dir = os.path.join(episode_dir, 'wrist_pc')
    os.makedirs(wrist_rgb_dir, exist_ok=True)
    if build_wrist_pc:
        os.makedirs(wrist_pc_dir, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_JAZZY)

    # Read all messages
    color_entries = []
    depth_entries = []

    with Reader(bag_path) as reader:
        color_connections = [c for c in reader.connections if c.topic == args.wrist_color_topic]
        depth_connections = [c for c in reader.connections if c.topic == args.wrist_depth_topic]

        if not color_connections:
            available = sorted(set(c.topic for c in reader.connections))
            cprint(f"  [Error] Topic '{args.wrist_color_topic}' not found.", "red")
            cprint(f"  Available: {available}", "yellow")
            return False

        all_connections = color_connections + depth_connections
        for connection, timestamp, rawdata in reader.messages(connections=all_connections):
            if connection.topic == args.wrist_color_topic:
                color_entries.append((timestamp, rawdata, connection))
            elif connection.topic == args.wrist_depth_topic:
                depth_entries.append((timestamp, rawdata, connection))

    color_entries.sort(key=lambda x: x[0])
    depth_entries.sort(key=lambda x: x[0])

    if not color_entries:
        cprint("  [Error] No wrist color messages found.", "red")
        return False

    # Header timestamps
    color_timestamps_ns = []
    for ts, rawdata, conn in color_entries:
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
        color_timestamps_ns.append(get_header_stamp_ns(msg, ts))
    color_timestamps_ns = np.array(color_timestamps_ns, dtype=np.int64)

    depth_timestamps_ns = np.array([], dtype=np.int64)
    if depth_entries and build_wrist_pc:
        d_ts = []
        for ts, rawdata, conn in depth_entries:
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            d_ts.append(get_header_stamp_ns(msg, ts))
        depth_timestamps_ns = np.array(d_ts, dtype=np.int64)

    # Sync to ZED timestamps
    output_ts = np.array(output_timestamps_ns, dtype=np.int64)
    sync_gaps_ms = []

    for idx, target_ts in enumerate(tqdm(output_ts, desc="  Syncing wrist", unit="frame")):
        # Color
        closest_idx = np.argmin(np.abs(color_timestamps_ns - target_ts))
        gap_ms = abs(int(color_timestamps_ns[closest_idx]) - int(target_ts)) / 1e6
        sync_gaps_ms.append(gap_ms)

        _, rawdata, conn = color_entries[closest_idx]
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
        color_img = decode_image_msg(msg, conn)
        cv2.imwrite(os.path.join(wrist_rgb_dir, f"frame_{idx:04d}.png"), color_img)

        # Depth → Point Cloud (no depth saved)
        if (build_wrist_pc and len(depth_entries) > 0
                and WRIST_INTRINSICS is not None):
            closest_d_idx = np.argmin(np.abs(depth_timestamps_ns - target_ts))
            _, rawdata_d, conn_d = depth_entries[closest_d_idx]
            msg_d = typestore.deserialize_cdr(rawdata_d, conn_d.msgtype)
            depth_img = decode_image_msg(msg_d, conn_d).astype(np.float32)

            # Detect units once (RealSense is typically mm).
            _valid = np.isfinite(depth_img) & (depth_img > 0)
            _is_mm = np.any(_valid) and np.median(depth_img[_valid]) > 20.0

            # Cap depth range (camera frame) before PC gen so the point cloud
            # focuses on the near grasp region.
            if args.wrist_max_depth > 0:
                cap = args.wrist_max_depth * 1000.0 if _is_mm else args.wrist_max_depth
                depth_img[depth_img > cap] = 0.0

            pc = generate_point_cloud_from_depth(
                color_img, depth_img.astype(np.float32),
                WRIST_INTRINSICS, WRIST_EXTRINSICS,
                num_points=args.num_points,
                workspace_bounds=None,
            )
            np.save(os.path.join(wrist_pc_dir, f"frame_{idx:04d}.npy"), pc)

    generate_sync_report(episode_dir, sync_gaps_ms, label="ZED-to-Wrist")
    return True


# ═══════════════════════════════════════════════════════════════
# ZED SVO Extraction (Front Camera)
# ═══════════════════════════════════════════════════════════════

def extract_episode(episode_dir, args, source_fps = 60):
    """
    Extract single episode from SVO.
    Saves: front_rgb/, front_pc/, state.npy, force.npy.
    Does NOT save depth images.
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
    front_pc_dir = os.path.join(episode_dir, 'front_pc')
    os.makedirs(front_rgb_dir, exist_ok=True)
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
    sync_gaps_ms = []
    output_timestamps_ns = []
    last_known_positions = np.zeros(12, dtype=np.float32)
    last_known_forces = np.zeros(6, dtype=np.float32)
    dropped_state_count = 0
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
            res = cam.retrieve_svo_data("joint_states", custom_data, last_ts, current_ts)

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
                        last_known_positions = np.array(
                            chosen['positions'], dtype=np.float32)
                        if 'force_values' in chosen:
                            last_known_forces = np.array(
                                chosen['force_values'], dtype=np.float32)
                        state_updated = True
                        gap_ms = abs(cam_ts_ns - chosen['timestamp_ns']) / 1e6
                        sync_gaps_ms.append(gap_ms)
                except json.JSONDecodeError:
                    pass

            last_ts = current_ts

            # Output frame (at target FPS)
            if frame_idx % skip_factor == 0:
                # RGB (no depth saved!)
                cam.retrieve_image(image_mat, sl.VIEW.LEFT)
                color_bgr = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)

                # Depth → Point Cloud inline
                if not args.skip_point_clouds and FRONT_INTRINSICS is not None:
                    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    depth_data = depth_mat.get_data().copy()

                    pc = generate_point_cloud_from_depth(
                        color_bgr, depth_data,
                        intrinsics=FRONT_INTRINSICS, 
                        extrinsic_matrix=FRONT_EXTRINSICS, 
                        workspace_bounds=WORKSPACE,
                        include_color=True,
                        num_points=args.num_points,
                    )
                    np.save(
                        os.path.join(front_pc_dir, f"frame_{output_idx:04d}.npy"), pc
                    )
                
                # 3. Downscale RGB and save
                color_resized = cv2.resize(color_bgr, (640, 480), interpolation=cv2.INTER_AREA)
                cv2.imwrite(
                    os.path.join(front_rgb_dir, f"frame_{output_idx:04d}.png"),
                    color_resized
                )

                # State + Force
                episode_states.append(last_known_positions.copy())
                episode_forces.append(last_known_forces.copy())
                output_timestamps_ns.append(cam_ts_ns)

                if not state_updated:
                    dropped_state_count += 1

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

    # Save timestamps
    np.save(os.path.join(episode_dir, 'frame_timestamps_ns.npy'),
            np.array(output_timestamps_ns, dtype=np.int64))

    # Sync report
    generate_sync_report(episode_dir, sync_gaps_ms, label="ZED-to-ROS")

    if dropped_state_count > 0:
        cprint(f"  [QA] Missed states: {dropped_state_count}/{output_idx} frames.", "yellow")

    # Wrist camera
    extract_wrist_camera(episode_dir, output_timestamps_ns, args)

    cprint(f"  [Done] {output_idx} frames.", "green")
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

    # Force
    force_path = os.path.join(episode_dir, 'force.npy')
    if os.path.exists(force_path):
        if len(np.load(force_path)) != n:
            return False

    # Front RGB
    front_rgb_dir = os.path.join(episode_dir, 'front_rgb')
    if not os.path.exists(front_rgb_dir):
        return False
    if len(glob.glob(os.path.join(front_rgb_dir, '*.png'))) != n:
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
        if check_pc:
            wrist_pc_dir = os.path.join(episode_dir, 'wrist_pc')
            if os.path.exists(wrist_pc_dir):
                if len(glob.glob(os.path.join(wrist_pc_dir, '*.npy'))) != n:
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

    return extract_wrist_camera(episode_dir, output_timestamps_ns, args)


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
            success_count += 1
            continue

        if not os.path.exists(episode_dir):
            deleted_count += 1
            continue

        if extract_episode(episode_dir, args):
            success_count += 1
        elif not os.path.exists(episode_dir):
            deleted_count += 1

    cprint(f"\n{'='*50}", "green")
    cprint(f"Done: {success_count}/{len(episodes)} extracted.", "green", attrs=['bold'])
    if wrist_backfilled > 0:
        cprint(f"Wrist backfilled: {wrist_backfilled} episodes.", "green")
    if deleted_count > 0:
        cprint(f"Deleted: {deleted_count} episodes (>{args.max_frames} frames).", "red")


if __name__ == '__main__':
    main()