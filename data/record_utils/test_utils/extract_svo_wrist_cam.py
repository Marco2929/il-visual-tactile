#!/usr/bin/env python3
"""
Post-hoc wrist camera extraction.
Adds wrist_img/ and wrist_depth/ to already-extracted episodes.
"""
import os
import cv2
import numpy as np
import pyzed.sl as sl
import argparse
import glob
from termcolor import cprint
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    raise ImportError("Install rosbags: pip install rosbags")

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")


def parse_args():
    parser = argparse.ArgumentParser(description="Add wrist camera to existing extracted episodes")
    parser.add_argument('--data_dir', type=str,
                        default=f'/root/ros2_ws/data/episodes/{FOLDER_NAME}')
    parser.add_argument('--target_fps', type=int, default=30)
    parser.add_argument('--wrist_color_topic', type=str,
                        default='/cam_right/cam_right/color/image_raw')
    parser.add_argument('--wrist_depth_topic', type=str,
                        default='/cam_right/cam_right/aligned_depth_to_color/image_raw')
    parser.add_argument('--max_acceptable_gap_ms', type=float, default=50.0,
                        help='Flag frames with sync gap above this threshold')
    return parser.parse_args()


def get_output_timestamps(episode_dir, target_fps=30):
    """Get ZED output timestamps - from saved file or re-extract from SVO."""
    ts_path = os.path.join(episode_dir, 'frame_timestamps_ns.npy')
    if os.path.exists(ts_path):
        return np.load(ts_path)

    cprint("  No frame_timestamps_ns.npy found, re-reading SVO timestamps...", "yellow")
    svo_path = os.path.join(episode_dir, 'recording.svo2')
    if not os.path.exists(svo_path):
        return None

    source_fps = 60
    skip_factor = source_fps // target_fps

    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NONE

    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        return None

    num_frames = cam.get_svo_number_of_frames()
    runtime = sl.RuntimeParameters()
    timestamps = []

    for frame_idx in range(num_frames):
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            if frame_idx % skip_factor == 0:
                ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                timestamps.append(ts)
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break

    cam.close()

    timestamps = np.array(timestamps, dtype=np.int64)
    np.save(ts_path, timestamps)
    cprint(f"  Saved {len(timestamps)} timestamps to frame_timestamps_ns.npy", "cyan")
    return timestamps


def decode_image_msg(msg, msgtype):
    """Decode sensor_msgs/Image or CompressedImage to numpy array."""
    if 'CompressedImage' in msgtype:
        img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    encoding = msg.encoding
    h, w = msg.height, msg.width

    if encoding in ('16UC1', 'mono16'):
        return np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w)
    elif encoding == '32FC1':
        return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(h, w)
    elif encoding == 'mono8':
        return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w)
    elif encoding == 'rgb8':
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == 'bgr8':
        return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
    elif encoding == 'bgra8':
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif encoding == 'rgba8':
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, -1)


def get_header_stamp_ns(msg, fallback_ts):
    try:
        stamp_ns = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        return stamp_ns if stamp_ns > 0 else fallback_ts
    except Exception:
        return fallback_ts


def extract_wrist_for_episode(episode_dir, target_fps, wrist_color_topic,
                               wrist_depth_topic, max_gap_ms):
    """Add wrist camera data to an already-extracted episode."""
    bag_path = os.path.join(episode_dir, 'realsense_bag')
    if not os.path.exists(bag_path):
        cprint(f"  [Skip] No realsense_bag", "yellow")
        return False

    wrist_img_dir = os.path.join(episode_dir, 'wrist_img')
    wrist_depth_dir = os.path.join(episode_dir, 'wrist_depth')
    state_path = os.path.join(episode_dir, 'state.npy')

    if not os.path.exists(state_path):
        cprint(f"  [Skip] No state.npy - episode not extracted yet", "red")
        return False

    expected_frames = len(np.load(state_path))

    # Check if wrist already extracted
    if os.path.exists(wrist_img_dir):
        existing = glob.glob(os.path.join(wrist_img_dir, '*.png'))
        if len(existing) == expected_frames:
            cprint(f"  [Skip] Wrist already extracted ({expected_frames} frames)", "yellow")
            return True

    # Get reference timestamps
    output_timestamps = get_output_timestamps(episode_dir, target_fps)
    if output_timestamps is None or len(output_timestamps) != expected_frames:
        cprint(f"  [Error] Timestamp count mismatch", "red")
        return False

    # Read bag
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    color_entries = []  # (timestamp_ns, rawdata, msgtype_string)
    depth_entries = []

    with Reader(bag_path) as reader:
        available_topics = sorted(set(c.topic for c in reader.connections))
        color_conns = [c for c in reader.connections if c.topic == wrist_color_topic]
        depth_conns = [c for c in reader.connections if c.topic == wrist_depth_topic]

        if not color_conns:
            cprint(f"  [Error] Topic '{wrist_color_topic}' not found.", "red")
            cprint(f"  Available: {available_topics}", "yellow")
            return False

        if not depth_conns:
            cprint(f"  [Warning] Depth topic '{wrist_depth_topic}' not found.", "yellow")

        all_conns = color_conns + depth_conns
        for conn, timestamp, rawdata in reader.messages(connections=all_conns):
            if conn.topic == wrist_color_topic:
                color_entries.append((timestamp, rawdata, conn.msgtype))
            elif conn.topic == wrist_depth_topic:
                depth_entries.append((timestamp, rawdata, conn.msgtype))

    color_entries.sort(key=lambda x: x[0])
    depth_entries.sort(key=lambda x: x[0])

    if not color_entries:
        cprint("  [Error] No color messages in bag.", "red")
        return False

    cprint(f"  Bag: {len(color_entries)} color, {len(depth_entries)} depth frames", "white")

    # Get header timestamps
    color_ts = np.array([
        get_header_stamp_ns(typestore.deserialize_cdr(raw, msgtype), ts)
        for ts, raw, msgtype in color_entries
    ], dtype=np.int64)

    depth_ts = np.array([], dtype=np.int64)
    if depth_entries:
        depth_ts = np.array([
            get_header_stamp_ns(typestore.deserialize_cdr(raw, msgtype), ts)
            for ts, raw, msgtype in depth_entries
        ], dtype=np.int64)

    # Create output dirs
    os.makedirs(wrist_img_dir, exist_ok=True)
    if depth_entries:
        os.makedirs(wrist_depth_dir, exist_ok=True)

    # Sync and extract
    output_ts = np.array(output_timestamps, dtype=np.int64)
    color_gaps = []
    depth_gaps = []
    bad_frame_count = 0

    for idx, target_ts in enumerate(tqdm(output_ts, desc="  Extracting wrist", unit="frame")):
        # --- Color ---
        ci = np.argmin(np.abs(color_ts - target_ts))
        gap = abs(int(color_ts[ci]) - int(target_ts)) / 1e6
        color_gaps.append(gap)
        if gap > max_gap_ms:
            bad_frame_count += 1

        _, rawdata, msgtype = color_entries[ci]
        msg = typestore.deserialize_cdr(rawdata, msgtype)
        img = decode_image_msg(msg, msgtype)
        cv2.imwrite(os.path.join(wrist_img_dir, f"frame_{idx:04d}.png"), img)

        # --- Depth ---
        if len(depth_entries) > 0:
            di = np.argmin(np.abs(depth_ts - target_ts))
            gap_d = abs(int(depth_ts[di]) - int(target_ts)) / 1e6
            depth_gaps.append(gap_d)

            _, rawdata_d, msgtype_d = depth_entries[di]
            msg_d = typestore.deserialize_cdr(rawdata_d, msgtype_d)
            depth_img = decode_image_msg(msg_d, msgtype_d)
            np.save(os.path.join(wrist_depth_dir, f"frame_{idx:04d}.npy"), depth_img)

    # Report
    color_gaps = np.array(color_gaps)
    avg_gap = np.mean(color_gaps)
    max_gap_val = np.max(color_gaps)
    p95_gap = np.percentile(color_gaps, 95)

    cprint(f"\n  --- Wrist Sync Summary ---", "cyan")
    cprint(f"  Avg: {avg_gap:.2f}ms | P95: {p95_gap:.2f}ms | Max: {max_gap_val:.2f}ms",
           "green" if max_gap_val < max_gap_ms else "yellow")
    if bad_frame_count > 0:
        cprint(f"  [!] {bad_frame_count}/{len(output_ts)} frames exceed {max_gap_ms}ms threshold", "red")

    np.save(os.path.join(episode_dir, 'wrist_color_sync_gaps.npy'), color_gaps)
    if depth_gaps:
        np.save(os.path.join(episode_dir, 'wrist_depth_sync_gaps.npy'), np.array(depth_gaps))

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(color_gaps, bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(avg_gap, color='red', linestyle='dashed', label=f'Mean: {avg_gap:.2f}ms')
    plt.axvline(max_gap_ms, color='black', linestyle='dotted', label=f'Threshold: {max_gap_ms}ms')
    plt.title("ZED-to-Wrist Synchronization")
    plt.xlabel("Gap (ms)")
    plt.ylabel("Frames")
    plt.legend()
    plt.savefig(os.path.join(episode_dir, 'sync_histogram_wrist.png'))
    plt.close()

    return True


def main():
    args = parse_args()
    episodes = sorted([d for d in os.listdir(args.data_dir) if d.startswith('episode_')])

    cprint(f"\n{'='*60}", "cyan")
    cprint(f"  Post-hoc Wrist Camera Extraction", "cyan", attrs=['bold'])
    cprint(f"  Episodes: {len(episodes)} | Target FPS: {args.target_fps}", "cyan")
    cprint(f"  Color: {args.wrist_color_topic}", "cyan")
    cprint(f"  Depth: {args.wrist_depth_topic}", "cyan")
    cprint(f"  Max acceptable gap: {args.max_acceptable_gap_ms}ms", "cyan")
    cprint(f"{'='*60}\n", "cyan")

    success = 0
    for ep in episodes:
        episode_dir = os.path.join(args.data_dir, ep)
        if not os.path.isdir(episode_dir):
            continue

        cprint(f"\n[{ep}]", "white", attrs=['bold'])
        if extract_wrist_for_episode(episode_dir, args.target_fps,
                                      args.wrist_color_topic, args.wrist_depth_topic,
                                      args.max_acceptable_gap_ms):
            success += 1

    cprint(f"\n{'='*60}", "green")
    cprint(f"  Done: {success}/{len(episodes)} episodes with wrist data.", "green", attrs=['bold'])
    cprint(f"{'='*60}", "green")


if __name__ == '__main__':
    main()