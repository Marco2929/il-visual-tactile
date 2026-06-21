#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pyzed.sl as sl
import json
import glob
import shutil
from termcolor import cprint
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    ROSBAGS_AVAILABLE = True
except ImportError:
    ROSBAGS_AVAILABLE = False

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=f'/root/ros2_ws/data/episodes/{FOLDER_NAME}',
                        help='Raw data directory containing episode folders')
    parser.add_argument('--target_fps', type=int, default=30,
                        help='Target FPS for extraction (source is 60)')
    parser.add_argument('--max_frames', type=int, default=1500,
                        help='Max SVO frames (60fps). Episodes above this are deleted.')
    parser.add_argument('--wrist_color_topic', type=str,
                        default='/cam_right/cam_right/color/image_raw',
                        help='ROS2 topic for wrist RealSense color image')
    parser.add_argument('--wrist_depth_topic', type=str,
                        default='/cam_right/cam_right/aligned_depth_to_color/image_raw',
                        help='ROS2 topic for wrist RealSense depth image')
    args = parser.parse_args()
    return args


def generate_sync_report(episode_dir, sync_gaps_ms, label="Camera-to-ROS"):
    """Generates statistical terminal output and saves a histogram of the sync gaps."""
    if not sync_gaps_ms:
        cprint(f"[QA Warning] No sync data to generate report for {label}.", "yellow")
        return

    gaps = np.array(sync_gaps_ms)
    avg_gap = np.mean(gaps)
    max_gap = np.max(gaps)
    min_gap = np.min(gaps)
    std_gap = np.std(gaps)

    cprint(f"\n--- {label} Synchronization Profile ---", "cyan")
    cprint(f"Average Gap: {avg_gap:.2f} ms", "green" if avg_gap < 5 else "yellow")
    cprint(f"Maximum Gap: {max_gap:.2f} ms", "green" if max_gap < 10 else "red")
    cprint(f"Minimum Gap: {min_gap:.2f} ms", "white")
    cprint(f"Jitter (StdDev): {std_gap:.2f} ms\n", "white")

    plt.figure(figsize=(8, 5))
    plt.hist(gaps, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_gap, color='red', linestyle='dashed', linewidth=1.5,
                label=f'Mean: {avg_gap:.2f}ms')
    plt.title(f"{label} Synchronization Delay")
    plt.xlabel("Time Gap (milliseconds)")
    plt.ylabel("Frequency (Frames)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    safe_label = label.lower().replace(" ", "_").replace("-", "_")
    plot_path = os.path.join(episode_dir, f'sync_histogram_{safe_label}.png')
    plt.savefig(plot_path)
    plt.close()
    cprint(f"Saved sync histogram to: {plot_path}", "cyan")


def decode_image_msg(msg, conn):
    """Decode a sensor_msgs/Image or CompressedImage message to a numpy array."""
    if 'CompressedImage' in conn.msgtype:
        img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    encoding = msg.encoding
    h, w = msg.height, msg.width

    if encoding in ('16UC1', 'mono16'):
        img = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w)
    elif encoding == '32FC1':
        img = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(h, w)
    elif encoding == 'mono8':
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w)
    elif encoding == 'rgb8':
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == 'bgr8':
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
    elif encoding == 'bgra8':
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif encoding == 'rgba8':
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, -1)

    return img


def get_header_stamp_ns(msg, fallback_ts):
    """Extract header timestamp in nanoseconds from a ROS message."""
    try:
        stamp_ns = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        if stamp_ns > 0:
            return stamp_ns
    except Exception:
        pass
    return fallback_ts


def extract_wrist_camera(episode_dir, output_timestamps_ns, wrist_color_topic, wrist_depth_topic):
    """Extract wrist camera color + depth from rosbag, synchronized to ZED timestamps."""
    if not ROSBAGS_AVAILABLE:
        cprint("[Error] 'rosbags' package not installed. Run: pip install rosbags", "red")
        return False

    bag_path = os.path.join(episode_dir, 'realsense_bag')
    if not os.path.exists(bag_path):
        cprint(f"[Warning] No realsense_bag found in {episode_dir}", "yellow")
        return False

    wrist_img_dir = os.path.join(episode_dir, 'wrist_img')
    wrist_depth_dir = os.path.join(episode_dir, 'wrist_depth')
    os.makedirs(wrist_img_dir, exist_ok=True)
    os.makedirs(wrist_depth_dir, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # --- Read all color and depth messages ---
    cprint("  Reading wrist camera bag...", "white")
    color_entries = []
    depth_entries = []

    with Reader(bag_path) as reader:
        available_topics = sorted(set(c.topic for c in reader.connections))

        color_connections = [c for c in reader.connections if c.topic == wrist_color_topic]
        depth_connections = [c for c in reader.connections if c.topic == wrist_depth_topic]

        if not color_connections:
            cprint(f"[Error] Color topic '{wrist_color_topic}' not found in bag.", "red")
            cprint(f"  Available topics: {available_topics}", "yellow")
            return False

        if not depth_connections:
            cprint(f"[Warning] Depth topic '{wrist_depth_topic}' not found in bag.", "yellow")
            cprint(f"  Available topics: {available_topics}", "yellow")
            cprint("  Will extract color only.", "yellow")

        all_connections = color_connections + depth_connections
        for connection, timestamp, rawdata in reader.messages(connections=all_connections):
            if connection.topic == wrist_color_topic:
                color_entries.append((timestamp, rawdata, connection))
            elif connection.topic == wrist_depth_topic:
                depth_entries.append((timestamp, rawdata, connection))

    color_entries.sort(key=lambda x: x[0])
    depth_entries.sort(key=lambda x: x[0])

    cprint(f"  Found {len(color_entries)} color frames, {len(depth_entries)} depth frames in bag.", "white")

    if not color_entries:
        cprint("[Error] No wrist color messages found.", "red")
        return False

    # --- Get header timestamps ---
    color_timestamps_ns = []
    for ts, rawdata, conn in color_entries:
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
        color_timestamps_ns.append(get_header_stamp_ns(msg, ts))
    color_timestamps_ns = np.array(color_timestamps_ns, dtype=np.int64)

    depth_timestamps_ns = np.array([], dtype=np.int64)
    if depth_entries:
        d_ts_list = []
        for ts, rawdata, conn in depth_entries:
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            d_ts_list.append(get_header_stamp_ns(msg, ts))
        depth_timestamps_ns = np.array(d_ts_list, dtype=np.int64)

    # --- Synchronize to ZED output timestamps ---
    output_ts = np.array(output_timestamps_ns, dtype=np.int64)
    color_sync_gaps_ms = []
    depth_sync_gaps_ms = []

    for idx, target_ts in enumerate(tqdm(output_ts, desc="  Syncing wrist frames", unit="frame")):
        # --- Color ---
        closest_color_idx = np.argmin(np.abs(color_timestamps_ns - target_ts))
        gap_ms = abs(int(color_timestamps_ns[closest_color_idx]) - int(target_ts)) / 1e6
        color_sync_gaps_ms.append(gap_ms)

        _, rawdata, conn = color_entries[closest_color_idx]
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
        color_img = decode_image_msg(msg, conn)
        cv2.imwrite(os.path.join(wrist_img_dir, f"frame_{idx:04d}.png"), color_img)

        # --- Depth ---
        if len(depth_entries) > 0:
            closest_depth_idx = np.argmin(np.abs(depth_timestamps_ns - target_ts))
            gap_ms_d = abs(int(depth_timestamps_ns[closest_depth_idx]) - int(target_ts)) / 1e6
            depth_sync_gaps_ms.append(gap_ms_d)

            _, rawdata_d, conn_d = depth_entries[closest_depth_idx]
            msg_d = typestore.deserialize_cdr(rawdata_d, conn_d.msgtype)
            depth_img = decode_image_msg(msg_d, conn_d)
            np.save(os.path.join(wrist_depth_dir, f"frame_{idx:04d}.npy"), depth_img)

    # --- Reports ---
    np.save(os.path.join(episode_dir, 'wrist_color_sync_gaps.npy'), np.array(color_sync_gaps_ms))
    generate_sync_report(episode_dir, color_sync_gaps_ms, label="ZED-to-Wrist-Color")

    if depth_sync_gaps_ms:
        np.save(os.path.join(episode_dir, 'wrist_depth_sync_gaps.npy'), np.array(depth_sync_gaps_ms))
        generate_sync_report(episode_dir, depth_sync_gaps_ms, label="ZED-to-Wrist-Depth")

    avg_gap = np.mean(color_sync_gaps_ms)
    max_gap = np.max(color_sync_gaps_ms)
    cprint(f"  [Wrist] Extracted {len(output_ts)} color + depth frames. "
           f"Color avg sync: {avg_gap:.2f}ms, max: {max_gap:.2f}ms",
           "green" if max_gap < 50 else "yellow")

    return True


def verify_extraction(episode_dir, expected_frames, check_wrist=True):
    img_dir = os.path.join(episode_dir, 'img')
    depth_dir = os.path.join(episode_dir, 'depth')
    state_path = os.path.join(episode_dir, 'state.npy')
    force_path = os.path.join(episode_dir, 'force.npy')

    img_files = glob.glob(os.path.join(img_dir, '*.png'))
    depth_files = glob.glob(os.path.join(depth_dir, '*.npy'))

    if len(img_files) != expected_frames: return False
    if len(depth_files) != expected_frames: return False
    if not os.path.exists(state_path): return False

    states = np.load(state_path)
    if len(states) != expected_frames: return False
    if np.isnan(states).any() or np.isinf(states).any(): return False

    if os.path.exists(force_path):
        forces = np.load(force_path)
        if len(forces) != expected_frames: return False

    if check_wrist and os.path.exists(os.path.join(episode_dir, 'realsense_bag')):
        wrist_img_dir = os.path.join(episode_dir, 'wrist_img')
        wrist_depth_dir = os.path.join(episode_dir, 'wrist_depth')

        if not os.path.exists(wrist_img_dir):
            return False
        wrist_img_files = glob.glob(os.path.join(wrist_img_dir, '*.png'))
        if len(wrist_img_files) != expected_frames:
            return False

        if os.path.exists(wrist_depth_dir):
            wrist_depth_files = glob.glob(os.path.join(wrist_depth_dir, '*.npy'))
            if len(wrist_depth_files) != expected_frames:
                return False

    return True


def is_already_extracted(episode_dir):
    """Checks if the episode has already been successfully extracted."""
    state_path = os.path.join(episode_dir, 'state.npy')

    if not os.path.exists(state_path):
        return False

    try:
        states = np.load(state_path)
        expected_frames = len(states)
        return verify_extraction(episode_dir, expected_frames, check_wrist=False)
    except Exception:
        return False


def extract_episode(episode_dir, target_fps=30, wrist_color_topic='/cam_right/cam_right/color/image_raw',
                    wrist_depth_topic='/cam_right/cam_right/aligned_depth_to_color/image_raw',
                    max_frames=1500):
    source_fps = 60
    skip_factor = source_fps // target_fps
    svo_path = os.path.join(episode_dir, "recording.svo2")
    if not os.path.exists(svo_path):
        return False

    img_dir = os.path.join(episode_dir, 'img')
    depth_dir = os.path.join(episode_dir, 'depth')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.depth_minimum_distance = 200
    init.depth_maximum_distance = 2000
    init.sdk_verbose = 1

    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        return False

    num_frames = cam.get_svo_number_of_frames()

    # --- Too long = likely a failed/stuck recording → delete ---
    if num_frames > max_frames:
        cam.close()
        cprint(f"  [DELETE] {os.path.basename(episode_dir)}: {num_frames} frames exceeds "
               f"{max_frames} limit ({num_frames/60:.1f}s). Deleting entire episode.", "red")
        shutil.rmtree(episode_dir)
        return False

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    runtime.confidence_threshold = 100
    runtime.texture_confidence_threshold = 100

    episode_states = []
    episode_forces = []
    sync_gaps_ms = []
    output_timestamps_ns = []
    last_known_positions = np.zeros(12, dtype=np.float32)
    last_known_forces = np.zeros(6, dtype=np.float32)

    dropped_state_count = 0
    last_ts = sl.Timestamp()
    last_ts.set_nanoseconds(0)

    expected_output_frames = num_frames // skip_factor
    cprint(f"\n--- Extracting {os.path.basename(episode_dir)} "
           f"({num_frames} frames @ 60fps → {expected_output_frames} frames @ {target_fps}fps) ---", "cyan")

    output_idx = 0

    with tqdm(total=num_frames, desc="Processing ZED Frames", unit="frame") as pbar:
        for frame_idx in range(num_frames):
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:

                current_ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                cam_ts_ns = current_ts.get_nanoseconds()

                custom_data = {}
                res = cam.retrieve_svo_data("joint_states", custom_data, last_ts, current_ts)
                state_updated_this_frame = False

                if res == sl.ERROR_CODE.SUCCESS and len(custom_data) > 0:
                    latest_key = sorted(custom_data.keys(), key=lambda t: t.get_nanoseconds())[-1]
                    svo_content = custom_data[latest_key]
                    json_string = (svo_content.get_content_as_string()
                                   if hasattr(svo_content, 'get_content_as_string')
                                   else svo_content.get_string_content())
                    try:
                        joint_states_batch = json.loads(json_string)
                        if len(joint_states_batch) > 0:
                            chosen_state = joint_states_batch[-1]
                            last_known_positions = np.array(chosen_state['positions'], dtype=np.float32)

                            if 'force_values' in chosen_state:
                                last_known_forces = np.array(chosen_state['force_values'], dtype=np.float32)

                            state_updated_this_frame = True
                            gap_ms = abs(cam_ts_ns - chosen_state['timestamp_ns']) / 1e6
                            sync_gaps_ms.append(gap_ms)
                    except json.JSONDecodeError:
                        pass

                last_ts = current_ts

                if frame_idx % skip_factor == 0:
                    cam.retrieve_image(image_mat, sl.VIEW.LEFT)
                    cv2.imwrite(
                        os.path.join(img_dir, f"frame_{output_idx:04d}.png"),
                        cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
                    )
                    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    np.save(
                        os.path.join(depth_dir, f"frame_{output_idx:04d}.npy"),
                        depth_mat.get_data()
                    )
                    episode_states.append(last_known_positions.copy())
                    episode_forces.append(last_known_forces.copy())
                    output_timestamps_ns.append(cam_ts_ns)

                    if not state_updated_this_frame:
                        dropped_state_count += 1

                    output_idx += 1

                pbar.update(1)

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break

    cam.close()

    # --- Save states ---
    states_array = np.stack(episode_states, axis=0)
    if len(states_array) > 1:
        if np.all(states_array[0] == 0) and not np.all(states_array[1] == 0):
            states_array[0] = states_array[1]
            cprint("[QA Info] Back-filled frame 0 with frame 1 data.", "cyan")
    np.save(os.path.join(episode_dir, 'state.npy'), states_array)

    # --- Save forces ---
    forces_array = np.stack(episode_forces, axis=0)
    if len(forces_array) > 1:
        if np.all(forces_array[0] == 0) and not np.all(forces_array[1] == 0):
            forces_array[0] = forces_array[1]
    np.save(os.path.join(episode_dir, 'force.npy'), forces_array)
    cprint(f"Saved force data: {forces_array.shape} (frames x 6 fingers, unit: grams)", "cyan")

    # --- Save ZED sync report ---
    np.save(os.path.join(episode_dir, 'sync_gaps.npy'), np.array(sync_gaps_ms))
    generate_sync_report(episode_dir, sync_gaps_ms, label="ZED-to-ROS")

    if dropped_state_count > 0:
        cprint(f"[QA Warning] Missed states for {dropped_state_count} frames.", "yellow")

    # --- Extract wrist camera (color + depth) from rosbag ---
    bag_path = os.path.join(episode_dir, 'realsense_bag')
    if os.path.exists(bag_path):
        cprint("\n  Extracting wrist camera (color + depth) from rosbag...", "cyan")
        wrist_ok = extract_wrist_camera(
            episode_dir, output_timestamps_ns, wrist_color_topic, wrist_depth_topic
        )
        if not wrist_ok:
            cprint("[QA Warning] Wrist camera extraction failed.", "yellow")
    else:
        cprint("[Info] No realsense_bag found, skipping wrist camera.", "white")

    # --- Save output timestamps ---
    np.save(os.path.join(episode_dir, 'frame_timestamps_ns.npy'),
            np.array(output_timestamps_ns, dtype=np.int64))

    # --- Final verification ---
    if verify_extraction(episode_dir, len(states_array)):
        cprint(f"[QA PASS] Episode fully synchronized and validated.", "green")
        return True
    return False


def main():
    args = parse_args()

    if not ROSBAGS_AVAILABLE:
        cprint("[Warning] 'rosbags' package not installed. Wrist camera extraction disabled.", "yellow")
        cprint("  Install with: pip install rosbags", "yellow")

    episodes = sorted([d for d in os.listdir(args.data_dir) if d.startswith('episode_')])

    success_count = 0
    deleted_count = 0
    for ep in episodes:
        episode_dir = os.path.join(args.data_dir, ep)
        if os.path.isdir(episode_dir):

            if is_already_extracted(episode_dir):
                cprint(f"\nSkipping {ep}: Already extracted and verified.", "yellow")
                success_count += 1
                continue

            if not os.path.exists(episode_dir):
                # Was deleted by extract_episode due to max_frames
                deleted_count += 1
                continue

            if extract_episode(episode_dir, target_fps=args.target_fps,
                               wrist_color_topic=args.wrist_color_topic,
                               wrist_depth_topic=args.wrist_depth_topic,
                               max_frames=args.max_frames):
                success_count += 1
            elif not os.path.exists(episode_dir):
                deleted_count += 1

    cprint(f"\nExtraction Complete: {success_count}/{len(episodes)} episodes passed all QA checks.",
           "green", attrs=['bold'])
    if deleted_count > 0:
        cprint(f"Deleted {deleted_count} episodes exceeding {args.max_frames} frame limit.",
               "red", attrs=['bold'])


if __name__ == '__main__':
    main()