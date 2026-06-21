#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pyzed.sl as sl
import json
import glob
from termcolor import cprint
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=f'/root/ros2_ws/data/episodes/{FOLDER_NAME}',
                        help='Raw data directory containing episode folders')
    parser.add_argument('--target_fps', type=int, default=30,
                        help='Target FPS for extraction (source is 60)')
    args = parser.parse_args()
    return args


def generate_sync_report(episode_dir, sync_gaps_ms):
    if not sync_gaps_ms:
        cprint("[QA Warning] No sync data to generate report.", "yellow")
        return

    gaps = np.array(sync_gaps_ms)
    avg_gap = np.mean(gaps)
    max_gap = np.max(gaps)
    min_gap = np.min(gaps)
    std_gap = np.std(gaps)

    cprint("\n--- Synchronization Profile ---", "cyan")
    cprint(f"Average Gap: {avg_gap:.2f} ms", "green" if avg_gap < 5 else "yellow")
    cprint(f"Maximum Gap: {max_gap:.2f} ms", "green" if max_gap < 10 else "red")
    cprint(f"Minimum Gap: {min_gap:.2f} ms", "white")
    cprint(f"Jitter (StdDev): {std_gap:.2f} ms\n", "white")

    plt.figure(figsize=(8, 5))
    plt.hist(gaps, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_gap, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {avg_gap:.2f}ms')
    plt.title("Camera-to-ROS Synchronization Delay")
    plt.xlabel("Time Gap (milliseconds)")
    plt.ylabel("Frequency (Frames)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plot_path = os.path.join(episode_dir, 'sync_histogram.png')
    plt.savefig(plot_path)
    plt.close()
    cprint(f"Saved sync histogram to: {plot_path}", "cyan")


def verify_extraction(episode_dir, expected_frames):
    img_dir = os.path.join(episode_dir, 'img')
    depth_dir = os.path.join(episode_dir, 'depth')
    arm_state_path = os.path.join(episode_dir, 'arm_state.npy')
    hand_joints_path = os.path.join(episode_dir, 'hand_joints.npy')
    hand_ft_path = os.path.join(episode_dir, 'hand_ft.npy')

    img_files = glob.glob(os.path.join(img_dir, '*.png'))
    depth_files = glob.glob(os.path.join(depth_dir, '*.npy'))

    if len(img_files) != expected_frames: return False
    if len(depth_files) != expected_frames: return False

    for path in [arm_state_path, hand_joints_path, hand_ft_path]:
        if not os.path.exists(path): return False
        data = np.load(path)
        if len(data) != expected_frames: return False
        if np.isnan(data).any() or np.isinf(data).any(): return False

    return True


def extract_episode(episode_dir, target_fps=30):
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
    expected_output_frames = num_frames // skip_factor
    cprint(f"\n--- Extracting {os.path.basename(episode_dir)} "
           f"({num_frames} frames @ 60fps → {expected_output_frames} frames @ {target_fps}fps) ---", "cyan")

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime.confidence_threshold = 100
    runtime.remove_saturated_areas = True
    runtime.texture_confidence_threshold = 100

    # State buffers
    episode_arm_states = []
    episode_hand_joints = []
    episode_hand_ft = []
    sync_gaps_ms = []

    # Last known values (carry forward if no update)
    last_arm_positions = np.zeros(6, dtype=np.float32)
    last_hand_joints = np.zeros(20, dtype=np.float32)
    last_hand_ft = np.zeros(30, dtype=np.float32)

    dropped_state_count = 0
    last_ts = sl.Timestamp()
    last_ts.set_nanoseconds(0)

    output_idx = 0

    with tqdm(total=num_frames, desc="Processing Frames", unit="frame") as pbar:
        for frame_idx in range(num_frames):
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:

                current_ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                cam_ts_ns = current_ts.get_nanoseconds()

                # Retrieve embedded robot state from SVO
                custom_data = {}
                res = cam.retrieve_svo_data("robot_state", custom_data, last_ts, current_ts)
                state_updated_this_frame = False

                if res == sl.ERROR_CODE.SUCCESS and len(custom_data) > 0:
                    latest_key = sorted(custom_data.keys(), key=lambda t: t.get_nanoseconds())[-1]
                    svo_content = custom_data[latest_key]
                    json_string = (svo_content.get_content_as_string()
                                   if hasattr(svo_content, 'get_content_as_string')
                                   else svo_content.get_string_content())
                    try:
                        state_batch = json.loads(json_string)
                        if len(state_batch) > 0:
                            # Take the most recent state in this batch
                            chosen_state = state_batch[-1]

                            # Arm
                            last_arm_positions = np.array(
                                chosen_state['arm_joints']['positions'], dtype=np.float32
                            )

                            # Hand joints (degrees, raw)
                            last_hand_joints = np.array(
                                chosen_state['hand_joints'], dtype=np.float32
                            )

                            # Hand F/T (raw, tip-local)
                            last_hand_ft = np.array(
                                chosen_state['hand_ft'], dtype=np.float32
                            )

                            state_updated_this_frame = True

                            gap_ms = abs(cam_ts_ns - chosen_state['timestamp_ns']) / 1e6
                            sync_gaps_ms.append(gap_ms)
                    except json.JSONDecodeError:
                        pass

                last_ts = current_ts

                # Downsample: only save every skip_factor-th frame
                if frame_idx % skip_factor == 0:
                    # RGB
                    cam.retrieve_image(image_mat, sl.VIEW.LEFT)
                    cv2.imwrite(
                        os.path.join(img_dir, f"frame_{output_idx:04d}.png"),
                        cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
                    )

                    # Depth
                    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    np.save(
                        os.path.join(depth_dir, f"frame_{output_idx:04d}.npy"),
                        depth_mat.get_data()
                    )

                    # States (raw)
                    episode_arm_states.append(last_arm_positions.copy())
                    episode_hand_joints.append(last_hand_joints.copy())
                    episode_hand_ft.append(last_hand_ft.copy())

                    if not state_updated_this_frame:
                        dropped_state_count += 1

                    output_idx += 1

                pbar.update(1)

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break

    cam.close()

    # Save arrays
    arm_states_array = np.stack(episode_arm_states, axis=0)        # (N, 6)
    hand_joints_array = np.stack(episode_hand_joints, axis=0)      # (N, 20)
    hand_ft_array = np.stack(episode_hand_ft, axis=0)              # (N, 30)

    # Back-fill frame 0 if needed
    if len(arm_states_array) > 1:
        if np.all(arm_states_array[0] == 0) and not np.all(arm_states_array[1] == 0):
            arm_states_array[0] = arm_states_array[1]
            hand_joints_array[0] = hand_joints_array[1]
            hand_ft_array[0] = hand_ft_array[1]
            cprint("[QA Info] Back-filled frame 0 with frame 1 data.", "cyan")

    np.save(os.path.join(episode_dir, 'arm_state.npy'), arm_states_array)
    np.save(os.path.join(episode_dir, 'hand_joints.npy'), hand_joints_array)
    np.save(os.path.join(episode_dir, 'hand_ft.npy'), hand_ft_array)
    np.save(os.path.join(episode_dir, 'sync_gaps.npy'), np.array(sync_gaps_ms))

    # Report
    generate_sync_report(episode_dir, sync_gaps_ms)

    if dropped_state_count > 0:
        cprint(f"[QA Warning] Missed states for {dropped_state_count}/{output_idx} frames.", "yellow")

    if verify_extraction(episode_dir, len(arm_states_array)):
        cprint(f"[QA PASS] Episode fully synchronized and validated.", "green")
        return True
    return False


def main():
    args = parse_args()
    episodes = sorted([d for d in os.listdir(args.data_dir) if d.startswith('episode_')])

    success_count = 0
    for ep in episodes:
        episode_dir = os.path.join(args.data_dir, ep)
        if os.path.isdir(episode_dir):
            if extract_episode(episode_dir, target_fps=args.target_fps):
                success_count += 1

    cprint(f"\nExtraction Complete: {success_count}/{len(episodes)} episodes passed all QA checks.",
           "green", attrs=['bold'])


if __name__ == '__main__':
    main()