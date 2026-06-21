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
    """Generates statistical terminal output and saves a histogram of the sync gaps."""
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
        
    return True

def is_already_extracted(episode_dir):
    """Checks if the episode has already been successfully extracted based on existing files."""
    state_path = os.path.join(episode_dir, 'state.npy')
    
    # If state.npy doesn't exist, it definitely hasn't been extracted
    if not os.path.exists(state_path):
        return False
        
    try:
        # Load states to get the expected frame count for this episode
        states = np.load(state_path)
        expected_frames = len(states)
        
        # Run standard QA verification to ensure nothing is missing or corrupted
        return verify_extraction(episode_dir, expected_frames)
    except Exception:
        # If loading the numpy array fails or any other error occurs, re-extract
        return False

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
    if cam.open(init) != sl.ERROR_CODE.SUCCESS: return False

    num_frames = cam.get_svo_number_of_frames()

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    runtime.confidence_threshold = 100
    runtime.texture_confidence_threshold = 100
    
    episode_states = []
    episode_forces = []
    sync_gaps_ms = []
    last_known_positions = np.zeros(12, dtype=np.float32)
    last_known_forces = np.zeros(6, dtype=np.float32)
    
    dropped_state_count = 0
    last_ts = sl.Timestamp()
    last_ts.set_nanoseconds(0)

    expected_output_frames = num_frames // skip_factor
    cprint(f"\n--- Extracting {os.path.basename(episode_dir)} "
           f"({num_frames} frames @ 60fps → {expected_output_frames} frames @ {target_fps}fps) ---", "cyan")

    output_idx = 0 

    with tqdm(total=num_frames, desc="Processing Frames", unit="frame") as pbar:
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
                    json_string = svo_content.get_content_as_string() if hasattr(svo_content, 'get_content_as_string') else svo_content.get_string_content()
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

                    if not state_updated_this_frame:
                        dropped_state_count += 1

                    output_idx += 1

                pbar.update(1)

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break

    cam.close()

    states_array = np.stack(episode_states, axis=0)
    if len(states_array) > 1:
        if np.all(states_array[0] == 0) and not np.all(states_array[1] == 0):
            states_array[0] = states_array[1]
            cprint("[QA Info] Back-filled frame 0 with frame 1 data.", "cyan")
    np.save(os.path.join(episode_dir, 'state.npy'), states_array)

    forces_array = np.stack(episode_forces, axis=0)
    if len(forces_array) > 1:
        if np.all(forces_array[0] == 0) and not np.all(forces_array[1] == 0):
            forces_array[0] = forces_array[1]
    np.save(os.path.join(episode_dir, 'force.npy'), forces_array)
    cprint(f"Saved force data: {forces_array.shape} (frames x 6 fingers, unit: grams)", "cyan")
    
    np.save(os.path.join(episode_dir, 'sync_gaps.npy'), np.array(sync_gaps_ms))

    generate_sync_report(episode_dir, sync_gaps_ms)

    if dropped_state_count > 0:
        cprint(f"[QA Warning] Missed states for {dropped_state_count} frames.", "yellow")

    if verify_extraction(episode_dir, len(states_array)):
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
            
            # --- Check if already extracted ---
            if is_already_extracted(episode_dir):
                cprint(f"\nSkipping {ep}: Already extracted and verified.", "yellow")
                success_count += 1
                continue
                
            if extract_episode(episode_dir, target_fps=args.target_fps):
                success_count += 1
            
    cprint(f"\nExtraction Complete: {success_count}/{len(episodes)} episodes passed all QA checks.", "green", attrs=['bold'])

if __name__ == '__main__':
    main()