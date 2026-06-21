import os
import numpy as np
import glob
import shutil
import sys
import re

# --- CONFIGURATION ---
ROOT_DIR = "/ros2_ws/data/episodes/second_dataset_filtered"
PAD_COUNT = 10  # Number of frames to duplicate at the end
# ---------------------

def get_last_index(file_list):
    """
    Parses filenames like 'frame_1234.jpg' to find the highest index.
    """
    max_idx = -1
    last_file = None
    
    # Regex to capture the number at the end of the filename
    # Assumes format like "anything_12345.ext"
    pattern = re.compile(r'_(\d+)\.')
    
    for f in file_list:
        match = pattern.search(f)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
                last_file = f
    
    return max_idx, last_file

def process_episode(episode_path):
    print(f"Processing {os.path.basename(episode_path)}...")
    
    # 1. MODIFY NUMPY ARRAYS (Action & State)
    # ---------------------------------------
    npy_files = ['action.npy', 'state.npy']
    
    original_length = 0
    
    for npy_name in npy_files:
        path = os.path.join(episode_path, npy_name)
        if not os.path.exists(path):
            continue
            
        try:
            # Load data: Shape (N, D)
            data = np.load(path)
            original_length = len(data)
            
            if original_length == 0:
                continue

            # A. Add Success Signal
            # Create a column of zeros (N, 1)
            success_col = np.zeros((original_length, 1), dtype=data.dtype)
            
            # Set the LAST frame to 1
            success_col[-1] = 1.0
            
            # Concatenate: Shape becomes (N, D+1)
            data_with_success = np.hstack((data, success_col))
            
            # B. Duplicate Last Frame
            # Get the last frame (which now has success=1)
            last_frame = data_with_success[-1]
            
            # Create the padding array (10, D+1)
            padding = np.tile(last_frame, (PAD_COUNT, 1))
            
            # Combine them
            final_data = np.vstack((data_with_success, padding))
            
            # Save back to disk
            np.save(path, final_data)
            print(f"  - Updated {npy_name}: Added success bit & padded +{PAD_COUNT} frames. New Shape: {final_data.shape}")
            
        except Exception as e:
            print(f"  - [Error] Failed to process {npy_name}: {e}")
            return # Stop processing this folder to prevent mismatched data

    # 2. EXTEND TIMESTAMPS (If exists)
    # --------------------------------
    ts_path = os.path.join(episode_path, 'timestamps.npy')
    if os.path.exists(ts_path):
        try:
            ts_data = np.load(ts_path)
            if len(ts_data) > 0:
                last_ts = ts_data[-1]
                # specific average dt or just 0.03 (30fps)
                dt = 0.033 
                if len(ts_data) > 1:
                    dt = (ts_data[-1] - ts_data[0]) / len(ts_data)

                new_times = [last_ts + (i+1)*dt for i in range(PAD_COUNT)]
                final_ts = np.concatenate((ts_data, new_times))
                np.save(ts_path, final_ts)
                print(f"  - Updated timestamps.npy")
        except:
            pass

    # 3. DUPLICATE IMAGES
    # -------------------
    img_dirs = ['img', 'depth', 'point_cloud']
    
    for d_dir in img_dirs:
        dir_path = os.path.join(episode_path, d_dir)
        if not os.path.exists(dir_path):
            continue
            
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        
        if not files:
            continue
            
        # Find the last file (highest index)
        last_idx, last_filename = get_last_index(files)
        
        if last_filename is None:
            print(f"  - [Warning] Could not parse filenames in {d_dir}. Skipping padding.")
            continue
            
        src_file_path = os.path.join(dir_path, last_filename)
        
        # Determine extension and prefix
        name_part, ext_part = os.path.splitext(last_filename)
        
        # Reconstruct the prefix
        match = re.match(r'(.*_)(\d+)$', name_part)
        
        if match:
            prefix = match.group(1)
            number_str = match.group(2)
            zfill_width = len(number_str)
        else:
            # Fallback for simple names
            prefix = "frame_"
            zfill_width = 4
            
        # Create copies
        created_count = 0
        for i in range(1, PAD_COUNT + 1):
            new_idx = last_idx + i
            new_filename = f"{prefix}{str(new_idx).zfill(zfill_width)}{ext_part}"
            dst_file_path = os.path.join(dir_path, new_filename)
            
            try:
                shutil.copy2(src_file_path, dst_file_path)
                created_count += 1
            except OSError as e:
                print(f"    [Error] Copy failed: {e}")
                
        print(f"  - Padded {d_dir}/: Copied last frame {created_count} times.")

def main():
    # Find all episode folders
    search_pattern = os.path.join(ROOT_DIR, "episode_*")
    episode_folders = sorted(glob.glob(search_pattern))

    if not episode_folders:
        print("No 'episode_XXX' folders found.")
        return

    print(f"Found {len(episode_folders)} episodes.")
    print("This script will ADD a success column and PAD the end by 10 frames.")
    print("WARNING: This modifies files in-place.")
    print("Type 'y' to continue:")
    
    if input("> ").strip().lower() != 'y':
        print("Aborted.")
        return

    for folder in episode_folders:
        process_episode(folder)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()