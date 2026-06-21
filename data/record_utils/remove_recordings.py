#!/usr/bin/env python3
import os
import numpy as np
import glob
import shutil

# --- CONFIGURATION ---
# If FOLDER_NAME is not set, 'box_opening_collection' is used by default
FOLDER_NAME = os.environ.get("FOLDER_NAME", "box_opening_collection")

# Dynamic path: Builds the path based on your current working directory (ros2_ws)
ROOT_DIR = os.path.join(os.getcwd(), "data", "episodes", FOLDER_NAME)
MIN_FRAMES = 70
# ---------------------

def main():
    search_pattern = os.path.join(ROOT_DIR, "episode_*")
    episode_folders = sorted(glob.glob(search_pattern))

    if not episode_folders:
        print(f"No 'episode_XXX' folders found in {ROOT_DIR}.")
        return

    print(f"Found {len(episode_folders)} episodes. Searching for lengths < {MIN_FRAMES} frames...")
    print("-" * 50)

    deleted_count = 0

    for folder in episode_folders:
        folder_name = os.path.basename(folder)
        state_path = os.path.join(folder, 'state.npy')
        
        if not os.path.exists(state_path):
            print(f"  [Warning] {folder_name} is missing state.npy. Skipping.")
            continue
            
        try:
            # mmap_mode='r' reads only metadata without loading the entire array into RAM
            data = np.load(state_path, mmap_mode='r')
            current_len = len(data)
            
            if current_len < MIN_FRAMES:
                print(f"[{folder_name}] Length: {current_len} -> DELETING")
                try:
                    shutil.rmtree(folder)
                    deleted_count += 1
                except OSError as e:
                    print(f"  -> [Error] Could not delete {folder_name}: {e}")
            else:
                pass # Silence for valid episodes

        except Exception as e:
            print(f"  [Error] Could not read {folder_name}/state.npy: {e}")

    print("-" * 50)
    print(f"Cleanup complete. Deleted {deleted_count} out of {len(episode_folders)} episodes.")

if __name__ == "__main__":
    main()