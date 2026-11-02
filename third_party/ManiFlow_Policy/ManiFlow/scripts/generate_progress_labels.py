#!/usr/bin/env python3
"""Generate or update progress labels directly in an existing Zarr dataset.

Allows recomputing and tuning progress labels (e.g. variant, peel_param, thresholds)
completely independently without having to recompute the DINOv3 feature cache.
Also supports instantly copying/syncing the computed progress array to a twin Zarr dataset.
"""

import argparse
import os
import sys

import numpy as np
import zarr
from tqdm import tqdm

# Import compute_progress from dataset_builder
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
builder_path = os.path.join(repo_root, "data/dataset_builder")
if builder_path not in sys.path:
    sys.path.insert(0, builder_path)

try:
    from processing.progress import compute_progress
except ImportError:
    # Docker path fallback
    if "/ros2_ws/data/dataset_builder" not in sys.path:
        sys.path.insert(0, "/ros2_ws/data/dataset_builder")
    from processing.progress import compute_progress


def write_progress_to_zarr(zarr_path: str, progress_key: str, all_progress: np.ndarray):
    """Write or overwrite the 1D progress array into a Zarr dataset."""
    root = zarr.open(zarr_path, mode='a')
    group = root['data'] if 'data' in root else root
    num_frames = len(all_progress)

    if progress_key in group:
        print(f"[Progress Labels] Overwriting existing '{progress_key}' array in {zarr_path}.")
        group[progress_key][:] = all_progress
    else:
        group.create_dataset(
            progress_key,
            data=all_progress,
            shape=(num_frames,),
            chunks=(min(1000, num_frames),),
            dtype='float32',
            overwrite=True,
        )
    print(f"[Progress Labels] Successfully wrote {num_frames} labels into '{progress_key}' in {zarr_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate progress labels in a Zarr dataset.")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to primary Zarr dataset.")
    parser.add_argument("--progress_key", type=str, default="progress", help="Target key for progress labels.")
    parser.add_argument("--variant", type=str, default="conservative", choices=["conservative", "legacy80"],
                        help="Progress curve variant: 'conservative' (y_grasp=0.40) or 'legacy80' (y_grasp=0.80).")
    parser.add_argument("--peel_param", type=str, default="frames", choices=["frames", "height"],
                        help="Phase 5 driver: 'frames' (time ramp) or 'height' (relative z travel).")
    parser.add_argument("--closed_frac", type=float, default=None,
                        help="Fraction of hand span to consider fingers closed on foil (default: 0.45).")
    parser.add_argument("--min_closed_frames", type=int, default=None,
                        help="Minimum sustained closed frames for a grasp attempt (default: 6).")
    parser.add_argument("--copy_to", type=str, default=None,
                        help="Optional twin Zarr path to copy the computed labels into immediately.")
    args = parser.parse_args()

    print(f"[Progress Labels] Opening Zarr: {args.zarr_path}")
    root = zarr.open(args.zarr_path, mode='a')
    group = root['data'] if 'data' in root else root

    if 'state' not in group:
        raise KeyError(f"'state' key not found in {args.zarr_path}. Available: {list(group.keys())}")

    state_arr = group['state']
    episode_ends = root['meta/episode_ends'][:] if 'meta/episode_ends' in root else group['meta/episode_ends'][:]
    num_frames = state_arr.shape[0]
    n_episodes = len(episode_ends)

    print(f"[Progress Labels] {n_episodes} episodes, {num_frames} total frames, state shape: {state_arr.shape}")
    print(f"[Progress Labels] Variant: {args.variant}, peel_param: {args.peel_param}")

    progress_dict = {
        "variant": args.variant,
        "peel_param": args.peel_param,
    }
    if args.closed_frac is not None:
        progress_dict["closed_frac"] = args.closed_frac
    if args.min_closed_frames is not None:
        progress_dict["min_closed_frames"] = args.min_closed_frames

    all_progress = np.zeros(num_frames, dtype=np.float32)

    for ep_idx in tqdm(range(n_episodes), desc="Computing progress curves"):
        start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        end = int(episode_ends[ep_idx])
        ep_len = end - start

        ep_state = state_arr[start:end]

        if ep_state.shape[-1] >= 15:
            ee_pose = ep_state[:, :7]
            hand_state = ep_state[:, 7:15]
            ep_progress = compute_progress(
                num_frames=ep_len,
                ee_pose=ee_pose,
                hand_state=hand_state,
                progress_dict=progress_dict,
            )
        else:
            ee_pose = ep_state[:, :3] if ep_state.shape[-1] >= 3 else None
            finger_sensor = ep_state[:, -1]
            ep_progress = compute_progress(
                num_frames=ep_len,
                ee_pose=ee_pose,
                finger_sensor=finger_sensor,
                progress_dict=progress_dict,
            )

        all_progress[start:end] = ep_progress.astype(np.float32)

    # Write to primary dataset
    write_progress_to_zarr(args.zarr_path, args.progress_key, all_progress)

    # If twin dataset specified, copy directly
    if args.copy_to:
        print(f"[Progress Labels] Syncing progress labels to twin dataset: {args.copy_to}")
        write_progress_to_zarr(args.copy_to, args.progress_key, all_progress)


if __name__ == "__main__":
    main()
