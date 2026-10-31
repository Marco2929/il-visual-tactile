"""Dump the zarr layout that a Flex-pi LeRobot conversion would have to read.

Run inside the ros2_dev container:

    docker exec ros2_dev /ros2_ws/venv/bin/python \
        /ros2_ws/.../inspect_zarr_for_flexpi.py <zarr_path>

Prints, per array: shape, dtype, chunking, and the value range of the first
frame. The range matters for depth/pointcloud: Flex-pi wants uint16 depth in
millimetres plus intrinsics, and this is where we find out whether either
exists in the source.
"""
import sys

import numpy as np
import zarr

path = sys.argv[1] if len(sys.argv) > 1 else (
    "/ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr")

root = zarr.open(path, mode="r")
print(f"=== {path} ===")
print(root.tree())

print("\n--- arrays ---")
for grp_name, grp in [("", root)] + [(k, root[k]) for k in root.group_keys()]:
    for key in grp.array_keys():
        arr = grp[key]
        full = f"{grp_name}/{key}" if grp_name else key
        print(f"{full:28s} {str(arr.shape):24s} {str(arr.dtype):10s} "
              f"chunks={arr.chunks}")

print("\n--- first-frame value ranges ---")
for key in root.array_keys():
    arr = root[key]
    try:
        sample = np.asarray(arr[0])
    except Exception as exc:  # noqa: BLE001
        print(f"{key:28s} <unreadable: {exc}>")
        continue
    print(f"{key:28s} min={sample.min():<12.5g} max={sample.max():<12.5g} "
          f"mean={sample.mean():<12.5g} shape={sample.shape}")

print("\n--- episodes ---")
meta = root["meta"]
ends = np.asarray(meta["episode_ends"])
print(f"episode_ends: {ends.shape[0]} episodes, total frames {ends[-1]}")
lengths = np.diff(np.concatenate([[0], ends]))
print(f"episode length: min={lengths.min()} max={lengths.max()} "
      f"mean={lengths.mean():.1f}")
if "episode_sources" in meta:
    src = np.asarray(meta["episode_sources"])
    uniq, counts = np.unique(src, return_counts=True)
    print(f"episode_sources: {dict(zip(uniq.tolist(), counts.tolist()))}")