import zarr
import numpy as np

z = zarr.open('/ros2_ws/data/zarr_datasets/test_collection.zarr', 'r')
actions = z['data/action'][:]
print(f"Action shape: {actions.shape}")
print(f"\nHand joints (cols 7:13):")
print(f"  min: {actions[:, 7:13].min(axis=0)}")
print(f"  max: {actions[:, 7:13].max(axis=0)}")
print(f"  mean: {actions[:, 7:13].mean(axis=0)}")
print(f"\nPose (cols 0:7):")
print(f"  min: {actions[:, 0:7].min(axis=0)}")
print(f"  max: {actions[:, 0:7].max(axis=0)}")

# Add to check_zarr_dataset.py
states = z['data/state'][:]
print(f"\nState (agent_pos) cols 0:7:")
print(f"  min: {states[:, 0:7].min(axis=0)}")
print(f"  max: {states[:, 0:7].max(axis=0)}")

pcs = z['data/point_cloud'][:]
print(f"\nPoint cloud shape: {pcs.shape}")
print(f"  xyz min: {pcs[:,:,:3].min(axis=(0,1))}")
print(f"  xyz max: {pcs[:,:,:3].max(axis=(0,1))}")