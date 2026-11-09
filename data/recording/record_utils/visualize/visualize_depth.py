import numpy as np
import matplotlib.pyplot as plt
import os

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

# Load one of your saved depth frames
depth_data = np.load(f'/ros2_ws/data/episodes/{FOLDER_NAME}/episode_002/depth/frame_0200.npy')

valid_depths = depth_data[np.isfinite(depth_data) & (depth_data > 0)]

print(f"Minimum distance: {np.min(valid_depths):.4f}")
print(f"Maximum distance: {np.max(valid_depths):.4f}")
print(f"Median distance:  {np.median(valid_depths):.4f}")

if np.max(valid_depths) < 20:
    print("\nConclusion: Your data is in METERS.")
    print("Set depth_scale = 1.0 in your Open3D script.")
else:
    print("\nConclusion: Your data is in MILLIMETERS.")
    print("Set depth_scale = 1000.0 in your Open3D script.")

plt.figure(figsize=(10, 6))
# 'magma' or 'viridis' are good for depth; 
# vmin/vmax should be set based on ZED's 0.01 to 15m range
plt.imshow(depth_data, cmap='magma', vmin=100, vmax=1600)  # in mm
plt.colorbar(label='Distance (millimeters)')
plt.title('Raw Depth Image Visualization')
plt.show()
