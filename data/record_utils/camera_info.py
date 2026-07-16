import open3d as o3d 
import numpy as np  
import json
import os

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

# Dynamische Pfadermittlung:
# 1. Ermittelt das aktuelle Verzeichnis (data/record_utils)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Geht zwei Ebenen nach oben ins ros2_ws Verzeichnis
ros2_ws_root = os.path.abspath(os.path.join(current_dir, "../.."))

# 3. Setzt den korrekten absoluten Pfad zusammen
DATA_PATH = os.path.join(ros2_ws_root, "data", "episodes", FOLDER_NAME)

with open(os.path.join(DATA_PATH, "camera_intrinsics.json")) as f:
    intr = json.load(f)

camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=intr["width"],
    height=intr["height"],
    fx=intr["fx"],
    fy=intr["fy"],
    cx=intr["cx"],
    cy=intr["cy"]
)

with open(os.path.join(DATA_PATH, "wrist_camera_intrinsics.json")) as f:
    intr = json.load(f)

wrist_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=intr["width"],
    height=intr["height"],
    fx=intr["fx"],
    fy=intr["fy"],
    cx=intr["cx"],
    cy=intr["cy"]
)

# Set extrinsic matrix
camera_extrinsics = np.array([
    [ 0.86730917,  0.24528566, -0.43313940,  0.85272407],
    [ 0.49681202, -0.48051557,  0.72269122, -0.87147040],
    [-0.03086443, -0.84198559, -0.53861643,  0.47582620],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
])

wrist_camera_extrinsics = np.array([
    [-0.98700938, -0.16046349,  0.00799649,  0.04177321],
    [ 0.16036115, -0.98698397, -0.01212213,  0.05306583],
    [ 0.00983757, -0.01068233,  0.99989455,  0.02709796],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
])

# Define workspace boundaries
workspace = {
    'x': (-1.0, 0),    # R
    'y': (0.0, 0.8),   # G
    'z': (0.03, 1.0)   # B
}