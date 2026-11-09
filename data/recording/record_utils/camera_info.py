
import numpy as np  
import json
import os
import glob

try:
    import open3d as o3d
except ImportError:
    o3d = None

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

# Subdirectory under data/episodes/ that holds the collections.
# The default "external" is the Docker bind mount of the second hard drive
# (/mnt/data/data -> /ros2_ws/data/episodes/external), the same one
# dataset_builder/cli.py uses. Note this is a real mount inside the container,
# not the repo directory of the same name on the host - a collection copied to
# <repo>/data/episodes/external/ will NOT be visible here.
# Set EPISODES_SUBDIR="" for local collections directly under data/episodes/.
EPISODES_SUBDIR = os.environ.get("EPISODES_SUBDIR", "external")

# Resolve the path dynamically:
# 1. Current directory (data/record_utils)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Path resolution for episodes root
ros2_ws_root = os.path.abspath(os.path.join(current_dir, "../../.."))
episodes_roots = [
    os.path.join(current_dir, "..", "episodes"),
    os.path.join(ros2_ws_root, "data", "recording", "episodes"),
    os.path.join(ros2_ws_root, "data", "episodes"),
]

# 3. Locate DATA_PATH and camera_intrinsics.json with fallbacks
candidate_paths = []
for ep_root in episodes_roots:
    candidate_paths.extend([
        os.path.join(ep_root, EPISODES_SUBDIR, FOLDER_NAME),
        os.path.join(ep_root, FOLDER_NAME),
        os.path.join(ep_root, "external", FOLDER_NAME),
    ])

DATA_PATH = candidate_paths[0]
_INTRINSICS_PATH = None

for cand in candidate_paths:
    p = os.path.join(cand, "camera_intrinsics.json")
    if os.path.exists(p):
        DATA_PATH = cand
        _INTRINSICS_PATH = p
        break

if _INTRINSICS_PATH is None:
    # Scan for any existing collection containing camera_intrinsics.json
    found = []
    for ep_root in episodes_roots:
        found = sorted(glob.glob(os.path.join(ep_root, "external", "*", "camera_intrinsics.json")))
        if not found:
            found = sorted(glob.glob(os.path.join(ep_root, "*", "camera_intrinsics.json")))
        if found:
            break
    if found:
        _INTRINSICS_PATH = found[0]
        DATA_PATH = os.path.dirname(_INTRINSICS_PATH)
        print(
            f"[camera_info] Notice: camera_intrinsics.json not found for "
            f"FOLDER_NAME={FOLDER_NAME!r} under {candidate_paths[0]}. "
            f"Using available calibration from: {_INTRINSICS_PATH}"
        )
    else:
        print(
            f"[camera_info] WARNING: camera_intrinsics.json not found anywhere under "
            f"{episodes_roots[0]}. Using default ZED HD720 intrinsics."
        )

# ---------------------------------------------------------------------------
# Extrinsic matrices and Workspace boundaries
# ---------------------------------------------------------------------------
# Extrinsics are invariant to collection folders and always available.
camera_extrinsics = np.array([
    [ 0.99991326, -0.01290498, -0.00263139,  0.39029286],
    [-0.01167947, -0.96117350,  0.27569747,  0.53888558],
    [-0.00608709, -0.27564283, -0.96124085,  1.28573327],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
])

wrist_camera_extrinsics = np.array([
    [-0.98700938, -0.16046349,  0.00799649,  0.04177321],
    [ 0.16036115, -0.98698397, -0.01212213,  0.05306583],
    [ 0.00983757, -0.01068233,  0.99989455,  0.02709796],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
])

# Workspace boundaries
workspace = {
    'x': (-1.1,  0.50),   # um Bildmitte x = -0.39
    'y': (-1.80, -0.60),   # um Bildmitte y = -0.91
    'z': ( -1.00,  0.80)    # Tisch bis ~50 cm Hoehe
}

# ---------------------------------------------------------------------------
# Front (ZED) intrinsics - TWO variants, one source of truth
# ---------------------------------------------------------------------------
# camera_intrinsics       -> NATIVE HD720 (1280x720), straight from the ZED
#                            calibration. Use ONLY for point cloud generation,
#                            which runs on the un-resized retrieve_image /
#                            retrieve_measure output.
#
# camera_intrinsics_half  -> DERIVED 640x360. Everything that operates on the
#                            STORED images (front_rgb/*.png, front_depth/*.png)
#                            must use this one.
#
# 640x360 is exactly half of HD720, so a single ISOTROPIC factor 0.5 applies to
# both axes and to all four intrinsic parameters. The half variant is computed
# from the native one below - never duplicate it as a literal, or the two will
# drift apart the first time the camera is recalibrated.
FRONT_RESIZE_FACTOR = 0.5

if _INTRINSICS_PATH and os.path.exists(_INTRINSICS_PATH):
    with open(_INTRINSICS_PATH) as f:
        intr = json.load(f)
    front_intrinsics_native = {
        "width": int(intr["width"]),
        "height": int(intr["height"]),
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
    }
else:
    # Standard ZED HD720 default calibration
    front_intrinsics_native = {
        "width": 1280,
        "height": 720,
        "fx": 519.959228515625,
        "fy": 519.959228515625,
        "cx": 647.6746826171875,
        "cy": 363.5537109375,
    }

if (front_intrinsics_native["width"], front_intrinsics_native["height"]) != (1280, 720):
    print(
        f"[camera_info] WARNING: front intrinsics are "
        f"{front_intrinsics_native['width']}x{front_intrinsics_native['height']}, "
        f"expected 1280x720 (sl.RESOLUTION.HD720). The 0.5 factor below assumes "
        f"the stored images are 640x360; verify before trusting any reprojection."
    )

# Scale width/height as integers, the optical parameters as floats. cx/cy are
# scaled by the SAME factor as fx/fy - a pinhole model has no separate offset
# correction, and treating them differently is the classic silent reprojection
# bug.
front_intrinsics_half = {
    "width": int(round(front_intrinsics_native["width"] * FRONT_RESIZE_FACTOR)),
    "height": int(round(front_intrinsics_native["height"] * FRONT_RESIZE_FACTOR)),
    "fx": front_intrinsics_native["fx"] * FRONT_RESIZE_FACTOR,
    "fy": front_intrinsics_native["fy"] * FRONT_RESIZE_FACTOR,
    "cx": front_intrinsics_native["cx"] * FRONT_RESIZE_FACTOR,
    "cy": front_intrinsics_native["cy"] * FRONT_RESIZE_FACTOR,
}

# Resolution of the front images as they are written to disk by the extractor.
# (W, H) order, matching cv2.resize.
FRONT_IMAGE_SIZE = (front_intrinsics_half["width"],
                    front_intrinsics_half["height"])

# ---------------------------------------------------------------------------
# Front depth range (CAMERA frame, meters)
# ---------------------------------------------------------------------------
FRONT_DEPTH_RANGE_M = (0.80, 1.70)
FRONT_DEPTH_RANGE_MEASURED = True

if not FRONT_DEPTH_RANGE_MEASURED:
    print(
        f"[camera_info] WARNING: FRONT_DEPTH_RANGE_M={FRONT_DEPTH_RANGE_M} is a "
        f"PLACEHOLDER, not a measured value. Depth clipping/normalization will "
        f"run with it, but the normalized channel is only meaningful once the "
        f"real table/hand depths are filled in and "
        f"FRONT_DEPTH_RANGE_MEASURED is set to True."
    )

if o3d:
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        **front_intrinsics_native
    )
    # For anything working on the SAVED 640x360 images.
    camera_intrinsics_half = o3d.camera.PinholeCameraIntrinsic(
        **front_intrinsics_half
    )
else:
    camera_intrinsics = None
    camera_intrinsics_half = None

# ---------------------------------------------------------------------------
# Wrist camera intrinsics
# ---------------------------------------------------------------------------
_wrist_intr_path = os.path.join(DATA_PATH, "wrist_camera_intrinsics.json")
if not os.path.exists(_wrist_intr_path):
    # Try other candidate paths or fallbacks
    for cand in candidate_paths:
        p = os.path.join(cand, "wrist_camera_intrinsics.json")
        if os.path.exists(p):
            _wrist_intr_path = p
            break

if os.path.exists(_wrist_intr_path):
    with open(_wrist_intr_path) as f:
        wrist_intr = json.load(f)
else:
    wrist_intr = {
        "width": 640,
        "height": 480,
        "fx": 604.1893920898438,
        "fy": 603.694091796875,
        "cx": 326.4269714355469,
        "cy": 256.4339904785156,
    }

if o3d:
    wrist_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=int(wrist_intr["width"]),
        height=int(wrist_intr["height"]),
        fx=float(wrist_intr["fx"]),
        fy=float(wrist_intr["fy"]),
        cx=float(wrist_intr["cx"]),
        cy=float(wrist_intr["cy"]),
    )
else:
    wrist_camera_intrinsics = None
