
import numpy as np  
import json
import os

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

# 2. Two levels up, to the ros2_ws root
ros2_ws_root = os.path.abspath(os.path.join(current_dir, "../.."))

# 3. Assemble the absolute path
DATA_PATH = os.path.join(
    ros2_ws_root, "data", "episodes", EPISODES_SUBDIR, FOLDER_NAME
)

_INTRINSICS_PATH = os.path.join(DATA_PATH, "camera_intrinsics.json")

if not os.path.exists(_INTRINSICS_PATH):
    raise FileNotFoundError(
        f"camera_intrinsics.json not found at: {_INTRINSICS_PATH}\n"
        f"  FOLDER_NAME={FOLDER_NAME!r}  EPISODES_SUBDIR={EPISODES_SUBDIR!r}\n"
        "Adjust FOLDER_NAME (collection name) or EPISODES_SUBDIR. "
        "EPISODES_SUBDIR='' uses data/episodes/<FOLDER_NAME> directly."
    )

with open(_INTRINSICS_PATH) as f:
    intr = json.load(f)

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

front_intrinsics_native = {
    "width": int(intr["width"]),
    "height": int(intr["height"]),
    "fx": float(intr["fx"]),
    "fy": float(intr["fy"]),
    "cx": float(intr["cx"]),
    "cy": float(intr["cy"]),
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
# Used to clip + normalize the stored depth maps in
# dataset_builder/processing/depth.py.
#
# This is deliberately NOT derived from `workspace` below: workspace is in WORLD
# coordinates after camera_extrinsics + the extra 180 deg Z rotation applied in
# point_cloud_utils.py, and its 'z' entry is height above the floor. The stored
# depth map holds camera-frame Z distance instead. Mixing the two would clip the
# table away entirely.
#
# Derivation (episode_002_success, 171 frames):
#   * table surface sits at ~1.5 m camera depth (measured valid median was
#     1268 mm, pulled below the table by the hand/foil in frame),
#   * workspace['z'] allows 0.5 m of height above the table, so the closest
#     relevant surface is ~1.5 - 0.5 = 1.0 m,
#   * everything at 2.0 m is background beyond the table edge; the ZED's
#     depth_maximum_distance clamps there, which is why valid_max was exactly
#     2000 mm.
# The upper bound is deliberately BELOW that background plateau: clipping it
# away turns the background into "invalid" instead of spending ~1/3 of the
# normalized range on a constant wall the policy must learn to ignore.
# A tight range matters because it IS the full dynamic range of the normalized
# depth channel - a wide one compresses the table-to-hand signal, which is the
# only part that carries slip information, into a few percent of the scale.
#
# Upper bound 1.70 instead of a tight 1.55: the table edge sits at ~1.5 m TODAY,
# but the table height is adjustable. A table lowered by 15 cm increases its
# camera depth by the same amount, and anything past the upper bound becomes
# "invalid" - i.e. a slightly lower table would silently blank out the entire
# work surface. The 1.70 m plateau still stays clear of the 2.0 m background
# wall, so the headroom costs ~10% of the range and buys robustness against a
# re-adjusted table.
#
# Lower bound 0.80 rather than 1.00: the measured valid MINIMUM over this
# episode was 451 mm, so the hand does come closer to the lens than 1.0 m.
# Clipping at 1.00 would mark those pixels invalid exactly during the contact
# phase - the moment the depth map matters most. 0.80 keeps the realistic
# approach range while still excluding the 451 mm outliers, which sit close to
# the ZED's own depth_minimum_distance (0.2 m) and are more likely stereo noise
# than the hand.
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

with open(os.path.join(DATA_PATH, "wrist_camera_intrinsics.json")) as f:
    intr = json.load(f)

if o3d:
    wrist_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=intr["width"],
        height=intr["height"],
        fx=intr["fx"],
        fy=intr["fy"],
        cx=intr["cx"],
        cy=intr["cy"]
    )

# Extrinsic matrices
# camera_extrinsics = np.array([
#     [-0.99991326,  0.01290498,  0.00263139, -0.39029286],
#     [ 0.01167947,  0.96117350, -0.27569747, -0.53888558],
#     [-0.00608709, -0.27564283, -0.96124085,  1.28573327],
#     [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
# ])

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