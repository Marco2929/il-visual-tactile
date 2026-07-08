#!/usr/bin/env python3
"""
Quick look at the merged (front + wrist) point cloud for a SINGLE frame,
built directly from the RAW dataset (SVO + realsense rosbag).

Pipeline (from the paper, NOT point_cloud_utils.py):
  1. Build a dense colored cloud per camera from depth + intrinsics.
  2. Transform each into the world (robot base_link) frame:
       - front (static ZED):  FRONT_EXTRINSICS + z-180 fix  (as in extract_episodes)
       - wrist (on flange):   FK(arm_joints -> end_link) @ WRIST_EXTRINSICS
         (reproduces transform_wrist_pc_to_base(); no z-180 fix on the wrist).
     The wrist camera rides on the moving end-effector, so a static extrinsic
     is not enough - we run forward kinematics on the arm joints for this frame
     (read from the SVO 'joint_states' custom data), exactly like the dataset
     builder's transform_wrist_pc_to_base().
  3. Concatenate -> single dense colored cloud (~200k-400k pts).
  4. Crop with the fixed workspace box (camera_info.workspace).
  5. FPS-downsample the cropped cloud to 8192 points.

Usage:
  python3 data/record_extractor/visualize_merged_pc.py \
      --episode /root/ros2_ws/data/episodes/foil_lifting_collection/episode_002_success \
      --frame_index 0 --distinguish

  # Color front points blue and wrist points red so you can tell them apart:
  python3 visualize_merged_pc.py --episode <dir> --frame_index 0 --distinguish
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
import open3d as o3d
import kinpy as kp

sys.path.insert(0, "/root/ros2_ws/data/record_utils")

from camera_info import (
    camera_intrinsics as FRONT_INTRINSICS,
    camera_extrinsics as FRONT_EXTRINSICS,
    wrist_camera_intrinsics as WRIST_INTRINSICS,
    wrist_camera_extrinsics as WRIST_EXTRINSICS,
    workspace as WORKSPACE,
)

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    Reader = None

# Kinematics defaults (match cond_preprocess / dataset_builder). The wrist
# camera is mounted on the moving flange, so its world transform needs FK on
# the arm joints, NOT a static extrinsic. Read from env like the other scripts.
DEFAULT_URDF = "/root/ros2_ws/data/record_extractor/inspire_strong_hand_ur10e_flange_workcell.urdf"
# Match the dataset builder config (foil_lifting_wrist.yaml -> end_link).
# transform_wrist_pc_to_base() runs FK to this link and applies the extrinsic
# on top of it, so we reproduce EXACTLY that to see what the dataset looks like.
DEFAULT_END_LINK = os.environ.get("ROBOT_TIP_FRAME", "right_inspire_base_link")
ARM_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Colors used when --distinguish is set (RGB in [0,1]).
FRONT_COLOR = np.array([0.1, 0.4, 1.0])   # blue  = front / scene camera
WRIST_COLOR = np.array([1.0, 0.2, 0.1])   # red   = wrist camera


# ─────────────────────────────────────────────────────────────
# Raw wrist decoding (copied minimal, no point-cloud logic)
# ─────────────────────────────────────────────────────────────
def decode_image_msg(msg, conn):
    if 'CompressedImage' in conn.msgtype:
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    enc = msg.encoding
    h, w = msg.height, msg.width
    dtype_map = {
        '16UC1': (np.uint16, (h, w)),
        'mono16': (np.uint16, (h, w)),
        '32FC1': (np.float32, (h, w)),
        'mono8': (np.uint8, (h, w)),
        'rgb8': (np.uint8, (h, w, 3)),
        'bgr8': (np.uint8, (h, w, 3)),
        'bgra8': (np.uint8, (h, w, 4)),
        'rgba8': (np.uint8, (h, w, 4)),
    }
    if enc in dtype_map:
        dt, shape = dtype_map[enc]
        img = np.frombuffer(bytes(msg.data), dtype=dt).reshape(shape)
    else:
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, -1)

    if enc == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc == 'bgra8':
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif enc == 'rgba8':
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def header_stamp_ns(msg, fallback):
    try:
        s = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        if s > 0:
            return s
    except Exception:
        pass
    return fallback


# ─────────────────────────────────────────────────────────────
# Plain Open3D cloud builder (paper-style, no filtering tricks)
# ─────────────────────────────────────────────────────────────
def build_camera_cloud(color_bgr, depth, intrinsics):
    """Return an Open3D cloud in the CAMERA frame (no extrinsic applied).
    color_bgr: (H,W,3) uint8; depth: (H,W) in meters or mm (auto-detected).
    """
    depth = depth.astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0

    # Normalize to meters if the source is in mm.
    valid = depth > 0
    if np.any(valid) and np.median(depth[valid]) > 20.0:
        depth = depth / 1000.0

    color_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])

    o3d_color = o3d.geometry.Image(color_rgb.astype(np.uint8))
    o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=1.0,      # depth already in meters
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)


def transform_front_to_base(pcd, extrinsic):
    """Static ZED: apply extrinsic, then the z-180 fix used in extract_episodes
    so the cloud matches the cached front_pc frame."""
    pcd.transform(extrinsic)
    R_fix = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
    pcd.rotate(R_fix, center=(0, 0, 0))
    return pcd


def fk_matrix(chain, arm_joints):
    """FK: base <- end_link (4x4)."""
    fk = chain.forward_kinematics(dict(zip(ARM_JOINT_NAMES, arm_joints)))
    return fk.matrix()


def transform_wrist_to_base(pcd, T_base_flange, extrinsic, invert_extrinsic=False):
    """Reproduce transform_wrist_pc_to_base():
        T_base_cam = FK(arm_joints -> end_link) @ WRIST_EXTRINSICS
    (no z-180 fix, matching merge_point_clouds()).

    The hand-in-eye calibration saved T_camera2gripper; the composition above is
    only correct if that means T_gripper<-camera. If the wrist is rigidly offset
    (moves with the arm but sits in the wrong place), the extrinsic direction is
    likely reversed -> try invert_extrinsic=True.
    """
    ext = np.linalg.inv(extrinsic) if invert_extrinsic else extrinsic
    T_base_cam = T_base_flange @ ext  # base <- end_link <- camera
    print(f"      [debug] invert_extrinsic={invert_extrinsic}")
    print(f"      [debug] FK (end_link) translation: {np.round(T_base_flange[:3, 3], 4)}")
    print(f"      [debug] T_base_cam   translation: {np.round(T_base_cam[:3, 3], 4)}")
    pcd.transform(T_base_cam)
    return pcd


def make_frame(T, size=0.15):
    """Coordinate-frame triad placed at pose T (4x4)."""
    f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    f.transform(T)
    return f


def run_diagnose(args, front, arm_state, f_ts):
    """Build raw wrist camera cloud once, apply all 4 transform hypotheses,
    and report each one's table-plane gap vs the (fixed) front table.

    The CORRECT hypothesis has a gap near 0 that stays constant across frames.
    """
    w_color, w_depth = read_wrist_frame(args.episode, f_ts)
    if w_color is None or w_depth is None:
        print("  [diagnose] no wrist frame.")
        return

    ft = segment_plane_z(front)
    front_z = ft[0] if ft else None
    print(f"\n===== DIAGNOSE (frame {args.frame_index}) =====")
    print(f"  front table Z = {front_z:+.4f} m" if front_z is not None
          else "  front table: no plane")

    chains = {
        'tool0': setup_kinematics(args.urdf_path, 'tool0'),
        'right_inspire_base_link':
            setup_kinematics(args.urdf_path, 'right_inspire_base_link'),
    }

    for link, chain in chains.items():
        T = fk_matrix(chain, arm_state[:6])
        for invert in (False, True):
            ext = np.linalg.inv(WRIST_EXTRINSICS) if invert else WRIST_EXTRINSICS
            w = build_camera_cloud(w_color, w_depth, WRIST_INTRINSICS)
            w.transform(T @ ext)
            wt = segment_plane_z(w)
            if wt and front_z is not None:
                gap = (wt[0] - front_z) * 100
                tag = 'ext' if not invert else 'inv(ext)'
                print(f"  [{link:24s} | {tag:8s}] "
                      f"wrist table Z={wt[0]:+.4f}  gap={gap:+.2f}cm")
    print("  (correct = gap ~0 AND identical across frames 50/200/400)")
    print("====================================\n")


def measure_offset_icp(wrist, front, threshold=0.05):
    """Register wrist -> front with ICP; the translation is the residual offset.

    Returns (T_icp, rmse, fitness). We copy the clouds so the caller's geometry
    is untouched (the visualizer still shows the UN-registered clouds).
    """
    src = o3d.geometry.PointCloud(wrist)
    tgt = o3d.geometry.PointCloud(front)
    # Point-to-plane needs normals on the target.
    tgt.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return reg.transformation, reg.inlier_rmse, reg.fitness


def segment_plane_z(pcd, dist=0.01):
    """RANSAC-fit the dominant plane (table); return its mean height along the
    plane normal's world Z as a coarse 'table height' estimate."""
    if len(pcd.points) < 100:
        return None
    plane, inliers = pcd.segment_plane(distance_threshold=dist,
                                       ransac_n=3, num_iterations=200)
    pts = np.asarray(pcd.points)[inliers]
    return float(np.median(pts[:, 2])), plane, len(inliers)


# ─────────────────────────────────────────────────────────────
# Raw readers
# ─────────────────────────────────────────────────────────────
def read_front_frame(episode_dir, frame_index):
    """Grab RGB + depth + timestamp + latest arm joints at `frame_index`.

    Joint states are read from the SVO 'joint_states' custom data (same source
    extract_episodes uses) so we can run FK for the wrist camera.
    """
    import pyzed.sl as sl
    svo = os.path.join(episode_dir, "recording.svo2")

    init = sl.InitParameters()
    init.set_from_svo_file(svo)
    init.svo_real_time_mode = False
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.2
    init.depth_maximum_distance = 2.0
    init.sdk_verbose = 0

    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Failed to open SVO")

    rt = sl.RuntimeParameters()
    rt.confidence_threshold = 100
    rt.texture_confidence_threshold = 100
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    last_positions = np.zeros(12, dtype=np.float32)
    last_ts = sl.Timestamp()
    last_ts.set_nanoseconds(0)

    # Grab up to and including frame_index, tracking latest joint states.
    for i in range(frame_index + 1):
        if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            cam.close()
            raise RuntimeError(f"Could not grab frame {i}")

        current_ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        custom_data = {}
        res = cam.retrieve_svo_data("joint_states", custom_data, last_ts, current_ts)
        if res == sl.ERROR_CODE.SUCCESS and len(custom_data) > 0:
            latest_key = sorted(custom_data.keys(),
                                key=lambda t: t.get_nanoseconds())[-1]
            sc = custom_data[latest_key]
            js = (sc.get_content_as_string() if hasattr(sc, 'get_content_as_string')
                  else sc.get_string_content())
            try:
                batch = json.loads(js)
                if len(batch) > 0:
                    last_positions = np.array(batch[-1]['positions'], dtype=np.float32)
            except json.JSONDecodeError:
                pass
        last_ts = current_ts

    cam.retrieve_image(img_mat, sl.VIEW.LEFT)
    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    color_bgr = cv2.cvtColor(img_mat.get_data(), cv2.COLOR_BGRA2BGR)
    depth = depth_mat.get_data().copy()
    ts_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
    cam.close()
    return color_bgr, depth, ts_ns, last_positions


def read_wrist_frame(episode_dir, target_ts_ns,
                     color_topic='/cam_right/cam_right/color/image_raw',
                     depth_topic='/cam_right/cam_right/aligned_depth_to_color/image_raw'):
    if Reader is None:
        print("[warn] rosbags not installed; skipping wrist.")
        return None, None

    bag = os.path.join(episode_dir, 'realsense_bag')
    if not os.path.exists(bag):
        print("[warn] no realsense_bag; skipping wrist.")
        return None, None

    ts = get_typestore(Stores.ROS2_JAZZY)
    color_entries, depth_entries = [], []
    with Reader(bag) as reader:
        color_conns = [c for c in reader.connections if c.topic == color_topic]
        depth_conns = [c for c in reader.connections if c.topic == depth_topic]
        for conn, t, raw in reader.messages(connections=color_conns + depth_conns):
            if conn.topic == color_topic:
                color_entries.append((t, raw, conn))
            else:
                depth_entries.append((t, raw, conn))

    if not color_entries:
        print("[warn] no wrist color frames; skipping wrist.")
        return None, None

    def stamps(entries):
        out = []
        for t, raw, conn in entries:
            m = ts.deserialize_cdr(raw, conn.msgtype)
            out.append(header_stamp_ns(m, t))
        return np.array(out, dtype=np.int64)

    c_ts = stamps(color_entries)
    ci = int(np.argmin(np.abs(c_ts - target_ts_ns)))
    _, raw, conn = color_entries[ci]
    color_bgr = decode_image_msg(ts.deserialize_cdr(raw, conn.msgtype), conn)

    depth = None
    if depth_entries:
        d_ts = stamps(depth_entries)
        di = int(np.argmin(np.abs(d_ts - target_ts_ns)))
        _, rawd, connd = depth_entries[di]
        depth = decode_image_msg(ts.deserialize_cdr(rawd, connd.msgtype), connd)
    return color_bgr, depth


# ─────────────────────────────────────────────────────────────
# Paper steps: crop + FPS
# ─────────────────────────────────────────────────────────────
def crop_workspace(pcd, ws):
    p = np.asarray(pcd.points)
    if len(p) == 0:
        return pcd
    m = ((p[:, 0] >= ws['x'][0]) & (p[:, 0] <= ws['x'][1]) &
         (p[:, 1] >= ws['y'][0]) & (p[:, 1] <= ws['y'][1]) &
         (p[:, 2] >= ws['z'][0]) & (p[:, 2] <= ws['z'][1]))
    return pcd.select_by_index(np.where(m)[0])


def fps_downsample(pcd, n=8192):
    if len(pcd.points) <= n:
        return pcd
    # Open3D >= 0.15 has native FPS.
    if hasattr(pcd, "farthest_point_down_sample"):
        return pcd.farthest_point_down_sample(n)
    # Fallback: uniform pick (rough, but avoids extra deps).
    idx = np.random.choice(len(pcd.points), n, replace=False)
    return pcd.select_by_index(idx)


def paint_source(pcd, color):
    """Overwrite colors so a whole cloud is a single solid color."""
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(pcd.points), 1))
    )
    return pcd


def setup_kinematics(urdf_path, end_link):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    with open(urdf_path, 'r') as f:
        return kp.build_serial_chain_from_urdf(
            f.read(), end_link_name=end_link, root_link_name="base_link")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', required=True)
    ap.add_argument('--frame_index', type=int, default=0)
    ap.add_argument('--num_points', type=int, default=8192)
    ap.add_argument('--urdf_path', type=str, default=DEFAULT_URDF)
    ap.add_argument('--end_link', type=str, default=DEFAULT_END_LINK,
                    help='FK tip for the wrist transform (matches dataset config).')
    ap.add_argument('--distinguish', action='store_true',
                    help='Tint front (blue) vs wrist (red) instead of true RGB.')
    ap.add_argument('--no_crop', action='store_true',
                    help='Skip workspace crop (debug: see full merged cloud).')
    ap.add_argument('--show_ee', action='store_true',
                    help='Draw a coordinate frame at the FK end-effector pose '
                         '(kinematic ground truth). The CORRECT cloud must have '
                         'the gripper/hand sitting at this frame.')
    ap.add_argument('--wrist_invert_extrinsic', action='store_true',
                    help='Use inv(WRIST_EXTRINSICS) — test if the calibration '
                         'extrinsic direction (camera<->gripper) is reversed.')
    ap.add_argument('--wrist_offset', type=float, nargs=3, default=None,
                    metavar=('DX', 'DY', 'DZ'),
                    help='Manual XYZ nudge (meters) applied to the wrist cloud '
                         'in base frame — use to measure/compensate the residual.')
    ap.add_argument('--save_split_ply', type=str, default=None,
                    metavar='PREFIX',
                    help='Save front/wrist clouds as separate PLYs '
                         '(PREFIX_front.ply, PREFIX_wrist.ply) for measuring.')
    ap.add_argument('--diagnose', action='store_true',
                    help='Test all 4 transform hypotheses (tool0/inspire x '
                         'ext/inv-ext) in one run; report wrist-vs-front table '
                         'gap for each. Correct one = gap near 0 AND constant '
                         'across frames.')
    ap.add_argument('--measure', action='store_true',
                    help='Auto-measure the wrist<->front residual offset via ICP '
                         'and table-plane height (objective, no clicking).')
    ap.add_argument('--save_ply', type=str, default=None)
    ap.add_argument('--no_show', action='store_true')
    args = ap.parse_args()

    print("[0/5] Setting up kinematics (FK for wrist camera) ...")
    chain = setup_kinematics(args.urdf_path, args.end_link)
    print(f"      wrist FK tip: {args.end_link}")

    print("[1/5] Reading front (ZED) frame ...")
    f_color, f_depth, f_ts, arm_state = read_front_frame(args.episode, args.frame_index)
    front = build_camera_cloud(f_color, f_depth, FRONT_INTRINSICS)
    front = transform_front_to_base(front, FRONT_EXTRINSICS)
    print(f"      front cloud: {len(front.points):,} pts")
    print(f"      arm joints:  {np.round(arm_state[:6], 4)}")

    if args.diagnose:
        run_diagnose(args, front, arm_state, f_ts)
        return

    T_base_flange = fk_matrix(chain, arm_state[:6])

    print("[2/5] Reading wrist (RealSense) frame + FK transform ...")
    w_color, w_depth = read_wrist_frame(args.episode, f_ts)
    wrist = None
    if w_color is not None and w_depth is not None:
        wrist = build_camera_cloud(w_color, w_depth, WRIST_INTRINSICS)
        wrist = transform_wrist_to_base(wrist, T_base_flange, WRIST_EXTRINSICS,
                                        invert_extrinsic=args.wrist_invert_extrinsic)
        if args.wrist_offset is not None:
            wrist.translate(np.array(args.wrist_offset, dtype=float))
            print(f"      [nudge] wrist translated by {args.wrist_offset}")
        print(f"      wrist cloud: {len(wrist.points):,} pts")

    if args.save_split_ply:
        o3d.io.write_point_cloud(f"{args.save_split_ply}_front.ply", front)
        if wrist is not None:
            o3d.io.write_point_cloud(f"{args.save_split_ply}_wrist.ply", wrist)
        print(f"      saved split PLYs: {args.save_split_ply}_front.ply / _wrist.ply")

    if args.measure and wrist is not None:
        print("\n===== OFFSET MEASUREMENT =====")
        # 1) ICP: translation that best snaps wrist onto front = the residual.
        T_icp, rmse, fitness = measure_offset_icp(wrist, front)
        dxyz = T_icp[:3, 3]
        print(f"  ICP residual translation (m): dx={dxyz[0]:+.4f} "
              f"dy={dxyz[1]:+.4f} dz={dxyz[2]:+.4f}")
        print(f"  ICP |offset| = {np.linalg.norm(dxyz)*100:.2f} cm | "
              f"rmse={rmse*1000:.1f}mm | fitness={fitness:.2f}")
        print(f"  -> to compensate, try: --wrist_offset "
              f"{dxyz[0]:.4f} {dxyz[1]:.4f} {dxyz[2]:.4f}")
        # 2) Table-plane height gap (pure vertical measure).
        wt = segment_plane_z(wrist)
        ft = segment_plane_z(front)
        if wt and ft:
            print(f"  table height  front={ft[0]:+.4f}m  wrist={wt[0]:+.4f}m  "
                  f"gap={(wt[0]-ft[0])*100:+.2f}cm")
        print("==============================\n")

    if args.distinguish:
        paint_source(front, FRONT_COLOR)
        if wrist is not None:
            paint_source(wrist, WRIST_COLOR)
        print("      [distinguish] front = BLUE, wrist = RED")

    print("[3/5] Merging ...")
    merged = front
    if wrist is not None:
        merged = front + wrist
    print(f"      merged raw:  {len(merged.points):,} pts")

    if args.no_crop:
        cropped = merged
        print("[4/5] Workspace crop SKIPPED (--no_crop)")
    else:
        cropped = crop_workspace(merged, WORKSPACE)
        print(f"[4/5] Workspace crop -> {len(cropped.points):,} pts")

    print(f"[5/5] FPS -> {args.num_points} ...")
    final = fps_downsample(cropped, args.num_points)
    print(f"      final:       {len(final.points):,} pts")

    # Report the source split after FPS (only meaningful with --distinguish).
    if args.distinguish and len(final.points) > 0:
        cols = np.asarray(final.colors)
        n_front = int(np.sum(np.all(np.isclose(cols, FRONT_COLOR, atol=0.05), axis=1)))
        n_wrist = int(np.sum(np.all(np.isclose(cols, WRIST_COLOR, atol=0.05), axis=1)))
        print(f"      split after FPS -> front(blue): {n_front:,} | wrist(red): {n_wrist:,}")

    if args.save_ply:
        o3d.io.write_point_cloud(args.save_ply, final)
        print(f"      saved: {args.save_ply}")

    if not args.no_show:
        geoms = [final]
        # base_link origin
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
        if args.show_ee:
            # FK end-effector pose = kinematic ground truth for the gripper.
            geoms.append(make_frame(T_base_flange, size=0.15))
            print(f"      [ground truth] EE ({args.end_link}) at "
                  f"{np.round(T_base_flange[:3, 3], 4)} — the correct cloud's "
                  f"gripper should sit here.")
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"Merged PC (frame {args.frame_index}) - {len(final.points)} pts",
        )


if __name__ == '__main__':
    main()