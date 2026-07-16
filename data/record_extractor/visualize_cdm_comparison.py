#!/usr/bin/env python3
"""
Compare the RAW point cloud vs the CDM (Camera Depth Model) cleaned point cloud
on YOUR OWN RGBD data (SVO front / RealSense wrist), for a single frame.

WHY TWO STAGES
--------------
The SVO reader (pyzed) lives ONLY in the zed_container, while CDM + torch +
checkpoints live ONLY in the ur_artus container. The two containers share just
the `data` folder. So we split the work and exchange data via files:

  Stage 1  export  (run in zed_container):
      read SVO/rosbag frame -> write rgb.png + depth.npy + meta.json to a dir
      under data/.  (needs pyzed / rosbags; NO torch)

  Stage 2  compare (run in ur_artus):
      read that dump -> run CDM -> build raw + cdm clouds -> visualize/save.
      (needs torch / CDM / huggingface_hub; NO pyzed)

EXAMPLES
--------
# In zed_container: dump frame 100 (front camera) into the shared data folder.
python3 data/record_extractor/visualize_cdm_comparison.py export \
    --episode /root/ros2_ws/data/episodes/foil_lifting_collection/episode_002_success \
    --frame_index 100 --camera front \
    --export_dir /root/ros2_ws/data/cdm_dumps/ep002_f100_front

# In ur_artus: run CDM on the dump and compare side-by-side.
python3 data/record_extractor/visualize_cdm_comparison.py compare \
    --export_dir /ros2_ws/data/cdm_dumps/ep002_f100_front \
    --cdm_model depth-anything/camera-depth-model-zed2i-neural

Note: the two containers mount data at different paths
(/root/ros2_ws/data vs /ros2_ws/data) - point --export_dir at the same physical
folder in each container's own path.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CDM lives under third_party (only importable/usable in ur_artus).
CDM_DIR = os.environ.get(
    "CDM_DIR", "/ros2_ws/third_party/manip-as-in-sim-suite/cdm")

RAW_COLOR = np.array([0.1, 0.4, 1.0])   # blue = raw sensor cloud
CDM_COLOR = np.array([1.0, 0.2, 0.1])   # red  = CDM cleaned cloud


# ────────────────────────────────────────────────────
# STAGE 1: export  (zed_container)  — reads SVO/rosbag, writes files
# ────────────────────────────────────────────────────
def stage_export(args):
    import cv2
    # pyzed / rosbags are pulled in lazily by these readers.
    from visualize_merged_pc import (
        read_front_frame, read_wrist_frame, DEFAULT_URDF, DEFAULT_END_LINK,
    )

    os.makedirs(args.export_dir, exist_ok=True)

    print("[export] Reading front (ZED) frame + arm joints ...")
    f_color, f_depth, f_ts, arm_state = read_front_frame(
        args.episode, args.frame_index)
    print(f"[export] arm joints: {np.round(arm_state[:6], 4)}")

    if args.camera == 'front':
        color_bgr, depth = f_color, f_depth
    else:
        print("[export] Reading wrist (RealSense) frame ...")
        color_bgr, depth = read_wrist_frame(args.episode, f_ts)
        if color_bgr is None or depth is None:
            raise SystemExit("No wrist frame available for this episode.")

    # Normalize depth to float32 meters so the compare stage needs no guessing.
    depth = depth.astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    valid = depth > 0
    if np.any(valid) and np.median(depth[valid]) > 20.0:
        depth = depth / 1000.0  # mm -> m

    # Save RGB as BGR PNG (cv2 round-trips BGR), depth as float32 meters npy.
    cv2.imwrite(os.path.join(args.export_dir, "rgb.png"), color_bgr)
    np.save(os.path.join(args.export_dir, "depth.npy"), depth)
    meta = {
        "episode": args.episode,
        "frame_index": args.frame_index,
        "camera": args.camera,
        "arm_joints": arm_state[:6].tolist(),
        "urdf_path": args.urdf_path or DEFAULT_URDF,
        "end_link": args.end_link or DEFAULT_END_LINK,
        "depth_units": "meters",
    }
    with open(os.path.join(args.export_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[export] wrote rgb.png / depth.npy / meta.json to {args.export_dir}")
    print(f"[export] depth: min={depth[valid].min():.3f} "
          f"max={depth[valid].max():.3f} m (valid px: {int(valid.sum()):,})")


# ────────────────────────────────────────────────────
# CDM model loading + inference  (compare stage only)
# ────────────────────────────────────────────────────
def resolve_cdm_checkpoint(spec):
    """Local .ckpt path OR HuggingFace repo id -> local checkpoint file."""
    if os.path.isfile(spec):
        return spec
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub required to resolve repo ids; "
            "pip install huggingface_hub or pass a local .ckpt path.")
    files = list_repo_files(spec)
    ckpts = [f for f in files if f.endswith((".ckpt", ".pth", ".safetensors"))]
    if not ckpts:
        raise SystemExit(f"No checkpoint in repo '{spec}'. Files: {files}")
    print(f"[cdm] resolving {spec} -> {ckpts[0]}")
    return hf_hub_download(repo_id=spec, filename=ckpts[0])


def load_cdm(checkpoint, encoder="vitl"):
    """Load a CDM model using cdm/infer.py's loader."""
    sys.path.insert(0, CDM_DIR)
    from infer import load_model
    return load_model(encoder, checkpoint)


def run_cdm_depth(model, color_bgr, depth_m, input_size=518, max_depth=25.0):
    """Run CDM on one RGBD frame; returns (H,W) float32 predicted metric depth."""
    depth = depth_m.astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    valid = depth > 0
    if np.any(valid) and np.median(depth[valid]) > 20.0:
        depth = depth / 1000.0
    depth[depth > max_depth] = 0.0

    rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])  # BGR -> RGB
    simi = np.zeros_like(depth)
    simi[depth > 0] = 1.0 / depth[depth > 0]  # inverse depth input

    pred_inv = model.infer_image(rgb, simi, input_size=input_size)
    pred = np.zeros_like(pred_inv, dtype=np.float32)
    nz = pred_inv > 1e-6
    pred[nz] = 1.0 / pred_inv[nz]
    return pred


def save_depth_comparison(color_bgr, raw_depth, cdm_depth, out_path,
                          vmin=0.1, vmax=5.0):
    """Write a 4-panel image-space comparison:

        [ RGB | raw depth | CDM depth | |CDM - raw| diff ]

    This surfaces exactly what CDM changes (filled holes, sharpened edges,
    removed flying pixels) that the sparse 8192-pt cloud averages away.
    Runs fully headless (no Open3D window) - ideal for the ur_artus container.
    """
    import cv2
    sys.path.insert(0, CDM_DIR)
    from infer import colorize_depth_maps, image_grid

    def color(depth, dmin, dmax):
        c = colorize_depth_maps(depth, min_depth=dmin, max_depth=dmax,
                                cmap="Spectral")
        c = np.rollaxis(c[0], 0, 3)  # [C,H,W] -> [H,W,C]
        return (c * 255).astype(np.uint8)

    rgb_display = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    raw_colored = color(raw_depth, vmin, vmax)
    cdm_colored = color(cdm_depth, vmin, vmax)

    # Absolute depth difference (meters), only where the raw sensor had data.
    diff = np.zeros_like(raw_depth)
    m = raw_depth > 0
    diff[m] = np.abs(cdm_depth[m] - raw_depth[m])
    diff_colored = color(diff, 0.0, 0.20)  # 0-20 cm scale

    # Label every tile so the panel is self-explanatory.
    #   RGB          : the input color image
    #   raw depth    : the sensor's depth
    #   cdm depth    : CDM-predicted (cleaned) depth
    #   |cdm-raw|    : how much CDM changed the depth per pixel (0-20cm)
    for img, txt in ((rgb_display, "RGB"),
                     (raw_colored, "raw depth"),
                     (cdm_colored, "cdm depth"),
                     (diff_colored, "|cdm-raw| diff (0-20cm)")):
        cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3, cv2.LINE_AA)

    grid = image_grid([rgb_display, raw_colored, cdm_colored, diff_colored],
                      rows=1, cols=4)
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    valid_mean = float(diff[m].mean()) if np.any(m) else 0.0
    print(f"[compare] depth panel saved: {out_path}")
    print(f"[compare] mean |cdm-raw| over valid px = {valid_mean*100:.2f} cm")


# ────────────────────────────────────────────────────
# Cloud building per camera (raw + cdm share this)  (compare stage)
# ────────────────────────────────────────────────────
def build_cloud_in_base(color_bgr, depth, camera, T_base_flange):
    from visualize_merged_pc import (
        FRONT_INTRINSICS, FRONT_EXTRINSICS, WRIST_INTRINSICS, WRIST_EXTRINSICS,
        build_camera_cloud, transform_front_to_base, transform_wrist_to_base,
    )
    if camera == "front":
        pcd = build_camera_cloud(color_bgr, depth, FRONT_INTRINSICS)
        return transform_front_to_base(pcd, FRONT_EXTRINSICS)
    else:
        pcd = build_camera_cloud(color_bgr, depth, WRIST_INTRINSICS)
        return transform_wrist_to_base(pcd, T_base_flange, WRIST_EXTRINSICS)


def stage_compare(args):
    import cv2
    import open3d as o3d
    from visualize_merged_pc import (
        WORKSPACE, fk_matrix, setup_kinematics,
        crop_workspace, fps_downsample, paint_source,
        DEFAULT_URDF, DEFAULT_END_LINK,
    )

    # ── Load the exported dump ──
    meta_path = os.path.join(args.export_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"No meta.json in {args.export_dir}; run 'export' first.")
    with open(meta_path) as f:
        meta = json.load(f)
    camera = meta["camera"]
    arm_state = np.array(meta["arm_joints"], dtype=np.float32)
    color_bgr = cv2.imread(os.path.join(args.export_dir, "rgb.png"),
                           cv2.IMREAD_COLOR)
    depth = np.load(os.path.join(args.export_dir, "depth.npy"))
    print(f"[compare] loaded {camera} frame {meta['frame_index']} from "
          f"{args.export_dir}")

    T_base_flange = None
    if camera == "wrist":
        # meta.json may hold the OTHER container's URDF path; fall back to the
        # file-relative DEFAULT_URDF (or --urdf_path) if it isn't found here.
        urdf = args.urdf_path or meta.get("urdf_path")
        if not urdf or not os.path.exists(urdf):
            urdf = DEFAULT_URDF
        end_link = args.end_link or meta.get("end_link") or DEFAULT_END_LINK
        print(f"[compare] wrist FK: urdf={urdf} end_link={end_link}")
        chain = setup_kinematics(urdf, end_link)
        T_base_flange = fk_matrix(chain, arm_state[:6])

    # ── Raw baseline cloud ──
    print(f"[compare] Building RAW cloud ({camera}) ...")
    raw = build_cloud_in_base(color_bgr, depth, camera, T_base_flange)
    print(f"          raw cloud: {len(raw.points):,} pts")

    # ── CDM cleaned cloud ──
    print(f"[compare] Running CDM ({args.cdm_model}) ...")
    ckpt = resolve_cdm_checkpoint(args.cdm_model)
    model = load_cdm(ckpt, args.encoder)
    cdm_depth = run_cdm_depth(model, color_bgr, depth, input_size=args.input_size)
    print(f"          cdm depth: min={cdm_depth.min():.3f} "
          f"max={cdm_depth.max():.3f} mean={cdm_depth.mean():.3f} m")

    # 2D image-space comparison (headless) - shows what the sparse cloud hides.
    if args.save_depth_png:
        save_depth_comparison(color_bgr, depth, cdm_depth, args.save_depth_png,
                              vmin=args.depth_vmin, vmax=args.depth_vmax)

    cdm = build_cloud_in_base(color_bgr, cdm_depth, camera, T_base_flange)
    print(f"          cdm cloud: {len(cdm.points):,} pts")

    # ── Crop + FPS both (same pipeline as the dataset) ──
    if not args.no_crop:
        raw = crop_workspace(raw, WORKSPACE)
        cdm = crop_workspace(cdm, WORKSPACE)
        print(f"[compare] after crop -> raw: {len(raw.points):,} | "
              f"cdm: {len(cdm.points):,}")
    raw = fps_downsample(raw, args.num_points)
    cdm = fps_downsample(cdm, args.num_points)
    print(f"[compare] after FPS  -> raw: {len(raw.points):,} | "
          f"cdm: {len(cdm.points):,}")

    if args.save_ply_prefix:
        o3d.io.write_point_cloud(f"{args.save_ply_prefix}_raw.ply", raw)
        o3d.io.write_point_cloud(f"{args.save_ply_prefix}_cdm.ply", cdm)
        print(f"[compare] saved {args.save_ply_prefix}_raw.ply / _cdm.ply")

    if args.no_show:
        return

    def marker(center, color, size=0.05):
        """Small solid sphere used as a color legend anchor in the 3D view."""
        s = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        s.translate(center)
        s.paint_uniform_color(color)
        return s

    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]

    if args.tint:
        # Optional: recolor whole clouds raw=blue / cdm=red.
        paint_source(raw, RAW_COLOR)
        paint_source(cdm, CDM_COLOR)
        tint = True
    else:
        # Default: keep the real sensor RGB; marker spheres label the source.
        tint = False

    if args.overlay:
        geoms += [raw, cdm]
        # Legend markers offset up in Z so they float above the clouds.
        geoms += [marker((0, 0, 0.4), RAW_COLOR), marker((0.1, 0, 0.4), CDM_COLOR)]
        print("[compare] OVERLAY (same location):")
    else:
        cdm.translate((0.0, args.gap, 0.0))
        geoms += [raw, cdm]
        geoms += [marker((0, 0, 0.4), RAW_COLOR),
                  marker((0, args.gap, 0.4), CDM_COLOR)]
        print(f"[compare] SIDE-BY-SIDE (cdm shifted +{args.gap} m along Y):")

    print("           RAW  cloud  = BLUE  marker" +
          (" + blue points" if tint else " (true RGB, at origin / left)"))
    print("           CDM  cloud  = RED   marker" +
          (" + red points" if tint else f" (true RGB, +{args.gap} m Y / right)"))

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"RAW(blue) vs CDM(red) - {camera} frame {meta['frame_index']}",
    )


# ────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    # export
    e = sub.add_parser("export", help="Read SVO/rosbag frame -> dump files.")
    e.add_argument('--episode', required=True)
    e.add_argument('--frame_index', type=int, default=0)
    e.add_argument('--camera', choices=['front', 'wrist'], default='front')
    e.add_argument('--export_dir', required=True)
    e.add_argument('--urdf_path', default=None)
    e.add_argument('--end_link', default=None)

    # compare
    c = sub.add_parser("compare", help="Load dump -> run CDM -> compare.")
    c.add_argument('--export_dir', required=True)
    c.add_argument('--cdm_model', required=True,
                   help='CDM checkpoint: local .ckpt path or HF repo id.')
    c.add_argument('--encoder', default='vitl',
                   choices=['vits', 'vitb', 'vitl', 'vitg'])
    c.add_argument('--input_size', type=int, default=518)
    c.add_argument('--num_points', type=int, default=8192)
    c.add_argument('--urdf_path', default=None,
                   help='Override meta.json urdf (wrist camera only).')
    c.add_argument('--end_link', default=None,
                   help='Override meta.json end_link (wrist camera only).')
    c.add_argument('--no_crop', action='store_true')
    c.add_argument('--overlay', action='store_true',
                   help='Overlay raw(blue)+cdm(red) instead of side-by-side.')
    c.add_argument('--tint', action='store_true',
                   help='Recolor whole clouds raw=blue/cdm=red. Default keeps '
                        'true sensor RGB (marker spheres still label each).')
    c.add_argument('--gap', type=float, default=1.0,
                   help='Side-by-side spacing (m) along +Y for the CDM cloud.')
    c.add_argument('--save_ply_prefix', default=None)
    c.add_argument('--save_depth_png', default=None,
                   help='Write a headless 4-panel depth comparison image '
                        '[RGB | raw | cdm | diff] to this path.')
    c.add_argument('--depth_vmin', type=float, default=0.1,
                   help='Min depth (m) for the depth-png color scale.')
    c.add_argument('--depth_vmax', type=float, default=5.0,
                   help='Max depth (m) for the depth-png color scale.')
    c.add_argument('--no_show', action='store_true')
    return p


def main():
    args = build_parser().parse_args()
    if args.stage == "export":
        stage_export(args)
    elif args.stage == "compare":
        stage_compare(args)


if __name__ == '__main__':
    main()