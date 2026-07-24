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
def _normalize_depth_m(depth):
    """Force depth to float32 meters (auto-detect mm)."""
    depth = depth.astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    valid = depth > 0
    if np.any(valid) and np.median(depth[valid]) > 20.0:
        depth = depth / 1000.0  # mm -> m
    return depth


def stage_export(args):
    """Export ONE frame (legacy single-frame dump)."""
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

    depth = _normalize_depth_m(depth)
    valid = depth > 0

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


def stage_export_episode(args):
    """Export MANY front-camera frames from one SVO into a browsable dump.

    Layout under export_dir:
        meta.json                 (episode-level: camera, frame list, urdf...)
        frames/frame_0000_rgb.png
        frames/frame_0000_depth.npy
        ...
    Only the FRONT (ZED) camera is supported here - it needs a single SVO
    pass. Wrist frames would require rosbag time-matching per frame; use the
    single-frame 'export' for that.
    """
    import cv2
    import pyzed.sl as sl
    from visualize_merged_pc import DEFAULT_URDF, DEFAULT_END_LINK, ARM_JOINT_NAMES

    frames_dir = os.path.join(args.export_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    svo = os.path.join(args.episode, "recording.svo2")
    if not os.path.exists(svo):
        raise SystemExit(f"No recording.svo2 in {args.episode}")

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
        raise SystemExit("Failed to open SVO")

    total = cam.get_svo_number_of_frames()
    rt = sl.RuntimeParameters()
    rt.confidence_threshold = 100
    rt.texture_confidence_threshold = 100
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    end = total if args.count <= 0 else min(args.start + args.count * args.stride, total)
    wanted = list(range(args.start, end, args.stride))
    wanted_set = set(wanted)
    print(f"[export-ep] SVO has {total} frames; exporting {len(wanted)} "
          f"(start={args.start}, stride={args.stride}) ...")

    last_positions = np.zeros(12, dtype=np.float32)
    last_ts = sl.Timestamp()
    last_ts.set_nanoseconds(0)

    saved = []
    idx = 0
    max_idx = max(wanted) if wanted else -1
    while idx <= max_idx:
        if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            break

        # Track latest joint states (same source extract_episodes uses).
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

        if idx in wanted_set:
            cam.retrieve_image(img_mat, sl.VIEW.LEFT)
            cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            color_bgr = cv2.cvtColor(img_mat.get_data(), cv2.COLOR_BGRA2BGR)
            depth = _normalize_depth_m(depth_mat.get_data().copy())

            base = f"frame_{idx:04d}"
            cv2.imwrite(os.path.join(frames_dir, base + "_rgb.png"), color_bgr)
            np.save(os.path.join(frames_dir, base + "_depth.npy"), depth)
            saved.append({
                "frame_index": idx,
                "rgb": f"frames/{base}_rgb.png",
                "depth": f"frames/{base}_depth.npy",
                "arm_joints": last_positions[:6].tolist(),
            })
            print(f"  saved frame {idx} ({len(saved)}/{len(wanted)})")
        idx += 1

    cam.close()

    meta = {
        "episode": args.episode,
        "camera": "front",
        "multi_frame": True,
        "urdf_path": args.urdf_path or DEFAULT_URDF,
        "end_link": args.end_link or DEFAULT_END_LINK,
        "depth_units": "meters",
        "frames": saved,
    }
    with open(os.path.join(args.export_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[export-ep] wrote {len(saved)} frames + meta.json to {args.export_dir}")


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


def _build_depth_panel(color_bgr, raw_depth, cdm_depth, vmin=0.1, vmax=5.0):
    """Build the 4-panel BGR image [RGB | raw | cdm | diff] + mean-diff (cm).

    Returns (panel_bgr, mean_diff_cm). Shared by the headless PNG writer and
    the interactive depth browser.
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

    for img, txt in ((rgb_display, "RGB"),
                     (raw_colored, "raw depth"),
                     (cdm_colored, "cdm depth"),
                     (diff_colored, "|cdm-raw| diff (0-20cm)")):
        cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3, cv2.LINE_AA)

    grid = image_grid([rgb_display, raw_colored, cdm_colored, diff_colored],
                      rows=1, cols=4)
    panel_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    mean_diff_cm = float(diff[m].mean()) * 100 if np.any(m) else 0.0
    return panel_bgr, mean_diff_cm


def save_depth_comparison(color_bgr, raw_depth, cdm_depth, out_path,
                          vmin=0.1, vmax=5.0):
    """Write a 4-panel image-space comparison [RGB | raw | cdm | diff]."""
    import cv2
    panel, mean_cm = _build_depth_panel(color_bgr, raw_depth, cdm_depth,
                                        vmin=vmin, vmax=vmax)
    cv2.imwrite(out_path, panel)
    print(f"[compare] depth panel saved: {out_path}")
    print(f"[compare] mean |cdm-raw| over valid px = {mean_cm:.2f} cm")


def _browse_depth_panels(args, frame_list, model):
    """Interactive matplotlib browser through the 4-panel depth comparison.

    Uses matplotlib (not cv2) because the container often ships headless
    OpenCV without GUI support. matplotlib uses the X display directly.

    Controls (figure window):
        n / Right / Space -> next    p / Left -> prev
        s -> save current panel PNG   q -> quit
    """
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt

    print(f"[browse-depth] {len(frame_list)} frames | "
          f"n/-> next | p/<- prev | s save | q quit")

    cache = {}

    def get_panel(pos):
        if pos not in cache:
            entry = frame_list[pos]
            color_bgr = cv2.imread(
                os.path.join(args.export_dir, entry["rgb"]), cv2.IMREAD_COLOR)
            depth = np.load(os.path.join(args.export_dir, entry["depth"]))
            cdm_depth = run_cdm_depth(model, color_bgr, depth,
                                      input_size=args.input_size)
            panel, mean_cm = _build_depth_panel(
                color_bgr, depth, cdm_depth,
                vmin=args.depth_vmin, vmax=args.depth_vmax)
            cache[pos] = (panel, mean_cm)
        return cache[pos]

    state = {"cur": 0}

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    img_handle = None

    def show():
        nonlocal img_handle
        panel, mean_cm = get_panel(state["cur"])
        frame_idx = frame_list[state["cur"]]["frame_index"]
        # panel is BGR -> RGB for matplotlib
        rgb = panel[:, :, ::-1]
        if img_handle is None:
            img_handle = ax.imshow(rgb)
        else:
            img_handle.set_data(rgb)
        ax.set_title(
            f"frame {frame_idx} ({state['cur']+1}/{len(frame_list)}) | "
            f"mean|cdm-raw|={mean_cm:.2f}cm   "
            f"[RGB | raw | cdm | diff]", fontsize=11)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ('n', 'right', ' '):
            state["cur"] = min(state["cur"] + 1, len(frame_list) - 1)
            show()
        elif event.key in ('p', 'left'):
            state["cur"] = max(state["cur"] - 1, 0)
            show()
        elif event.key == 's':
            panel, _ = get_panel(state["cur"])
            frame_idx = frame_list[state["cur"]]["frame_index"]
            out = os.path.join(args.export_dir,
                               f"depth_compare_{frame_idx:04d}.png")
            cv2.imwrite(out, panel)
            print(f"[browse-depth] saved {out}")
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    show()
    plt.show()


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


def stage_browse(args, meta):
    """Interactive frame-by-frame RAW vs CDM browser for a multi-frame dump.

    Loads the CDM once, then lets you step through the exported front-camera
    frames with the keyboard, building RAW + CDM clouds on demand (cached).

    Controls:
        N / Right -> next    P / Left -> prev
        O -> overlay/side toggle    T -> tint/RGB toggle    Q -> quit
    """
    import cv2
    import open3d as o3d
    from visualize_merged_pc import (
        FRONT_INTRINSICS, FRONT_EXTRINSICS,
        build_camera_cloud, transform_front_to_base,
        crop_workspace, fps_downsample, paint_source, WORKSPACE,
    )

    frame_list = meta["frames"]
    if not frame_list:
        raise SystemExit("multi_frame dump has no frames.")
    print(f"[browse] {len(frame_list)} frames in {args.export_dir}")

    # Load CDM once.
    print(f"[browse] Running CDM ({args.cdm_model}) ...")
    ckpt = resolve_cdm_checkpoint(args.cdm_model)
    model = load_cdm(ckpt, args.encoder)

    # --- 2D depth-panel browser (cv2) instead of 3D point clouds ---
    if getattr(args, "depth_mode", False):
        return _browse_depth_panels(args, frame_list, model)

    def build_clouds(color_bgr, depth):
        raw = build_camera_cloud(color_bgr, depth, FRONT_INTRINSICS)
        raw = transform_front_to_base(raw, FRONT_EXTRINSICS)
        cdm_depth = run_cdm_depth(model, color_bgr, depth,
                                  input_size=args.input_size)
        cdm = build_camera_cloud(color_bgr, cdm_depth, FRONT_INTRINSICS)
        cdm = transform_front_to_base(cdm, FRONT_EXTRINSICS)
        if not args.no_crop:
            raw = crop_workspace(raw, WORKSPACE)
            cdm = crop_workspace(cdm, WORKSPACE)
        raw = fps_downsample(raw, args.num_points)
        cdm = fps_downsample(cdm, args.num_points)
        return raw, cdm

    state = {"cur": 0, "overlay": args.overlay, "tint": args.tint, "cache": {}}

    def get_clouds(pos):
        if pos not in state["cache"]:
            entry = frame_list[pos]
            color_bgr = cv2.imread(
                os.path.join(args.export_dir, entry["rgb"]), cv2.IMREAD_COLOR)
            depth = np.load(os.path.join(args.export_dir, entry["depth"]))
            state["cache"][pos] = build_clouds(color_bgr, depth)
        return state["cache"][pos]

    def refresh(vis):
        raw, cdm = get_clouds(state["cur"])
        frame_idx = frame_list[state["cur"]]["frame_index"]
        raw_c = o3d.geometry.PointCloud(raw)
        cdm_c = o3d.geometry.PointCloud(cdm)
        if state["tint"]:
            paint_source(raw_c, RAW_COLOR)
            paint_source(cdm_c, CDM_COLOR)
        if not state["overlay"]:
            cdm_c.translate((0.0, args.gap, 0.0))
        vis.clear_geometries()
        vis.add_geometry(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3),
            reset_bounding_box=False)
        vis.add_geometry(raw_c, reset_bounding_box=False)
        vis.add_geometry(cdm_c, reset_bounding_box=False)
        mode = "OVERLAY" if state["overlay"] else f"SIDE-BY-SIDE(+{args.gap}m Y)"
        rgb = "tint" if state["tint"] else "true RGB"
        print(f"[browse] frame {frame_idx} ({state['cur']+1}/{len(frame_list)}) "
              f"| {mode} | {rgb} | raw={len(raw_c.points):,} "
              f"cdm={len(cdm_c.points):,}")

    def nxt(vis):
        if state["cur"] < len(frame_list) - 1:
            state["cur"] += 1
            refresh(vis)
        return False

    def prv(vis):
        if state["cur"] > 0:
            state["cur"] -= 1
            refresh(vis)
        return False

    def tog_overlay(vis):
        state["overlay"] = not state["overlay"]
        refresh(vis)
        return False

    def tog_tint(vis):
        state["tint"] = not state["tint"]
        refresh(vis)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="RAW(blue) vs CDM(red) episode browser")
    vis.register_key_callback(ord("N"), nxt)
    vis.register_key_callback(ord("P"), prv)
    vis.register_key_callback(262, nxt)   # Right arrow
    vis.register_key_callback(263, prv)   # Left arrow
    vis.register_key_callback(ord("O"), tog_overlay)
    vis.register_key_callback(ord("T"), tog_tint)

    raw, cdm = get_clouds(0)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
    vis.add_geometry(o3d.geometry.PointCloud(raw))
    vis.add_geometry(o3d.geometry.PointCloud(cdm))
    refresh(vis)
    print("\n[browse] N/-> next | P/<- prev | O overlay | T tint | Q quit\n")
    vis.run()
    vis.destroy_window()


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

    # Multi-frame dump -> interactive browser.
    if meta.get("multi_frame"):
        return stage_browse(args, meta)

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

    # export (single frame)
    e = sub.add_parser("export", help="Read ONE SVO/rosbag frame -> dump files.")
    e.add_argument('--episode', required=True)
    e.add_argument('--frame_index', type=int, default=0)
    e.add_argument('--camera', choices=['front', 'wrist'], default='front')
    e.add_argument('--export_dir', required=True)
    e.add_argument('--urdf_path', default=None)
    e.add_argument('--end_link', default=None)

    # export-episode (many front frames for browsing)
    ee = sub.add_parser("export-episode",
                        help="Dump MANY front-camera frames for browsing.")
    ee.add_argument('--episode', required=True)
    ee.add_argument('--export_dir', required=True)
    ee.add_argument('--start', type=int, default=0,
                    help='First SVO frame index.')
    ee.add_argument('--count', type=int, default=40,
                    help='How many frames (<=0 = all until end of SVO).')
    ee.add_argument('--stride', type=int, default=5,
                    help='Grab every Nth SVO frame (60fps source).')
    ee.add_argument('--urdf_path', default=None)
    ee.add_argument('--end_link', default=None)

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
    c.add_argument('--depth_mode', action='store_true',
                   help='(multi_frame) browse 2D depth panels '
                        '[RGB|raw|cdm|diff] with cv2 instead of 3D clouds.')
    return p


def main():
    args = build_parser().parse_args()
    if args.stage == "export":
        stage_export(args)
    elif args.stage == "export-episode":
        stage_export_episode(args)
    elif args.stage == "compare":
        stage_compare(args)


if __name__ == '__main__':
    main()