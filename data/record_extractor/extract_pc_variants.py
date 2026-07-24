#!/usr/bin/env python3
"""
Extract FRONT point clouds from one episode's SVO with specific ZED depth
parameters (confidence + texture_confidence). Writes to a config-named
folder inside the episode so different settings can be browsed side by side.

Output:
    <episode>/pc_variants/c<CONF>_t<TEX>/frame_0000.npy ...

Example:
    python3 data/record_extractor/extract_pc_variants.py \
        --episode /root/ros2_ws/data/episodes/external/test_collection_small/episode_000_success \
        --conf 90 --tex 80
"""

import os
import sys
import argparse
import numpy as np
import cv2
import pyzed.sl as sl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/ros2_ws/data/record_utils")

from point_cloud_utils import generate_point_cloud_from_depth
from camera_info import (
    camera_intrinsics as FRONT_INTRINSICS,
    camera_extrinsics as FRONT_EXTRINSICS,
    workspace as WORKSPACE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', required=True)
    ap.add_argument('--conf', type=int, required=True,
                    help='confidence_threshold (0-100)')
    ap.add_argument('--tex', type=int, required=True,
                    help='texture_confidence_threshold (0-100)')
    ap.add_argument('--target_fps', type=int, default=20)
    ap.add_argument('--num_points', type=int, default=4096)
    args = ap.parse_args()

    svo = os.path.join(args.episode, "recording.svo2")
    if not os.path.exists(svo):
        raise SystemExit(f"No recording.svo2 in {args.episode}")

    out_dir = os.path.join(args.episode, "pc_variants",
                           f"c{args.conf}_t{args.tex}")
    os.makedirs(out_dir, exist_ok=True)

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

    num_frames = cam.get_svo_number_of_frames()
    skip = 60 // args.target_fps

    rt = sl.RuntimeParameters()
    rt.confidence_threshold = args.conf
    rt.texture_confidence_threshold = args.tex
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    print(f"[c{args.conf}_t{args.tex}] {num_frames} frames @60fps "
          f"-> every {skip}th ...")

    out_idx = 0
    for fi in range(num_frames):
        err = cam.grab(rt)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            continue
        if fi % skip != 0:
            continue

        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        color_bgr = cv2.cvtColor(img_mat.get_data(), cv2.COLOR_BGRA2BGR)
        depth = depth_mat.get_data().copy()

        pc = generate_point_cloud_from_depth(
            color_bgr, depth,
            intrinsics=FRONT_INTRINSICS,
            extrinsic_matrix=FRONT_EXTRINSICS,
            workspace_bounds=WORKSPACE,
            include_color=True,
            num_points=args.num_points,
        )
        np.save(os.path.join(out_dir, f"frame_{out_idx:04d}.npy"), pc)
        out_idx += 1

    cam.close()
    print(f"[c{args.conf}_t{args.tex}] wrote {out_idx} clouds to {out_dir}")


if __name__ == '__main__':
    main()