#!/usr/bin/env python3
"""Throwaway: estimate the workspace box visible to the top-mounted ZED.

Projects the image corners through the new extrinsics onto z=0 (table) and
z=0.4 (object height) to see which base-frame box the camera actually covers.
"""
import json

import numpy as np

T = np.array([
    [-0.99991326, 0.01290498, 0.00263139, -0.39029286],
    [0.01167947, 0.96117350, -0.27569747, -0.53888558],
    [-0.00608709, -0.27564283, -0.96124085, 1.28573327],
    [0.0, 0.0, 0.0, 1.0],
])

fx = fy = 519.8837280273438
cx, cy = 647.674072265625, 363.5533447265625
W, H = 1280, 720

cam_origin = T[:3, 3]
print(f"camera origin in base : {np.round(cam_origin, 4)}")

# Optical axis (camera +z) in base frame.
axis = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
print(f"optical axis in base  : {np.round(axis, 4)}")

# Image corners + centre, as rays in the camera frame.
pix = [(0, 0), (W, 0), (0, H), (W, H), (cx, cy)]
names = ["top-left", "top-right", "bottom-left", "bottom-right", "centre"]

for z_plane in (0.0, 0.40):
    print(f"\n--- intersection with z = {z_plane} m ---")
    pts = []
    for (u, v), nm in zip(pix, names):
        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        d_base = T[:3, :3] @ d_cam
        if abs(d_base[2]) < 1e-9:
            continue
        # cam_origin + t*d_base has z == z_plane
        t = (z_plane - cam_origin[2]) / d_base[2]
        if t <= 0:
            continue
        p = cam_origin + t * d_base
        pts.append(p)
        print(f"  {nm:<13} {np.round(p, 4)}")
    if pts:
        P = np.array(pts)
        print(f"  x range: {P[:,0].min():+.3f} .. {P[:,0].max():+.3f}")
        print(f"  y range: {P[:,1].min():+.3f} .. {P[:,1].max():+.3f}")