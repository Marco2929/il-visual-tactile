#!/usr/bin/env python3
"""
Minimal ZED 2i XYZ point-map reader.

Uses the same camera settings as record.py so the geometry matches recorded
episodes:
    HD720 @ 60fps, DEPTH_MODE.NEURAL, depth range 0.2 - 2.0 m

Difference to record.py: coordinate_units is set EXPLICITLY to METER here.
record.py leaves it at the SDK default (MILLIMETER) and passes 200/2000, which
is the same 0.2-2.0 m range expressed in mm.

Usage
-----
    from zed_xyz_stream import ZedXYZStream

    with ZedXYZStream() as zed:
        xyz = zed.get_xyz()          # torch.Tensor [3, H, W], meters
        pts = zed.get_points()       # np.ndarray  [N, 3], finite points only

Standalone smoke test:
    python zed_xyz_stream.py
"""

import numpy as np
import pyzed.sl as sl

try:
    import torch
except ImportError:
    torch = None


# Same as record.py / extract_episodes.py
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 2.0


class ZedXYZStream:
    def __init__(self,
                 resolution=sl.RESOLUTION.HD720,
                 fps=60,
                 depth_mode=sl.DEPTH_MODE.NEURAL,
                 depth_min_m=DEPTH_MIN_M,
                 depth_max_m=DEPTH_MAX_M,
                 svo_path=None):
        init = sl.InitParameters()
        init.camera_resolution = resolution
        init.camera_fps = fps
        init.depth_mode = depth_mode
        # Explicit: MEASURE.XYZ then returns meters.
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = depth_min_m
        init.depth_maximum_distance = depth_max_m
        init.sdk_verbose = 0

        if svo_path is not None:
            init.set_from_svo_file(svo_path)
            init.svo_real_time_mode = False

        self.cam = sl.Camera()
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        # Matches record.py's grab loop.
        self.runtime = sl.RuntimeParameters(
            confidence_threshold=100,
            texture_confidence_threshold=100,
            remove_saturated_areas=True,
        )
        self.xyz_mat = sl.Mat()

    # --- context manager ---
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        if self.cam is not None:
            self.cam.close()
            self.cam = None

    # --- intrinsics (same fields record.py writes to camera_intrinsics.json) ---
    def intrinsics(self):
        calib = (self.cam.get_camera_information()
                 .camera_configuration.calibration_parameters)
        left = calib.left_cam
        return {
            "width": left.image_size.width,
            "height": left.image_size.height,
            "fx": left.fx, "fy": left.fy,
            "cx": left.cx, "cy": left.cy,
        }

    # --- core ---
    def _grab_xyz_np(self):
        """Grab one frame; return the [H, W, 3] XYZ map in meters, or None."""
        if self.cam.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self.cam.retrieve_measure(self.xyz_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
        # get_data() is [H, W, 4]; the 4th channel is unused for MEASURE.XYZ.
        # Copy, because the Mat buffer is reused on the next grab.
        return self.xyz_mat.get_data()[:, :, :3].copy()

    def get_xyz(self, device="cuda"):
        """XYZ map as torch.Tensor [3, H, W] in meters. NaN/Inf -> 0.0."""
        if torch is None:
            raise RuntimeError("torch not installed; use get_points() instead.")
        xyz_np = self._grab_xyz_np()
        if xyz_np is None:
            return None
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        t = torch.from_numpy(xyz_np).permute(2, 0, 1).to(device)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def get_points(self, workspace_bounds=None):
        """
        Flat point list [N, 3] in meters, camera frame.

        Invalid pixels (NaN/Inf from occlusions/reflections) are DROPPED here
        rather than zeroed, so no fake points at the origin are produced.

        workspace_bounds: optional (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        xyz_np = self._grab_xyz_np()
        if xyz_np is None:
            return None

        pts = xyz_np.reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]

        if workspace_bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = workspace_bounds
            m = ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
                 (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
                 (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax))
            pts = pts[m]
        return pts


def main():
    with ZedXYZStream() as zed:
        print("intrinsics:", zed.intrinsics())

        pts = zed.get_points()
        if pts is None:
            print("grab failed")
            return
        print(f"valid points: {len(pts)}")
        if len(pts):
            print("  x range: %.3f .. %.3f m" % (pts[:, 0].min(), pts[:, 0].max()))
            print("  y range: %.3f .. %.3f m" % (pts[:, 1].min(), pts[:, 1].max()))
            print("  z range: %.3f .. %.3f m" % (pts[:, 2].min(), pts[:, 2].max()))

        if torch is not None:
            t = zed.get_xyz(device="cpu")
            print("tensor:", tuple(t.shape), t.dtype)


if __name__ == "__main__":
    main()