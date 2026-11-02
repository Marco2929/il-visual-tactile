# dataset_builder/processing/progress/__init__.py
"""Progress module for dataset builder.

Computes continuous, physically-grounded progress labels for the foil
pinch-and-peel task.
"""

from typing import Optional

import numpy as np

from .curve import CurveParams, CONSERVATIVE, LEGACY80, get_curve_params, build_progress_curve
from .detector import (
    DEFAULT_CLOSED_FRAC,
    DEFAULT_MIN_CLOSED_FRAMES,
    RAW_GRASP_DIMS,
    HAND_PINCH_DIMS,
    smooth,
    runs,
    extract_grasp_signal,
    segment_grasp_attempts,
)


def compute_progress(
    num_frames: Optional[int] = None,
    ee_pose: Optional[np.ndarray] = None,
    finger_sensor: Optional[np.ndarray] = None,
    progress_dict: Optional[dict] = None,
    *,
    hand_state: Optional[np.ndarray] = None,
    joint_states: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute progress signal for one episode in the dataset builder pipeline.

    Supports both legacy arguments (num_frames, ee_pose, finger_sensor)
    and full proprioceptive arrays (joint_states, hand_state).
    """
    if joint_states is not None:
        N = len(joint_states)
    elif hand_state is not None:
        N = len(hand_state)
    elif finger_sensor is not None:
        N = len(finger_sensor)
    elif num_frames is not None:
        N = int(num_frames)
    else:
        raise ValueError("compute_progress requires length or state array.")

    if N <= 0:
        return np.zeros(0, dtype=np.float32)

    try:
        sig = extract_grasp_signal(
            joint_states=joint_states,
            hand_state=hand_state,
            finger_sensor=finger_sensor,
        )
    except Exception:
        return np.linspace(0.0, 1.0, N, dtype=np.float32)

    cfg = progress_dict or {}
    closed_frac = float(cfg.get("closed_frac", DEFAULT_CLOSED_FRAC))
    min_closed_frames = int(cfg.get("min_closed_frames", DEFAULT_MIN_CLOSED_FRAMES))
    smooth_w = int(cfg.get("smooth_w", 5))

    seg = segment_grasp_attempts(
        sig,
        smooth_w=smooth_w,
        closed_frac=closed_frac,
        min_closed_frames=min_closed_frames,
    )
    if not seg["has_grasp"]:
        return np.linspace(0.0, 1.0, N, dtype=np.float32)

    # End-effector z proxy if available
    z_arr = None
    if z is not None and len(z) == N:
        z_arr = z
    elif ee_pose is not None and ee_pose.ndim == 2 and ee_pose.shape[1] >= 3:
        z_arr = ee_pose[:, 2]
    elif joint_states is not None and joint_states.ndim == 2 and joint_states.shape[1] >= 3:
        q = np.asarray(joint_states, dtype=np.float64)
        z_arr = -(q[:, 1] + q[:, 2])

    variant = cfg.get("variant", "conservative")
    curve_overrides = {
        k: v for k, v in cfg.items()
        if k in ("peel_param", "y_grasp", "y_contact", "y_retry", "gamma", "theta", "retreat_frac")
    }
    curve_params = get_curve_params(variant=variant, **curve_overrides)

    return build_progress_curve(
        num_frames=N,
        attempts=seg["attempts"],
        t_first=seg["t_first"],
        t_last=seg["t_last"],
        t_lock=seg["t_lock"],
        z=z_arr,
        params=curve_params,
    )


__all__ = [
    "CurveParams",
    "CONSERVATIVE",
    "LEGACY80",
    "get_curve_params",
    "build_progress_curve",
    "DEFAULT_CLOSED_FRAC",
    "DEFAULT_MIN_CLOSED_FRAMES",
    "RAW_GRASP_DIMS",
    "HAND_PINCH_DIMS",
    "smooth",
    "runs",
    "extract_grasp_signal",
    "segment_grasp_attempts",
    "compute_progress",
]

