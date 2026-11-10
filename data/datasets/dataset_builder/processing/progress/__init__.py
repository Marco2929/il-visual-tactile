# dataset_builder/processing/progress/__init__.py
"""Progress module for dataset builder.

Computes continuous, physically-grounded progress labels for the foil
pinch-and-peel task.
"""

from typing import Optional

import numpy as np

from .curve import (
    CurveParams,
    CONSERVATIVE,
    LEGACY80,
    STATE_APPROACH,
    STATE_PINCH_CONTACT,
    STATE_LIFT_SUCCESS,
    STATE_MISS_OR_LOST,
    GRASP_STATE_NAMES,
    get_curve_params,
    build_progress_curve,
    build_pchip_progress_curve,
    build_phase_labels,
    build_progress_from_phases,
)
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
    is_success: Optional[bool] = None,
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

    # End-effector z proxy if available
    z_arr = None
    if z is not None and len(z) == N:
        z_arr = z
    elif ee_pose is not None and ee_pose.ndim == 2 and ee_pose.shape[1] >= 3:
        z_arr = ee_pose[:, 2]
    elif joint_states is not None and joint_states.ndim == 2 and joint_states.shape[1] >= 3:
        q = np.asarray(joint_states, dtype=np.float64)
        z_arr = -(q[:, 1] + q[:, 2])

    seg = segment_grasp_attempts(
        sig,
        smooth_w=smooth_w,
        closed_frac=closed_frac,
        min_closed_frames=min_closed_frames,
        z=z_arr,
    )
    if not seg["has_grasp"]:
        return np.linspace(0.0, 1.0, N, dtype=np.float32)

    variant = cfg.get("variant", "conservative")
    success_flag = is_success if is_success is not None else cfg.get("is_success", True)

    if str(variant).lower() in ("pchip", "initial_pchip", "pchip_initial"):
        return build_pchip_progress_curve(
            num_frames=N,
            t_first=seg["t_first"],
            z=z_arr,
            is_success=success_flag,
        )

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
        is_success=success_flag,
    )


def compute_phase_labels(
    num_frames: Optional[int] = None,
    ee_pose: Optional[np.ndarray] = None,
    finger_sensor: Optional[np.ndarray] = None,
    progress_dict: Optional[dict] = None,
    *,
    hand_state: Optional[np.ndarray] = None,
    joint_states: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    is_success: Optional[bool] = None,
) -> np.ndarray:
    """Compute discrete grasp state labels (0..3) for one episode."""
    if joint_states is not None:
        N = len(joint_states)
    elif hand_state is not None:
        N = len(hand_state)
    elif finger_sensor is not None:
        N = len(finger_sensor)
    elif num_frames is not None:
        N = int(num_frames)
    else:
        raise ValueError("compute_phase_labels requires length or state array.")

    if N <= 0:
        return np.zeros(0, dtype=np.int64)

    try:
        sig = extract_grasp_signal(
            joint_states=joint_states,
            hand_state=hand_state,
            finger_sensor=finger_sensor,
        )
    except Exception:
        return np.zeros(N, dtype=np.int64)

    cfg = progress_dict or {}
    closed_frac = float(cfg.get("closed_frac", DEFAULT_CLOSED_FRAC))
    min_closed_frames = int(cfg.get("min_closed_frames", DEFAULT_MIN_CLOSED_FRAMES))
    smooth_w = int(cfg.get("smooth_w", 5))

    z_arr = None
    if z is not None and len(z) == N:
        z_arr = z
    elif ee_pose is not None and ee_pose.ndim == 2 and ee_pose.shape[1] >= 3:
        z_arr = ee_pose[:, 2]
    elif joint_states is not None and joint_states.ndim == 2 and joint_states.shape[1] >= 3:
        q = np.asarray(joint_states, dtype=np.float64)
        z_arr = -(q[:, 1] + q[:, 2])

    seg = segment_grasp_attempts(
        sig,
        smooth_w=smooth_w,
        closed_frac=closed_frac,
        min_closed_frames=min_closed_frames,
        z=z_arr,
    )
    if not seg["has_grasp"]:
        return np.full(N, STATE_APPROACH, dtype=np.int64)

    success_flag = is_success if is_success is not None else cfg.get("is_success", True)
    lift_delta_z = float(cfg.get("lift_delta_z", 0.015))

    return build_phase_labels(
        num_frames=N,
        attempts=seg["attempts"],
        t_lock=seg["t_lock"],
        sig=sig,
        z=z_arr,
        is_success=success_flag,
        lift_delta_z=lift_delta_z,
    )


__all__ = [
    "CurveParams",
    "CONSERVATIVE",
    "LEGACY80",
    "STATE_APPROACH",
    "STATE_PINCH_CONTACT",
    "STATE_LIFT_SUCCESS",
    "STATE_MISS_OR_LOST",
    "GRASP_STATE_NAMES",
    "get_curve_params",
    "build_progress_curve",
    "build_phase_labels",
    "DEFAULT_CLOSED_FRAC",
    "DEFAULT_MIN_CLOSED_FRAMES",
    "RAW_GRASP_DIMS",
    "HAND_PINCH_DIMS",
    "smooth",
    "runs",
    "extract_grasp_signal",
    "segment_grasp_attempts",
    "compute_progress",
    "compute_phase_labels",
]

