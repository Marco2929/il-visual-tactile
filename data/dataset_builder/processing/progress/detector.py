# dataset_builder/processing/progress/detector.py
"""Grasp detection and attempt segmentation for progress labelling."""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Tuned on 358 folien_* episodes:
# 0.45 ensures brief failed slips (e.g. 6-9 frames) are reliably detected.
DEFAULT_CLOSED_FRAC = 0.45
DEFAULT_MIN_CLOSED_FRAMES = 6

# rj_dg_2_2 .. rj_dg_2_4 in raw 26-dim joint recording layout
RAW_GRASP_DIMS = (11, 12, 13)

# Distal joints of finger 2 (index) in 20-dim or 8-dim hand layout
HAND_PINCH_DIMS = (5, 6, 7)


def smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    """Uniform moving average with replicate edge-padding."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0 or w <= 1:
        return x
    w = min(w, len(x))
    pad = w // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True spans [start, end) where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []
    padded = np.empty(len(mask) + 2, dtype=bool)
    padded[0] = False
    padded[1:-1] = mask
    padded[-1] = False
    diff = np.diff(padded.view(np.int8))
    starts = np.flatnonzero(diff > 0)
    stops = np.flatnonzero(diff < 0)
    return list(zip(starts.tolist(), stops.tolist()))


def extract_grasp_signal(
    joint_states: Optional[np.ndarray] = None,
    hand_state: Optional[np.ndarray] = None,
    finger_sensor: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract 1D finger flex angle signal (higher = more closed)."""
    if joint_states is not None and joint_states.ndim == 2 and joint_states.shape[1] >= 14:
        return np.asarray(joint_states)[:, list(RAW_GRASP_DIMS)].mean(axis=1)

    if hand_state is not None:
        arr = np.asarray(hand_state)
        if arr.ndim == 2:
            if arr.shape[1] >= 8:
                return arr[:, list(HAND_PINCH_DIMS)].mean(axis=1)
            return arr.mean(axis=1)
        return arr

    if finger_sensor is not None:
        arr = np.asarray(finger_sensor)
        if arr.ndim == 2:
            if arr.shape[1] >= 8:
                return arr[:, list(HAND_PINCH_DIMS)].mean(axis=1)
            return arr.mean(axis=1)
        # Check if legacy Inspire sensor (sensor < 800 convention)
        if len(arr) > 0 and np.nanpercentile(arr, 95) > 100:
            return -arr
        return arr

    raise ValueError("extract_grasp_signal requires joint_states, hand_state, or finger_sensor.")


def segment_grasp_attempts(
    sig: np.ndarray,
    smooth_w: int = 5,
    closed_frac: float = DEFAULT_CLOSED_FRAC,
    min_closed_frames: int = DEFAULT_MIN_CLOSED_FRAMES,
) -> Dict:
    """Segment grasp attempts and determine grasp locking frame.

    Returns:
        dict with:
          - n_frames: total frames
          - attempts: list of (start, end) frames for each detected grasp
          - n_attempts: number of attempts
          - t_first: start of first grasp attempt
          - t_last: start of final grasp attempt
          - t_lock: frame within final attempt where firm grip is established
          - has_grasp: bool
          - threshold: angle threshold used
    """
    sig = smooth(sig, smooth_w)
    T = len(sig)
    out = {
        "n_frames": T,
        "attempts": [],
        "n_attempts": 0,
        "t_first": -1,
        "t_last": -1,
        "t_lock": -1,
        "has_grasp": False,
        "threshold": float("nan"),
    }
    if T < 10:
        return out

    lo = float(np.percentile(sig, 5))
    hi = float(np.percentile(sig, 95))
    thr = lo + closed_frac * max(hi - lo, 1e-4)
    out["threshold"] = thr

    closed = sig >= thr
    attempts = [(int(a), int(b)) for a, b in runs(closed) if b - a >= min_closed_frames]
    out["attempts"] = attempts
    out["n_attempts"] = len(attempts)

    if not attempts:
        return out

    glast = int(attempts[-1][0])
    out["t_first"] = int(attempts[0][0])
    out["t_last"] = glast
    out["has_grasp"] = True

    # Firm grasp lock: frame where fingers reach solid contact (>= 70% closed span)
    thr_lock = lo + 0.70 * max(hi - lo, 1e-4)
    hit_lock = np.flatnonzero(sig[glast:] >= thr_lock)
    t_lock = glast + int(hit_lock[0]) if len(hit_lock) else glast
    out["t_lock"] = min(t_lock, min(glast + 25, T - 1))
    return out

