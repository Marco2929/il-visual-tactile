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
    min_open_frames: int = 15,
    min_span: float = 0.0,
    z: Optional[np.ndarray] = None,
    max_contact_height: float = 0.035,
) -> Dict:
    """Segment grasp attempts at the part and determine grasp locking frame.

    Distinguishes pinches at the part (within `max_contact_height` of episode z_min)
    from pinches in the air:
      - Pre-contact mid-air finger flexes during approach are ignored.
      - Post-lift mid-air re-squeezes (where the hand already grasped at the part,
        lifted up into the air, and adjusted grip) are merged into the active hold
        rather than misclassified as a new retry at the part.
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
    if (hi - lo) < min_span:
        return out

    thr = lo + closed_frac * max(hi - lo, 1e-4)
    out["threshold"] = thr

    closed = sig >= thr
    # Require each candidate closed run to reach a genuine pinch peak (>= 62% of hand span),
    # filtering out half-closed repositioning hovers (e.g. ep_174 f153..219 at 51% span).
    peak_thr = lo + max(closed_frac, 0.62) * max(hi - lo, 1e-4)
    raw_attempts = [
        (int(a), int(b))
        for a, b in runs(closed)
        if (b - a >= min_closed_frames) and (float(np.max(sig[a:b])) >= peak_thr)
    ]
    if not raw_attempts:
        raw_attempts = [(int(a), int(b)) for a, b in runs(closed) if b - a >= min_closed_frames]

    # Merge consecutive closed runs separated by brief open dips (< min_open_frames),
    # UNLESS the hand clearly opened (dropped below lo + 0.30*(hi-lo)) between them.
    open_reset_thr = lo + 0.30 * max(hi - lo, 1e-4)
    merged = []
    for a, b in raw_attempts:
        if merged and (a - merged[-1][1]) < min_open_frames:
            gap_min = float(np.min(sig[merged[-1][1]:a])) if a > merged[-1][1] else float(sig[a])
            if gap_min >= open_reset_thr:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        else:
            merged.append((a, b))

    # Filter out mid-air pinches if height z is provided
    if z is not None and len(z) == T and len(merged) > 1:
        z_arr = np.asarray(z, dtype=np.float64)
        z_min = float(np.min(z_arr))
        contact_ceil = z_min + max_contact_height

        part_attempts = []
        for a, b in merged:
            # Check lowest height reached just before or during the closure onset
            win_a = max(0, a - 5)
            z_att_min = float(np.min(z_arr[win_a:b]))
            z_att_start = float(z_arr[a])

            if z_att_min <= contact_ceil:
                # Genuine pinch at the part (pallet/foil level)
                part_attempts.append((a, b))
            else:
                # Pinch above z_min + max_contact_height:
                # Check if the arm descended significantly (>= 4 cm) from its prior height
                # to make a fresh retry (e.g. ep_498), vs staying/lifting in the air (e.g. ep_144, ep_328).
                prev_end = part_attempts[-1][1] if part_attempts else 0
                z_peak_since_prev = float(np.max(z_arr[prev_end:a + 1])) if a >= prev_end else z_att_start
                descended_for_retry = (z_peak_since_prev - z_att_min) >= 0.040 and (z_att_min <= z_min + 0.075)

                if descended_for_retry:
                    part_attempts.append((a, b))
                elif part_attempts and z_att_start > contact_ceil:
                    # Hand already grasped at the part and is now up in the air
                    # re-tightening/adjusting its grip -> extend the prior grasp!
                    part_attempts[-1] = (part_attempts[-1][0], b)
                else:
                    # Pre-contact mid-air finger flex while still approaching -> ignore
                    pass

        if part_attempts:
            attempts = part_attempts
        else:
            attempts = merged
    else:
        attempts = merged

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

