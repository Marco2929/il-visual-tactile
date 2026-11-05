# dataset_builder/processing/progress/curve.py
"""Progress curve parameterisation and construction."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CurveParams:
    """Parameters of the progress curve labelling."""
    name: str = "conservative"
    y_grasp: float = 0.40
    y_contact: float = 0.32
    y_retry: float = 0.20
    gamma: float = 1.5
    theta: float = 0.90
    retreat_frac: float = 1.0 / 3.0
    peel_param: str = "frames"  # "frames" or "height"

    def __post_init__(self):
        if not 0.0 < self.y_retry < self.y_contact < self.y_grasp < 1.0:
            raise ValueError(
                f"{self.name}: must satisfy 0 < y_retry ({self.y_retry}) "
                f"< y_contact ({self.y_contact}) < y_grasp ({self.y_grasp}) < 1")
        if self.gamma <= 0:
            raise ValueError(f"{self.name}: gamma must be > 0, got {self.gamma}")


CONSERVATIVE = CurveParams(
    name="conservative",
    y_grasp=0.40,
    y_contact=0.32,
    y_retry=0.20,
    gamma=1.5,
    theta=0.90,
    retreat_frac=1.0 / 3.0,
    peel_param="frames",
)

LEGACY80 = CurveParams(
    name="legacy80",
    y_grasp=0.80,
    y_contact=0.72,
    y_retry=0.60,
    gamma=1.0,
    theta=0.90,
    retreat_frac=1.0 / 3.0,
    peel_param="frames",
)


def get_curve_params(variant: str = "conservative", **overrides) -> CurveParams:
    """Get curve parameters by variant name with optional field overrides."""
    base = LEGACY80 if str(variant).lower() == "legacy80" else CONSERVATIVE
    if not overrides:
        return base
    kwargs = {
        "name": overrides.get("name", f"{base.name}_custom"),
        "y_grasp": float(overrides.get("y_grasp", base.y_grasp)),
        "y_contact": float(overrides.get("y_contact", base.y_contact)),
        "y_retry": float(overrides.get("y_retry", base.y_retry)),
        "gamma": float(overrides.get("gamma", base.gamma)),
        "theta": float(overrides.get("theta", base.theta)),
        "retreat_frac": float(overrides.get("retreat_frac", base.retreat_frac)),
        "peel_param": str(overrides.get("peel_param", base.peel_param)),
    }
    return CurveParams(**kwargs)


def build_progress_curve(
    num_frames: int,
    attempts: List[Tuple[int, int]],
    t_first: Optional[int] = None,
    t_last: Optional[int] = None,
    t_lock: Optional[int] = None,
    z: Optional[np.ndarray] = None,
    params: CurveParams = CONSERVATIVE,
    is_success: bool = True,
) -> np.ndarray:
    """Generate smooth, continuous progress curve.
    
    If is_success=True: peel phase ramps yg -> 1.0.
    If is_success=False: failed/missed grasp drops yg -> 0.0 during lift.
    """
    T = int(num_frames)
    if T <= 0:
        raise ValueError("empty episode")
    if not attempts:
        return np.linspace(0.0, 1.0 if is_success else 0.0, T, dtype=np.float32)

    g0 = int(attempts[0][0])
    glast = int(attempts[-1][0])

    y = np.zeros(T, dtype=np.float32)
    yc = params.y_contact
    yg = params.y_grasp
    yr = params.y_retry
    gamma = params.gamma

    # --- Phase 1: initial approach (0 to g0) ---
    if g0 > 0:
        t_app = np.linspace(0.0, 1.0, g0 + 1)
        y[:g0 + 1] = yc * (t_app ** gamma)
    else:
        y[0] = yc

    # --- Phases 2-4: failed attempts between g0 and glast ---
    for i in range(len(attempts) - 1):
        gt, rt = attempts[i]
        next_gt = attempts[i + 1][0]
        rt = min(rt, next_gt)
        t_ret = rt + max(1, int(round((next_gt - rt) * params.retreat_frac)))
        t_ret = min(t_ret, next_gt)

        # (a) Pinch / grasp: smoothly rises yc -> yg (quarter-sine)
        if rt > gt:
            p_pinch = np.linspace(0.0, 1.0, rt - gt + 1)
            y[gt:rt + 1] = yc + (yg - yc) * np.sin(0.5 * np.pi * p_pinch)

        # (b) Release / retreat: smooth drop yg -> yr (raised cosine)
        if t_ret > rt:
            p_drop = np.linspace(0.0, 1.0, t_ret - rt + 1)
            y[rt:t_ret + 1] = yg - (yg - yr) * 0.5 * (1.0 - np.cos(np.pi * p_drop))

        # (c) Re-approach: smooth rise yr -> yc (power curve)
        if next_gt > t_ret:
            p_re = np.linspace(0.0, 1.0, next_gt - t_ret + 1)
            y[t_ret:next_gt + 1] = yr + (yc - yr) * (p_re ** gamma)

    # --- Phase 5: final grasp lock and peel ---
    t_end_pinch = int(t_lock) if t_lock is not None and t_lock >= glast else glast
    t_end_pinch = min(t_end_pinch, T - 1)

    # (a) Grip lock: fingers flex onto foil, yc -> yg
    if t_end_pinch > glast:
        p_pinch = np.linspace(0.0, 1.0, t_end_pinch - glast + 1)
        y[glast:t_end_pinch + 1] = yc + (yg - yc) * np.sin(0.5 * np.pi * p_pinch)

    # (b) Peel ramp: foil is peeled, yg -> 1.0
    n_peel = T - t_end_pinch
    if n_peel > 0:
        if not is_success:
            # Grasp missed/failed: drop cleanly and rapidly to 0.0 within ~20 frames,
            # keeping progress flat at 0.0 during the remaining empty lift.
            drop_len = min(n_peel, 20)
            if drop_len > 0:
                p_drop = np.linspace(0.0, 1.0, drop_len)
                y[t_end_pinch : t_end_pinch + drop_len] = yg * ((1.0 - p_drop) ** 2)
                y[t_end_pinch + drop_len :] = 0.0
            else:
                y[t_end_pinch:] = 0.0
        elif params.peel_param == "height" and z is not None and len(z) == T:
            seg = np.maximum.accumulate(np.asarray(z[t_end_pinch:], dtype=np.float64))
            lifted = seg - seg[0]
            travel = float(lifted[-1])
            if travel > 1e-6:
                p_h = np.clip(lifted / travel, 0.0, 1.0)
                p_f = np.linspace(0.0, 1.0, n_peel)
                p_peel = 0.5 * p_h + 0.5 * p_f
            else:
                p_peel = np.linspace(0.0, 1.0, n_peel)
            y[t_end_pinch:] = yg + (1.0 - yg) * p_peel
        else:
            p_peel = np.linspace(0.0, 1.0, n_peel)
            y[t_end_pinch:] = yg + (1.0 - yg) * p_peel

    y = np.clip(y, 0.0, 1.0)
    y[-1] = 1.0 if is_success else 0.0
    return y.astype(np.float32)


def build_pchip_progress_curve(
    num_frames: int,
    t_first: int,
    z: Optional[np.ndarray] = None,
    is_success: bool = True,
) -> np.ndarray:
    """The original 4-point cubic spline (PCHIP) progress curve.
    
    Reproduces the exact original logic from early dataset builder:
      - x = [0, idx_grasp, idx_peak, idx_end]
      - y = [0.0, 0.5, 0.8, 1.0] (or 0.0 on failure)
    """
    from scipy.interpolate import PchipInterpolator
    T = int(num_frames)
    if T <= 0:
        return np.zeros(0, dtype=np.float32)
    if t_first < 0 or t_first >= T:
        return np.linspace(0.0, 1.0 if is_success else 0.0, T, dtype=np.float32)

    idx_start = 0
    idx_end = T - 1
    idx_grasp = int(t_first)

    # In original logic: idx_peak was the z-peak after grasp
    if z is not None and len(z) == T and idx_grasp < T - 1:
        idx_peak = idx_grasp + int(np.argmax(z[idx_grasp:]))
    else:
        idx_peak = (idx_grasp + idx_end) // 2

    x_points = [idx_start, idx_grasp]
    y_points = [0.0, 0.5]

    if idx_grasp < idx_peak < idx_end:
        x_points.append(idx_peak)
        y_points.append(0.8)

    x_points.append(idx_end)
    y_points.append(1.0 if is_success else 0.0)

    x_points, indices = np.unique(x_points, return_index=True)
    y_points = np.array(y_points)[indices]

    frames = np.arange(T)
    interpolator = PchipInterpolator(x_points, y_points)
    progress_signal = interpolator(frames)
    return np.clip(progress_signal, 0.0, 1.0).astype(np.float32)


