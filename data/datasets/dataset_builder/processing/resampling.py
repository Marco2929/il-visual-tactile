"""Joint-level + force smoothing and decimation.

Runs in the dataset builder BEFORE forward kinematics.
State and force are smoothed INDEPENDENTLY, but decimated with a
SHARED keep_idx (kept time-aligned with RGB/PC).
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt


def measure_fps(episode_dir):
    """Robustly measure the actual source rate from frame timestamps."""
    ts = np.load(episode_dir / "frame_timestamps_ns.npy")
    dt = np.diff(ts) / 1e9
    return 1.0 / np.median(dt)


def _smooth(x, cfg, fps):
    if not cfg:
        return x
    method, n = cfg.get("method", "none"), x.shape[0]
    if method == "none" or n < 5:
        return x

    if method == "savgol":
        win = cfg.get("savgol_window", 7)
        win = min(win, n if n % 2 == 1 else n - 1)   # <= n, must be odd
        if win % 2 == 0:
            win -= 1
        if win < 3:
            return x
        poly = min(cfg.get("savgol_polyorder", 3), win - 1)
        return savgol_filter(x, win, poly, axis=0)

    if method == "butter":
        wn = min(cfg.get("butter_cutoff_hz", 4.0) / (0.5 * fps), 0.99)
        b, a = butter(cfg.get("butter_order", 4), wn, btype="low")
        return filtfilt(b, a, x, axis=0)

    return x


def compute_keep_idx(n_src, target_fps, src_fps, enabled):
    """Shared target indices for all modalities (nearest-index)."""
    if not enabled or target_fps >= src_fps:
        return np.arange(n_src)
    n_tgt = max(1, round(n_src * target_fps / src_fps))
    return np.round(np.linspace(0, n_src - 1, n_tgt)).astype(int)


def resample_episode(state, force, target_fps, src_fps,
                     state_smooth=None, force_smooth=None, enabled=True):
    """
    state: (N, 12) joint positions (6 arm + 6 hand)
    force: (N, 6) or None  (currently inactive -> None allowed)
    returns: state_ds, force_ds, keep_idx
    """
    n_src = state.shape[0]

    # --- 1) State: independent smoothing ---
    q = state.astype(np.float64)
    q[:, :6] = np.unwrap(q[:, :6], axis=0)      # unwrap arm joints only
    q = _smooth(q, state_smooth, src_fps)

    # --- 2) Force: independent smoothing (if present) ---
    f = None
    if force is not None:
        f = _smooth(force.astype(np.float64), force_smooth, src_fps)

    # --- 3) Shared decimation ---
    keep_idx = compute_keep_idx(n_src, target_fps, src_fps, enabled)

    state_ds = q[keep_idx].astype(state.dtype)
    force_ds = f[keep_idx].astype(force.dtype) if f is not None else None
    return state_ds, force_ds, keep_idx