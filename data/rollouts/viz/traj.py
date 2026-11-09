#!/usr/bin/env python3
"""Compare commanded trajectories across rollouts.

Used by the proprioception-overfitting baseline (T1), the XY test (T3) and
the height test (T4), so the alignment choice is made once here instead of
three times with three different answers.

TWO ALIGNMENTS, because each one lies on its own:

  phase     t -> [0, 1] over the whole gate window. Shows overall shape, but
            a run that idles for three seconds at the start gets compressed
            relative to one that does not, which invents a difference.
  event     t measured from the commanded closure peak. The physically
            meaningful anchor: it lines up the moment the policy decided to
            grasp, so the approach can be compared without the gate timing
            leaking in.

`event` is the one to believe for the approach; `phase` is a cross-check.
When they disagree, the cause is almost always a difference in how long the
run sat still before moving, which startup_phase.py already measures.

Everything here works on the COMMANDED pose. No URDF, no forward kinematics:
the question is where the policy AIMED, and `cmd_pose` is exactly that.
FK would add the controller's tracking error on top, which is a different
question (arm_speed.py, pump_check.py).
"""
import numpy as np

from .attempts import best


def _xyz(r):
    return np.asarray(r.cmd_pose[:, :3], dtype=float)


def resample_phase(r, n=200):
    """-> (n, 3) commanded XYZ over normalised time."""
    t = np.asarray(r.cmd_t, dtype=float)
    xyz = _xyz(r)
    if len(t) < 2:
        return None
    span = t[-1] - t[0]
    if span <= 0:
        return None
    u = (t - t[0]) / span
    grid = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(grid, u, xyz[:, i]) for i in range(3)], axis=1)


def resample_event(r, pre=6.0, post=2.0, hz=20.0):
    """-> (times, (T, 3)) commanded XYZ around the closure peak.

    `times` is relative to the peak, so 0.0 is the instant of maximum
    commanded closure. Returns None when the rollout has no identifiable
    attempt, which is itself a finding and is reported as such by the
    callers rather than hidden by a fallback anchor.
    """
    a = best(r)
    if a is None:
        return None, None
    t = np.asarray(r.cmd_t, dtype=float)
    xyz = _xyz(r)
    t0 = a.t_peak
    grid = np.arange(-pre, post + 1e-9, 1.0 / hz)

    # Clipped, not extrapolated: a run that started 4 s before its peak has
    # no data at -6 s, and np.interp would silently repeat the first sample.
    # The mask tells the caller which part is real.
    lo, hi = t[0] - t0, t[-1] - t0
    valid = (grid >= lo) & (grid <= hi)
    if valid.sum() < 10:
        return None, None
    g = grid[valid]
    out = np.stack([np.interp(g + t0, t, xyz[:, i]) for i in range(3)],
                   axis=1)
    return g, out


def common_event_window(rollouts, pre=6.0, post=2.0, hz=20.0):
    """Event-aligned curves on ONE shared grid, so they can be differenced.

    -> (times, {name: (T, 3)}). The grid is the intersection of every
    rollout's valid span: comparing over a window where only some runs have
    data would make the RMSE depend on which run was shortest.
    """
    per = {}
    for r in rollouts:
        g, xyz = resample_event(r, pre, post, hz)
        if g is not None:
            per[r.name] = (g, xyz)
    if len(per) < 2:
        return None, {}

    lo = max(g[0] for g, _ in per.values())
    hi = min(g[-1] for g, _ in per.values())
    if hi - lo < 0.5:
        return None, {}

    grid = np.arange(lo, hi + 1e-9, 1.0 / hz)
    out = {}
    for name, (g, xyz) in per.items():
        out[name] = np.stack(
            [np.interp(grid, g, xyz[:, i]) for i in range(3)], axis=1)
    return grid, out


def rmse_mm(a, b):
    """3D RMS distance between two (T, 3) curves, in millimetres."""
    d = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)
    return float(np.sqrt(np.mean(d ** 2)) * 1000.0)


def max_dev_mm(a, b):
    d = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)
    return float(d.max() * 1000.0)


def pairwise(curves):
    """-> (names, (N, N) RMSE matrix in mm). Diagonal is 0."""
    names = list(curves)
    n = len(names)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = rmse_mm(curves[names[i]], curves[names[j]])
            m[i, j] = m[j, i] = v
    return names, m


def spread(curves):
    """Scalar summary of how much a set of curves differs.

    median/max over the off-diagonal pairwise RMSE. The median is the
    headline: with five runs there are ten pairs, and one odd run out should
    not set the number that gets quoted as the noise floor.
    """
    names, m = pairwise(curves)
    if len(names) < 2:
        return None
    iu = np.triu_indices(len(names), k=1)
    vals = m[iu]
    return {
        "n": len(names),
        "n_pairs": int(len(vals)),
        "median_mm": float(np.median(vals)),
        "p90_mm": float(np.percentile(vals, 90)),
        "max_mm": float(vals.max()),
        "min_mm": float(vals.min()),
    }


def point_spread(points):
    """Scatter of grasp positions. points: (N, 3) in metres.

    -> dict in millimetres. `radius_mm` is the mean distance from the
    centroid and `max_pair_mm` the widest separation; the first is the
    typical scatter, the second is what an outlier does to it.
    """
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return None
    c = p.mean(axis=0)
    d = np.linalg.norm(p - c, axis=1)
    pair = [np.linalg.norm(p[i] - p[j])
            for i in range(len(p)) for j in range(i + 1, len(p))]
    return {
        "n": len(p),
        "centroid": c,
        "radius_mm": float(d.mean() * 1000.0),
        "max_radius_mm": float(d.max() * 1000.0),
        "max_pair_mm": float(max(pair) * 1000.0),
        "std_mm": (p.std(axis=0) * 1000.0),
    }


def hover(r, band_mm=10.0, min_s=2.0):
    """How long the policy sat at its lowest commanded depth without going
    deeper. The detector for "it stops and does not travel further down".

    NOT a flatness test, and that correction matters. The first version of
    this looked for stretches where the commanded z changed by less than
    1 mm, and it reported 0.1-0.3 s for every single rollout - including the
    runs that visibly stalled. The reason is in the signal: the commanded z
    moves a MEDIAN of ~3 mm per tick over the whole run, so it is never flat
    at a millimetre scale. The policy dithers continuously; what a stalled
    run does is dither WITHOUT MAKING NET PROGRESS DOWNWARD.

    So the measurement is residence near the floor it reached: total and
    longest contiguous time with z within `band_mm` of that run's minimum.
    A run that touches its lowest point and lifts away spends little time
    there; one that arrives and gives up spends seconds.

    Commanded, not executed, on purpose: a stuck EXECUTED z can equally mean
    the arm is pressed against the layer and cannot move, which is the
    opposite diagnosis. A commanded z that stops descending is the policy
    itself no longer asking.

    -> dict or None.
    """
    t = np.asarray(r.cmd_t, dtype=float)
    z = np.asarray(r.cmd_pose[:, 2], dtype=float)
    if len(t) < 3:
        return None

    zmin = float(z.min())
    near = z <= zmin + band_mm / 1000.0

    # Longest contiguous run of `near`, in seconds.
    longest, start_i, end_i = 0.0, None, None
    i = 0
    while i < len(near):
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(near) and near[j + 1]:
            j += 1
        dur = float(t[j] - t[i])
        if dur > longest:
            longest, start_i, end_i = dur, i, j
        i = j + 1

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    total = float(near.sum()) * dt

    return {
        "z_min": zmin,
        "band_mm": band_mm,
        "total_s": total,
        "longest_s": longest,
        "t_start": float(t[start_i]) if start_i is not None else float("nan"),
        "t_end": float(t[end_i]) if end_i is not None else float("nan"),
        "is_stall": longest >= min_s,
    }


def descent_profile(r, after_s=1.0):
    """Net downward progress, to separate "never descended" from "descended
    then stopped".

    Reported as the time from the start of motion to the lowest commanded z,
    and how much of the run remains after it. A policy that reaches its
    floor at 40% of the run and then stays there is behaving differently
    from one that is still descending when the gate closes.
    """
    t = np.asarray(r.cmd_t, dtype=float)
    z = np.asarray(r.cmd_pose[:, 2], dtype=float)
    if len(t) < 3:
        return None
    sel = t >= t[0] + after_s
    if sel.sum() < 3:
        sel = np.ones(len(t), bool)
    tt, zz = t[sel], z[sel]
    k = int(np.argmin(zz))
    span = float(tt[-1] - tt[0])
    return {
        "t_zmin": float(tt[k] - tt[0]),
        "frac_to_zmin": float((tt[k] - tt[0]) / span) if span > 0 else 0.0,
        "after_zmin_s": float(tt[-1] - tt[k]),
        "descent_mm": float((zz[0] - zz[k]) * 1000.0),
    }
