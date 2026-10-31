#!/usr/bin/env python3
"""Text reports: the cross-rollout table and the filter replays.

One row per rollout, so a systematic offset is distinguishable from scatter.
The deficit column is the measurement that matters: the filter replay shows
the executor's chain costs ~1 deg, so anything much larger is below the topic
layer (controller or hardware), not in our code.
"""
import numpy as np

from . import filters as flt
from .data import GRASP_LOCAL, POSE_DIM, flag_outliers


def _rule(n=104):
    return "-" * n


def summary_table(rollouts, flagged=None):
    """Per-rollout table plus aggregates and the two correlations."""
    flagged = flagged if flagged is not None else flag_outliers(rollouts)
    stats = [r.stats() for r in rollouts]
    lines = []
    a = lines.append

    a(f"{'rollout':<13} {'dur':>6} {'steps':>6} {'cmd':>7} {'exec':>7} "
      f"{'defic':>7} {'%':>6} {'reopen':>7} {'lag':>7} {'z_close':>8} "
      f"{'z_rise':>7}  notes")
    a(_rule())
    for s in stats:
        mark = " !" if s["name"] in flagged else "  "
        a(f"{s['name']:<13} {s['dur']:>5.0f}s {s['steps']:>6d} "
          f"{s['peak_cmd']:>7.1f} {s['peak_exec']:>7.1f} "
          f"{s['deficit']:>7.1f} {s['deficit_pct']:>5.0f}% "
          f"{s['reopen']:>7.1f} {s['lag']:>+6.2f}s {s['z_at_close']:>8.4f} "
          f"{s['z_rise']:>+7.3f}{mark}{s['notes'][:34]}")

    # Flagged runs are listed but excluded from every aggregate below: a
    # non-grasp has no meaningful closure deficit, and including one moved
    # corr(commanded, deficit) from +0.496 to +0.934.
    good = [s for s in stats if s["name"] not in flagged]
    if flagged:
        a("")
        a(f"excluded from the aggregates ({len(flagged)}):")
        for name, why in flagged.items():
            a(f"  {name}: {why}")
    if not good:
        return "\n".join(lines)

    def col(k):
        return np.array([s[k] for s in good], dtype=float)

    a("")
    a(f"--- aggregate over {len(good)} grasped rollouts ---")
    for key, label, unit in [
        ("peak_cmd", "commanded closure", "deg"),
        ("peak_exec", "executed closure", "deg"),
        ("deficit", "deficit", "deg"),
        ("deficit_pct", "deficit", "%"),
        ("reopen", "reopening after peak", "deg"),
        ("lag", "command -> execution lag", "s"),
        ("z_at_close", "z at closure", "m"),
        ("z_rise", "z rise after closure", "m"),
    ]:
        c = col(key)
        a(f"{label:<26} median {np.median(c):8.3f} {unit:<4} "
          f"p10 {np.percentile(c, 10):8.3f}  p90 {np.percentile(c, 90):8.3f}  "
          f"min {c.min():8.3f}  max {c.max():8.3f}")

    d, rp = col("deficit"), col("reopen")
    a("")
    a(f"deficit > 10 deg : {int((d > 10).sum())} / {len(good)}")
    a(f"deficit >  5 deg : {int((d > 5).sum())} / {len(good)}")
    a(f"reopening > 5 deg: {int((rp > 5).sum())} / {len(good)}")

    # Is the deficit constant, or does it scale with how far the policy asks
    # the hand to close? A fixed offset points at a calibration or
    # steady-state error; a proportional one at a torque/current limit.
    if len(good) > 4:
        pc, zc = col("peak_cmd"), col("z_at_close")
        if pc.std() > 1e-6 and d.std() > 1e-6:
            a(f"corr(commanded, deficit) : {np.corrcoef(pc, d)[0, 1]:+.3f}")
        if zc.std() > 1e-6 and d.std() > 1e-6:
            a(f"corr(z_at_close, deficit): {np.corrcoef(zc, d)[0, 1]:+.3f}")
    return "\n".join(lines)


def hand_filter_report(r):
    """How much of the commanded closure survives the hand filter chain."""
    hand = r.cmd_hand
    lines = [f"=== hand filters: {r.name} ===",
             f"executed {len(r.steps)} steps, "
             f"step {int(r.steps[0])}..{int(r.steps[-1])}",
             "",
             "grasp finger 2, sum of 4 joints (deg)",
             f"{'variant':<26} {'peak':>8} {'loss':>8} {'t_peak':>8}",
             _rule(54)]
    base = None
    for name, kw in flt.HAND_VARIANTS:
        out, _ = flt.run_hand_chain(hand, **kw)
        s = out[:, GRASP_LOCAL].sum(axis=1)
        peak = float(s.max())
        if base is None:
            base = peak
        lines.append(f"{name:<26} {peak:>8.2f} {base - peak:>8.2f} "
                     f"{int(np.argmax(s)) / flt.CONTROL_HZ:>7.1f}s")

    # Per-joint, to see whether a single joint is rate limited rather than
    # the sum being slightly reduced.
    full, _ = flt.run_hand_chain(hand)
    lines += ["", "per joint, commanded vs filtered (deg)",
              f"{'joint':<10} {'cmd max':>9} {'filt max':>9} {'loss':>7} "
              f"{'cmd maxrate':>12} {'clamped':>8}", _rule(62)]
    for k, j in enumerate(GRASP_LOCAL, start=1):
        c, f = hand[:, j], full[:, j]
        rate = np.abs(np.diff(c)) if len(c) > 1 else np.array([0.0])
        lines.append(f"rj_dg_2_{k:<2} {c.max():>9.2f} {f.max():>9.2f} "
                     f"{c.max() - f.max():>7.2f} {rate.max():>11.2f} "
                     f"{int((rate > flt.HAND_MAX_STEP_DEG).sum()):>8d}")

    rate_all = np.abs(np.diff(hand, axis=0))
    lines += ["", "rate limit relevance (all 20 joints)",
              f"max commanded |delta| per tick : {rate_all.max():.2f} deg "
              f"(limit {flt.HAND_MAX_STEP_DEG})",
              f"ticks exceeding the clamp      : "
              f"{int((rate_all > flt.HAND_MAX_STEP_DEG).sum())} "
              f"of {rate_all.size}",
              f"median commanded |delta|       : "
              f"{np.median(rate_all):.3f} deg "
              f"(deadband {flt.HAND_DEADBAND_DEG})",
              f"ticks inside the deadband      : "
              f"{int((rate_all < flt.HAND_DEADBAND_DEG).sum())} "
              f"of {rate_all.size} "
              f"({100 * (rate_all < flt.HAND_DEADBAND_DEG).mean():.0f}%)"]
    return "\n".join(lines)


def pose_filter_report(r):
    """Same question for the cartesian side, in metres and degrees."""
    pose = r.cmd_pose[:, :POSE_DIM]
    lines = [f"=== pose filters: {r.name} ===",
             f"executed {len(r.steps)} steps", "", "commanded pose range"]
    for i, ax in enumerate("xyz"):
        lo, hi = flt.WORKSPACE[ax]
        c = pose[:, i]
        ok = "OK" if c.min() >= lo and c.max() <= hi else "OUT OF BOX"
        lines.append(f"{ax}: {c.min():+.4f} .. {c.max():+.4f} m   "
                     f"box {lo:+.3f} .. {hi:+.3f}   {ok}")
    qn = np.linalg.norm(pose[:, 3:7], axis=1)
    lines.append(f"quat norm as published: {qn.min():.6f} .. {qn.max():.6f} "
                 f"(deploy renormalises, so ~1)")

    lines += ["", "effect on the trajectory",
              f"{'variant':<26} {'z min':>8} {'z max':>8} {'pos rmse':>9} "
              f"{'rot rmse':>9} {'held':>5}", _rule(70)]
    ref = None
    for name, kw in flt.POSE_VARIANTS:
        out, st = flt.run_pose_chain(pose, **kw)
        if ref is None:
            ref = out
        pos_rmse = float(np.sqrt(np.mean(
            np.sum((out[:, :3] - ref[:, :3]) ** 2, axis=1))))
        rot = np.array([flt.quat_angle_deg(out[t, 3:7], ref[t, 3:7])
                        for t in range(len(out))])
        lines.append(f"{name:<26} {out[:, 2].min():>8.4f} "
                     f"{out[:, 2].max():>8.4f} {pos_rmse * 1000:>8.2f}mm "
                     f"{rot.mean():>8.2f}d {st['held']:>5d}")

    out, st = flt.run_pose_chain(pose)
    lines += ["", "stage activity (full chain)",
              f"translation clamp (>{flt.MAX_POSITION_STEP} m/tick): "
              f"{st['pos_clamped']} of {len(pose)}",
              f"rotation clamp (chord >{flt.MAX_ROTATION_STEP})     : "
              f"{st['rot_clamped']} of {len(pose)}",
              f"ticks REJECTED (arm holds)             : {st['rejected']}",
              f"quat norm after Butterworth            : "
              f"{st['quat_norm_min']:.6f} .. {st['quat_norm_max']:.6f}"]

    step = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    rot_step = np.array([flt.quat_angle_deg(pose[t, 3:7], pose[t + 1, 3:7])
                         for t in range(len(pose) - 1)])
    lines += ["", "commanded rates",
              f"max translation per tick : {step.max() * 1000:.1f} mm "
              f"(limit {flt.MAX_POSITION_STEP * 1000:.0f} mm)",
              f"median                   : {np.median(step) * 1000:.2f} mm",
              f"max rotation per tick    : {rot_step.max():.2f} deg",
              f"median                   : {np.median(rot_step):.3f} deg"]

    # Phase-resolved: a filter that only matters during the fast descent is
    # invisible in a whole-rollout RMSE.
    zc, zf = pose[:, 2], out[:, 2]
    err = zf - zc
    k = int(np.argmin(zc))
    lines += ["", "z tracking, commanded vs full chain",
              f"max |z error| : {np.abs(err).max() * 1000:.2f} mm "
              f"at step {int(np.argmax(np.abs(err)))}",
              f"mean z error  : {err.mean() * 1000:+.2f} mm",
              f"at lowest commanded z (step {k}): cmd {zc[k]:.4f} -> "
              f"filt {zf[k]:.4f} ({(zf[k] - zc[k]) * 1000:+.2f} mm)"]
    return "\n".join(lines)