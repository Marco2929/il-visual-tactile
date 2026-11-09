#!/usr/bin/env python3
"""T3: does the policy follow the pallet in XY? (013 centre, 014 left, 015 right)

These three runs are the XY reference: same mid layer, pallet deliberately
moved to centre / left / right. Each run contains SEVERAL attempts, so
anything per-rollout would average a hit with a miss - everything here is
per attempt, using the one segmentation in viz/attempts.py.

The comparison that decides it:

    between-condition  distance between the centre / left / right centroids
    within-condition   scatter of the attempts inside one condition

If the pallet moved and the policy followed, the centroids separate by much
more than the attempts scatter. If between ~ within, the policy aims at the
same place regardless of where the pallet is.

The within-condition scatter also answers a second question for free: across
the retries of ONE run, does the policy search for a new spot or re-aim at
the same one? Re-aiming identically after a failure is the reactive-regrasp
claim failing in a specific, visible way.

LIMITATION: the real left/right displacement was never recorded, so a
separation can be called present or absent but not correct in magnitude.

Usage:
    python data/rollout_logs/xy_overfit.py
    python data/rollout_logs/xy_overfit.py --baseline nominal_baseline
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz import attempts, groups, traj  # noqa: E402


def collect(group, root):
    """-> (label, [Attempt]) for one condition."""
    out = []
    for r in groups.load_group(group, root=root):
        out.extend(attempts.segments(r))
    return out


def per_attempt_table(conds):
    print("  every attempt (commanded TCP at the closure peak)")
    print(f"  {'cond':>7s} {'run':>5s} {'#':>3s} {'t_peak':>8s} "
          f"{'x':>9s} {'y':>9s} {'z':>9s} {'cmd':>7s} {'exec':>7s} "
          f"{'defic':>7s}")
    print("  " + "-" * 78)
    for label, segs in conds:
        for s in segs:
            x, y, z = s.xyz
            print(f"  {label:>7s} {s.r.name[-3:]:>5s} {s.index:>3d} "
                  f"{s.t_peak:>7.2f}s {x:>9.4f} {y:>9.4f} {z:>9.4f} "
                  f"{s.peak_cmd_close:>7.1f} {s.peak_exec_close:>7.1f} "
                  f"{s.deficit:>7.1f}")
        if not segs:
            print(f"  {label:>7s}   no attempts found")


def within(conds):
    """Scatter inside each condition, i.e. across that run's retries."""
    print("\n  within condition (across the retries of the same run)")
    print(f"  {'cond':>7s} {'n':>3s} {'XY radius':>11s} {'XY max':>9s} "
          f"{'z spread':>10s}")
    print("  " + "-" * 46)
    radii = []
    for label, segs in conds:
        if len(segs) < 2:
            print(f"  {label:>7s} {len(segs):>3d}   (needs >=2 attempts)")
            continue
        p = np.array([s.xyz for s in segs])
        xy = p[:, :2]
        d = np.linalg.norm(xy - xy.mean(axis=0), axis=1) * 1000
        zs = (p[:, 2].max() - p[:, 2].min()) * 1000
        radii.append(d.mean())
        print(f"  {label:>7s} {len(segs):>3d} {d.mean():>10.1f}m "
              f"{d.max():>8.1f}m {zs:>9.1f}m")
    return float(np.mean(radii)) if radii else float("nan")


def between(conds):
    """Centroid separation between the three pallet positions."""
    cents = {}
    for label, segs in conds:
        if segs:
            cents[label] = np.array([s.xyz for s in segs]).mean(axis=0)

    if len(cents) < 2:
        print("\n  (fewer than two conditions have attempts)")
        return float("nan"), cents

    print("\n  centroid per condition")
    for label, c in cents.items():
        print(f"    {label:>7s}  {c[0]:+.4f} {c[1]:+.4f} {c[2]:+.4f} m")

    print("\n  between-condition centroid distance")
    labels = list(cents)
    dists = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = cents[labels[i]], cents[labels[j]]
            xy = np.linalg.norm((a - b)[:2]) * 1000
            full = np.linalg.norm(a - b) * 1000
            dists.append(xy)
            print(f"    {labels[i]:>7s} <-> {labels[j]:<7s} "
                  f"XY {xy:7.1f} mm   3D {full:7.1f} mm")
    return float(np.mean(dists)), cents


def verdict(btw, wtn, base_xy):
    print("\n  verdict")
    if not np.isfinite(btw) or not np.isfinite(wtn):
        print("    not enough data to decide")
        return

    ratio = btw / max(wtn, 1e-9)
    print(f"    mean between-condition XY separation : {btw:7.1f} mm")
    print(f"    mean within-condition XY scatter     : {wtn:7.1f} mm")
    if np.isfinite(base_xy):
        print(f"    T1 baseline XY scatter (pallet fixed): {base_xy:7.1f} mm")
    print(f"    ratio between/within                 : {ratio:7.2f}")

    if ratio < 1.5:
        print("\n    * The three pallet positions produce grasp points no")
        print("      further apart than the retries within a single run.")
        print("      The policy aims at the same place regardless of where")
        print("      the pallet is -> XY overfitting confirmed.")
    elif ratio < 3.0:
        print("\n    * Partial response: the centroids do separate, but not")
        print("      by much more than the within-run scatter. Without the")
        print("      real pallet offsets this cannot be graded further.")
    else:
        print("\n    * The grasp point clearly follows the pallet position.")
        print("      XY overfitting is NOT supported; compare the separation")
        print("      against the real offsets to judge whether the size of")
        print("      the response is right.")

    if np.isfinite(base_xy) and btw < base_xy:
        print("\n    NOTE: the between-condition separation is smaller than")
        print("    the run-to-run scatter measured with the pallet NOT moved")
        print("    deliberately (T1). Whatever separation exists is within")
        print("    the policy's own noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=groups.CKPT)
    ap.add_argument("--baseline", default="nominal_baseline",
                    help="group for the T1 yardstick; '' to skip")
    a = ap.parse_args()

    root = groups.root_for(a.ckpt)
    print("=== T3 XY response to pallet position ===")
    for g in groups.XY_GROUPS:
        print(f"  {g.xy:>7s}: {', '.join(g.names)}  ({g.label})")
    print("  pallet offsets were not recorded -> qualitative only\n")

    conds = [(g.xy, collect(g, root)) for g in groups.XY_GROUPS]
    per_attempt_table(conds)
    wtn = within(conds)
    btw, _ = between(conds)

    base_xy = float("nan")
    if a.baseline:
        bg = groups.BY_KEY.get(a.baseline)
        if bg is not None:
            pts = []
            for r in groups.load_group(bg, root=root, verbose=False):
                b = attempts.best(r)
                if b is not None:
                    pts.append(b.xyz)
            if len(pts) >= 2:
                xy = np.array(pts)[:, :2]
                base_xy = float(np.linalg.norm(
                    xy - xy.mean(axis=0), axis=1).mean() * 1000)

    verdict(btw, wtn, base_xy)


if __name__ == "__main__":
    main()
