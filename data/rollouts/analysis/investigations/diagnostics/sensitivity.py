#!/usr/bin/env python3
"""Why are two runs of the SAME condition one success and one failure?

The observation this is built for: both policies sometimes work almost
perfectly ("vorgestern war die absolute policy in quasi allen Faellen der
Nominalhoehe in der Lage zu greifen") and sometimes fail on what looks like
the same setup. The relative findings name three candidates and cannot
separate them at the rig: the wrist RGB, the pallet position, or the arm's
START POSE.

The start pose is the one the logs can settle, because it is recorded
exactly: /joint_states before the policy has done anything. If successes and
failures of the same condition separate by start pose, the sensitivity is
proprioceptive and has nothing to do with the camera. If they do not, the
start pose is exonerated and the remaining candidates are visual.

Four comparisons, within each condition so the condition cannot explain the
difference:

  1. START POSE    first commanded and first measured TCP pose, successes
                   against failures.
  2. APPROACH      where the policy went in its first second. A policy whose
                   opening move differs run to run is reacting to something
                   that differs run to run.
  3. CONVERGENCE   do the paths come together later, or stay apart? Coming
                   together means the start offset was recovered; staying
                   apart means it propagated into the grasp.
  4. OFFSET vs OUTCOME  distance from this run's start pose to the median
                   start pose of the SUCCESSES, plotted against the verdict.
                   A monotone relation is the quantitative version of "it
                   works when it starts in the right place".

Usage:
    python data/rollout_logs/sensitivity.py
    python data/rollout_logs/sensitivity.py --ckpt foil_lifting_relative_e40
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz import attempts, labels as lab  # noqa: E402
from viz.data import Rollout  # noqa: E402
from viz.filters import quat_angle_deg  # noqa: E402

CKPTS = ["foil_lifting_wrist_e40", "foil_lifting_relative_e40"]
ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(ckpt):
    d = ROOT / ckpt
    labels = lab.load(d)
    rs = {}
    for name in sorted(labels):
        r = Rollout(d / name)
        if r.ok:
            rs[name] = r
    return labels, rs


def start_pose(r):
    """First commanded TCP pose, i.e. where the policy began from."""
    return np.asarray(r.cmd_pose[0, :3], float), np.asarray(
        r.cmd_pose[0, 3:7], float)


def early_path(r, seconds=1.0):
    t = np.asarray(r.cmd_t, float)
    sel = t <= t[0] + seconds
    if sel.sum() < 3:
        return None
    return np.asarray(r.cmd_pose[sel, :3], float)


def block_start(labels, rs):
    """Start pose of every run, grouped by condition, success marked."""
    print("\n  1. START POSE  (first commanded TCP, by condition)")
    by_cond = {}
    for name, r in rs.items():
        by_cond.setdefault(labels[name].conditions, []).append((name, r))

    for cond, rows in by_cond.items():
        if len(rows) < 2:
            continue
        print(f"\n    {cond}")
        pts, oks = [], []
        for name, r in sorted(rows):
            p, q = start_pose(r)
            ok = labels[name].ok
            pts.append(p)
            oks.append(ok)
            print(f"      {name[-3:]} {'OK' if ok else '--'} "
                  f"start {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}")
        pts = np.array(pts)
        oks = np.array(oks)
        spread = np.linalg.norm(pts - pts.mean(axis=0), axis=1) * 1000
        print(f"      start-pose scatter: mean {spread.mean():.1f} mm, "
              f"max {spread.max():.1f} mm")
        if oks.any() and (~oks).any():
            d = np.linalg.norm(pts[oks].mean(axis=0)
                               - pts[~oks].mean(axis=0)) * 1000
            print(f"      success centroid vs failure centroid: {d:.1f} mm")


def block_offset(labels, rs):
    """Distance from the successes' start pose, against the verdict.

    Pooled over conditions on purpose: the question is whether STARTING
    somewhere unusual predicts failure regardless of which condition was
    being tested.
    """
    print("\n  4. START OFFSET vs OUTCOME")
    good = [np.asarray(r.cmd_pose[0, :3], float)
            for n, r in rs.items() if labels[n].ok]
    if len(good) < 2:
        print("    fewer than two successes - cannot form a reference")
        return
    ref = np.median(np.array(good), axis=0)
    print(f"    reference = median start pose of the {len(good)} "
          f"successes: {ref[0]:+.4f} {ref[1]:+.4f} {ref[2]:+.4f}")

    rows = []
    for name, r in rs.items():
        d = float(np.linalg.norm(np.asarray(r.cmd_pose[0, :3], float) - ref)
                  * 1000)
        rows.append((d, name, labels[name].ok, labels[name].failure_mode))
    rows.sort()

    print(f"    {'dist':>8s} {'run':>5s} {'verdict':>9s}  failure mode")
    for d, name, ok, fm in rows:
        print(f"    {d:>7.1f}m {name[-3:]:>5s} "
              f"{'success' if ok else 'fail':>9s}  {fm}")

    ok_d = np.array([d for d, _, ok, _ in rows if ok])
    bad_d = np.array([d for d, _, ok, _ in rows if not ok])
    if len(ok_d) and len(bad_d):
        print(f"\n    successes: median {np.median(ok_d):6.1f} mm "
              f"(n={len(ok_d)})")
        print(f"    failures : median {np.median(bad_d):6.1f} mm "
              f"(n={len(bad_d)})")
        if np.median(bad_d) > 1.5 * max(np.median(ok_d), 1.0):
            print("    -> failures start measurably further from where the")
            print("       successes start: start pose predicts outcome.")
        else:
            print("    -> no separation. The start pose does NOT explain the")
            print("       difference; the remaining candidates (wrist RGB,")
            print("       pallet position) are visual and cannot be settled")
            print("       from the joint logs alone.")


def block_pairs(labels, rs):
    """Success/failure pairs inside one condition, compared directly."""
    print("\n  2+3. SAME CONDITION, DIFFERENT OUTCOME")
    by_cond = {}
    for name, r in rs.items():
        by_cond.setdefault(labels[name].conditions, []).append(name)

    found = False
    for cond, names in by_cond.items():
        oks = [n for n in names if labels[n].ok]
        bad = [n for n in names if not labels[n].ok]
        if not oks or not bad:
            continue
        found = True
        print(f"\n    {cond}")
        for a in oks:
            for b in bad:
                ra, rb = rs[a], rs[b]
                pa, qa = start_pose(ra)
                pb, qb = start_pose(rb)
                dstart = np.linalg.norm(pa - pb) * 1000
                dang = quat_angle_deg(qa, qb)

                ea, eb = early_path(ra), early_path(rb)
                dearly = float("nan")
                if ea is not None and eb is not None:
                    n = min(len(ea), len(eb))
                    dearly = float(np.linalg.norm(
                        ea[:n] - eb[:n], axis=1).mean() * 1000)

                ga = attempts.best(ra)
                gb = attempts.best(rb)
                dgrasp = (np.linalg.norm(ga.xyz - gb.xyz) * 1000
                          if ga and gb else float("nan"))

                print(f"      {a[-3:]} (OK) vs {b[-3:]} ({labels[b].failure_mode})")
                print(f"        start pose   {dstart:7.1f} mm, "
                      f"{dang:5.2f} deg apart")
                print(f"        first 1 s    {dearly:7.1f} mm apart")
                print(f"        grasp point  {dgrasp:7.1f} mm apart")
                if np.isfinite(dgrasp) and np.isfinite(dstart):
                    if dgrasp > dstart * 1.5:
                        print("        -> diverges: a small start difference")
                        print("           grew into a large grasp difference.")
                    elif dstart > dgrasp * 1.5:
                        print("        -> converges: started apart, grasped")
                        print("           in nearly the same place.")
    if not found:
        print("    no condition has both a success and a failure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=None)
    a = ap.parse_args()

    for ckpt in (a.ckpt or CKPTS):
        print(f"\n{'=' * 74}")
        print(f"{ckpt}")
        print(f"{'=' * 74}")
        labels, rs = load(ckpt)
        if not labels:
            print("  no index.csv")
            continue
        block_start(labels, rs)
        block_pairs(labels, rs)
        block_offset(labels, rs)


if __name__ == "__main__":
    main()
