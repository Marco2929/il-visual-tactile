#!/usr/bin/env python3
"""Does chunk composition explain the 3x jerkiness over the demos?

Established so far: demos have speed CV 0.31 (n=160, consistent across four
collections), the policy's commanded stream has 0.94, and the controller only
ever reduces CV downstream. So the roughness is created between the policy's
output and the merged stream - and the one thing that stream has which the
demos do not is chunking.

The suspicion is overlap density. min_execution_horizon=20 refills whenever
buffer < delay+20; with 26-action chunks that means a new chunk every ~8
steps and 2-3 chunks covering any given step. LWW keeps the newest, so each
step's value can be rewritten twice before it executes, and consecutive steps
may come from different chunks even far from a seam.

Tests, in order of how directly they implicate that:
  1. How many chunks cover each step, and how often the winning chunk
     changes between consecutive steps.
  2. CV computed WITHIN one chunk (single-chunk motion, no merging) against
     CV of the merged stream. If a lone chunk is as smooth as a demo, the
     merge is the culprit and the policy itself is fine.
  3. Disagreement between chunks over the same step - the raw material LWW
     switches between.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.data import Rollout  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rollouts", nargs="+")
ap.add_argument("--ckpt", default="epoch=0040-val_action_mse=9.478029")
ap.add_argument("--t0", type=float, default=3.0)
a = ap.parse_args()

root = pathlib.Path(__file__).resolve().parents[2] / a.ckpt


def cv(xyz):
    if len(xyz) < 6:
        return None
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000.0 * 20.0
    return float(np.std(v) / max(np.mean(v), 1e-9))


merged_cv, inchunk_cv, switch_frac, cover = [], [], [], []

for name in a.rollouts:
    r = Rollout(root / name)
    if not r.ok:
        print(f"{name}: {r.error}")
        continue

    z = np.load(root / name / "rollout.npz", allow_pickle=True)
    acts = z["chunk_actions"]
    offs = np.asarray(z["chunk_offsets"], dtype=int)
    anch = np.asarray(z["chunk_anchors"], dtype=int)
    n_ch = len(anch)

    # Which chunk wins each absolute step, and how many cover it.
    owner, count = {}, {}
    for c in range(n_ch):
        lo, hi = offs[c], offs[c + 1] if c + 1 < len(offs) else len(acts)
        for k in range(hi - lo):
            s = int(anch[c]) + k
            owner[s] = c                      # later chunk overwrites: LWW
            count[s] = count.get(s, 0) + 1

    steps = np.array(sorted(owner))
    if len(steps) < 10:
        continue
    t_of = dict(zip(r.steps, r.cmd_t))
    keep = np.array([t_of.get(int(s), -1.0) >= a.t0 for s in steps])
    steps = steps[keep]
    if len(steps) < 10:
        continue

    own = np.array([owner[int(s)] for s in steps])
    cnt = np.array([count[int(s)] for s in steps])
    sw = float(np.mean(own[1:] != own[:-1]))

    m = cv(r.cmd_pose[:, :3][np.isin(r.steps, steps)])

    # Motion inside a single chunk: no merging involved at all.
    singles = []
    for c in range(n_ch):
        lo, hi = offs[c], offs[c + 1] if c + 1 < len(offs) else len(acts)
        if int(anch[c]) not in t_of or t_of[int(anch[c])] < a.t0:
            continue
        s = cv(acts[lo:hi, :3])
        if s is not None:
            singles.append(s)

    # How far apart do two chunks put the same step?
    dis = []
    for c in range(n_ch - 1):
        lo, hi = offs[c], offs[c + 1] if c + 1 < len(offs) else len(acts)
        lo2, hi2 = offs[c + 1], (offs[c + 2] if c + 2 < len(offs)
                                 else len(acts))
        s0, s1 = int(anch[c]), int(anch[c + 1])
        for k in range(hi2 - lo2):
            step = s1 + k
            j = step - s0
            if 0 <= j < hi - lo:
                d = np.linalg.norm(acts[lo + j, :3] - acts[lo2 + k, :3])
                dis.append(d * 1000.0)

    print(f"\n{name}: {n_ch} chunks, {len(steps)} steps after t={a.t0}s")
    print(f"  chunks covering a step : p50 {np.median(cnt):.0f}  "
          f"max {cnt.max():.0f}")
    print(f"  winning chunk switches : {100 * sw:.1f}% of steps")
    print(f"  CV merged stream       : {m:.2f}" if m else "")
    if singles:
        print(f"  CV within one chunk    : {np.median(singles):.2f} "
              f"(n={len(singles)})")
    if dis:
        print(f"  chunk disagreement     : p50 {np.median(dis):.1f} mm  "
              f"p95 {np.percentile(dis, 95):.1f} mm")

    if m:
        merged_cv.append(m)
    if singles:
        inchunk_cv.append(np.median(singles))
    switch_frac.append(sw)
    cover.append(np.median(cnt))

if merged_cv and inchunk_cv:
    print(f"\n{'=' * 58}")
    print(f"  demos (measured, n=160)   CV 0.31")
    print(f"  within a single chunk     CV {np.median(inchunk_cv):.2f}")
    print(f"  merged stream             CV {np.median(merged_cv):.2f}")
    print(f"  steps whose owner changed    {100 * np.mean(switch_frac):.1f}%")
    print(f"  chunks per step (median)     {np.median(cover):.0f}")
    print("\n  single-chunk CV near 0.31 -> the policy is smooth and the")
    print("  merge creates the roughness; near 0.94 -> the policy itself")
    print("  predicts uneven motion and chunking is not to blame.")