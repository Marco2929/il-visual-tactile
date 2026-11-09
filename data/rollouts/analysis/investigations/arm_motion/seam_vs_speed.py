#!/usr/bin/env python3
"""Is the jump at a chunk seam bigger than the motion already was?

A previous pass found seam steps averaging 17.0 mm against 8.0 mm within a
chunk and read that as a merge artefact. That comparison is confounded: the
approach phase moves fast AND receives most chunks, so a seam step is drawn
from a faster part of the run. This controls for it three ways.

  1. LOCAL BASELINE. For each seam, compare the jump against the median step
     size in a window just BEFORE it. A merge artefact exceeds the local
     motion; fast motion does not.
  2. PREDICTION RESIDUAL. Extrapolate the last two pre-seam steps linearly
     and measure how far the first post-seam command deviates. Constant-speed
     motion gives ~0 regardless of speed.
  3. ANCHOR STALENESS. /action_buffer_status is published at 10 Hz while the
     executor ticks at 20, so the anchor the producer reads can be 1-2 steps
     old. Reconstructed per chunk from buffer_status.jsonl and correlated
     against the residual.

Only (2) and (3) can distinguish the hypotheses; (1) is reported because it
is what the eye sees in the viewer.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.data import Rollout  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rollouts", nargs="*", default=[])
ap.add_argument("--ckpt", default="epoch=0040-val_action_mse=9.478029")
ap.add_argument("--win", type=int, default=6,
                help="steps of local baseline before each seam")
ap.add_argument("--skip-early", type=float, default=1.5,
                help="ignore seams before this time (approach phase)")
a = ap.parse_args()

root = pathlib.Path(__file__).resolve().parents[2] / a.ckpt
names = a.rollouts or sorted(d.name for d in root.glob("rollout_*")
                             if d.is_dir())

all_rows = []
for name in names:
    d = root / name
    r = Rollout(d)
    if not r.ok:
        continue
    z = np.load(d / "rollout.npz", allow_pickle=True)
    anch = np.asarray(z["chunk_anchors"], dtype=int)
    offs = np.asarray(z["chunk_offsets"], dtype=int)

    # anchor staleness: how old was the status the producer anchored on?
    bs = []
    p = d / "buffer_status.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                o = json.loads(line)
                bs.append((float(o["t"]), int(o["next_step"])))
            except Exception:
                pass
    bs_t = np.array([x[0] for x in bs])
    bs_s = np.array([x[1] for x in bs])

    step_of = {int(s): i for i, s in enumerate(r.steps)}
    pos = r.cmd_pose[:, :3] * 1000.0
    dstep = np.linalg.norm(np.diff(pos, axis=0), axis=1)

    for c in range(1, len(anch)):
        seam = int(anch[c])
        i = step_of.get(seam)
        if i is None or i < a.win + 2 or i >= len(pos) - 1:
            continue
        if r.cmd_t[i] < a.skip_early:
            continue

        local = float(np.median(dstep[i - a.win:i - 1]))
        jump = float(np.linalg.norm(pos[i] - pos[i - 1]))

        # linear extrapolation from the two steps before the seam
        pred = pos[i - 1] + (pos[i - 1] - pos[i - 2])
        resid = float(np.linalg.norm(pos[i] - pred))

        # same residual at non-seam steps nearby, as the null distribution
        nulls = []
        for j in range(i - a.win, i - 1):
            if j < 2:
                continue
            pr = pos[j - 1] + (pos[j - 1] - pos[j - 2])
            nulls.append(float(np.linalg.norm(pos[j] - pr)))
        null = float(np.median(nulls)) if nulls else np.nan

        stale = np.nan
        if len(bs_t):
            k = np.searchsorted(bs_t, r.cmd_t[i]) - 1
            if 0 <= k < len(bs_s):
                stale = float(seam - bs_s[k])

        all_rows.append((name, c, seam, r.cmd_t[i], local, jump,
                         resid, null, stale))

if not all_rows:
    raise SystemExit("no seams matched")

arr = np.array([x[4:] for x in all_rows], dtype=float)
local, jump, resid, null, stale = arr.T

print(f"{len(all_rows)} seams over {len(names)} rollouts, "
      f"t >= {a.skip_early}s\n")

print("1. seam jump vs local motion")
print(f"   local median step before seam : {np.nanmedian(local):6.1f} mm")
print(f"   step AT seam                  : {np.nanmedian(jump):6.1f} mm")
print(f"   ratio (median of per-seam)    : "
      f"{np.nanmedian(jump / np.maximum(local, 1e-6)):6.2f}x")

print("\n2. deviation from constant-velocity extrapolation")
print(f"   at seam     : {np.nanmedian(resid):6.1f} mm")
print(f"   within chunk: {np.nanmedian(null):6.1f} mm")
ratio = np.nanmedian(resid) / max(np.nanmedian(null), 1e-6)
print(f"   ratio       : {ratio:6.2f}x")
print("   (this is speed-independent: constant motion predicts itself)")

ok = ~np.isnan(stale)
if ok.sum() > 3:
    print("\n3. anchor staleness (seam_step - last published next_step)")
    for v in sorted(set(stale[ok].astype(int))):
        sel = ok & (stale == v)
        print(f"   stale={v:+2d} steps  n={sel.sum():3d}  "
              f"residual median {np.nanmedian(resid[sel]):6.1f} mm")
    cc = np.corrcoef(stale[ok], resid[ok])[0, 1]
    print(f"   correlation staleness vs residual: r={cc:+.3f}")

print("\nworst 10 seams by residual:")
for row in sorted(all_rows, key=lambda x: -x[6])[:10]:
    n, c, seam, t, lo, jp, rs, nl, st = row
    print(f"   {n} chunk {c:3d} step {seam:5d} t={t:6.2f}  "
          f"resid {rs:7.1f} mm  local {lo:6.1f}  stale {st:+.0f}")