#!/usr/bin/env python3
"""How far apart are two chunks where they overlap?

The executor merges chunks last-write-wins: a new chunk overwrites the tail
of the previous one for the steps they share. If both predict the same
trajectory the swap is invisible. If they disagree, the commanded stream
jumps by exactly that disagreement, at the chunk boundary and nowhere else.

This measures the disagreement directly: for every pair of consecutive
chunks, the distance between what the OLD chunk predicted for a step and
what the NEW one says for the same step. Reported at the seam (the first
step the new chunk owns), because that is where the discontinuity lands.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.data import POSE_DIM, Rollout  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rollout", nargs="?", default="rollout_000")
ap.add_argument("--ckpt", default="epoch=0040-val_action_mse=9.478029")
ap.add_argument("--n", type=int, default=14, help="how many seams to print")
a = ap.parse_args()

root = pathlib.Path(__file__).resolve().parents[2] / a.ckpt
r = Rollout(root / a.rollout)
if not r.ok:
    raise SystemExit(r.error)

z = np.load(root / a.rollout / "rollout.npz", allow_pickle=True)
acts = np.asarray(z["chunk_actions"], dtype=float)
offs = np.asarray(z["chunk_offsets"], dtype=int)
anch = np.asarray(z["chunk_anchors"], dtype=int)
ct = np.asarray(z["chunk_t"], dtype=float)


def chunk(c):
    """-> (steps, actions) for chunk c."""
    lo, hi = offs[c], offs[c + 1]
    act = acts[lo:hi]
    return anch[c] + np.arange(hi - lo), act


print(f"{a.rollout}: {r.notes}")
print(f"{len(anch)} chunks\n")
print("seam = first step the new chunk owns; disagreement is old vs new "
      "at the SAME step\n")
print(" pair  seam_step  seam_t  overlap |  at seam        | over whole "
      "overlap")
print("                                  |  dpos_mm  drot | mean_mm  "
      "max_mm")
print("-" * 78)

rows = []
for c in range(len(anch) - 1):
    s_old, a_old = chunk(c)
    s_new, a_new = chunk(c + 1)
    common = np.intersect1d(s_old, s_new)
    if len(common) == 0:
        continue
    io = np.searchsorted(s_old, common)
    inn = np.searchsorted(s_new, common)
    dp = np.linalg.norm(a_old[io, :3] - a_new[inn, :3], axis=1) * 1000.0

    seam = int(s_new[0])
    k = int(np.flatnonzero(common == seam)[0])
    q1 = a_old[io[k], 3:7]
    q2 = a_new[inn[k], 3:7]
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    drot = float(np.degrees(2 * np.arccos(
        np.clip(abs(float(np.dot(q1, q2))), -1, 1))))

    rows.append((c, seam, ct[c + 1], len(common), dp[k], drot,
                 dp.mean(), dp.max()))

for c, seam, t, ov, dseam, drot, dmean, dmax in rows[:a.n]:
    print(f"{c:3d}->{c+1:<3d} {seam:8d} {t:7.2f} {ov:8d} | "
          f"{dseam:8.1f} {drot:5.2f} | {dmean:7.1f} {dmax:7.1f}")

if rows:
    arr = np.array([(x[4], x[6], x[7]) for x in rows])
    print(f"\nover all {len(rows)} seams:")
    print(f"  disagreement at seam : mean {arr[:, 0].mean():6.1f} mm   "
          f"max {arr[:, 0].max():6.1f} mm")
    print(f"  disagreement in tail : mean {arr[:, 1].mean():6.1f} mm   "
          f"max {arr[:, 2].max():6.1f} mm")

    # How does a seam step compare with a non-seam step? If seams are not
    # special, the two are equal and the jumps are just the policy's own
    # step size.
    owner = {}
    for c in range(len(offs) - 1):
        s, _ = chunk(c)
        for st in s:
            owner[int(st)] = c
    d = np.linalg.norm(np.diff(r.cmd_pose[:, :3], axis=0), axis=1) * 1000.0
    is_seam = np.array([owner.get(int(r.steps[i + 1]), -1)
                        != owner.get(int(r.steps[i]), -2)
                        for i in range(len(d))])
    print(f"\ncommanded step size, executed stream:")
    print(f"  at a chunk seam  ({is_seam.sum():3d} steps): "
          f"mean {d[is_seam].mean():6.1f} mm   max {d[is_seam].max():6.1f}")
    print(f"  within a chunk   ({(~is_seam).sum():3d} steps): "
          f"mean {d[~is_seam].mean():6.1f} mm   max {d[~is_seam].max():6.1f}")