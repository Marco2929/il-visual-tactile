#!/usr/bin/env python3
"""What happens in the commanded stream around a given time?

Written for the jump near t=2.5 s in rollout_000. Prints, per control step:
the commanded pose, the step-to-step delta, and WHICH CHUNK the value came
from after last-write-wins. The chunk provenance is the point: a jump that
lines up with a chunk boundary is a merge/anchor problem, one that sits in
the middle of a chunk is the policy itself.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.data import Rollout  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rollout", nargs="?", default="rollout_000")
ap.add_argument("--ckpt", default="epoch=0040-val_action_mse=9.478029")
ap.add_argument("--t0", type=float, default=0.0)
ap.add_argument("--t1", type=float, default=5.0)
a = ap.parse_args()

root = pathlib.Path(__file__).resolve().parents[2] / a.ckpt
r = Rollout(root / a.rollout)
if not r.ok:
    raise SystemExit(r.error)

z = np.load(root / a.rollout / "rollout.npz", allow_pickle=True)
offs = z["chunk_offsets"]
anch = z["chunk_anchors"]
ct = z["chunk_t"]

# Rebuild LWW but remember who won each step.
owner = {}
for c in range(len(offs) - 1):
    lo, hi = int(offs[c]), int(offs[c + 1])
    base = int(anch[c])
    for k in range(hi - lo):
        owner[base + k] = c

print(f"{a.rollout}: {r.notes}")
print(f"{len(anch)} chunks, {len(r.steps)} merged steps\n")

print("first 8 chunks:")
print("  idx  arrival_t   anchor  n_act  anchor-prev_anchor  overlap")
for c in range(min(8, len(anch))):
    n = int(offs[c + 1]) - int(offs[c])
    d = int(anch[c]) - int(anch[c - 1]) if c else 0
    ov = (int(anch[c - 1]) + (int(offs[c]) - int(offs[c - 1]))
          - int(anch[c])) if c else 0
    print(f"  {c:3d}  {ct[c]:9.4f}  {int(anch[c]):7d}  {n:5d}  "
          f"{d:18d}  {ov:7d}")

m = (r.cmd_t >= a.t0) & (r.cmd_t <= a.t1)
idx = np.flatnonzero(m)
print(f"\ncommanded steps in t=[{a.t0}, {a.t1}]  "
      f"({len(idx)} of {len(r.cmd_t)})\n")
print("   t      step  chunk |      x       y       z   |  dpos_mm  "
      "drot_deg | closure")
prev = None
for i in idx:
    st = int(r.steps[i])
    p = r.cmd_pose[i]
    cl = r.cmd_close[i]
    if prev is None:
        dp = dq = 0.0
    else:
        dp = float(np.linalg.norm(p[:3] - prev[:3])) * 1000.0
        q1 = p[3:7] / max(np.linalg.norm(p[3:7]), 1e-12)
        q2 = prev[3:7] / max(np.linalg.norm(prev[3:7]), 1e-12)
        dq = float(np.degrees(2 * np.arccos(
            np.clip(abs(float(np.dot(q1, q2))), -1, 1))))
    mark = "  <-- chunk start" if owner.get(st) != owner.get(st - 1) else ""
    print(f"{r.cmd_t[i]:6.2f} {st:8d} {owner.get(st, -1):6d} | "
          f"{p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f} | {dp:8.1f} {dq:9.2f} | "
          f"{cl:7.1f}{mark}")
    prev = p

# Biggest jumps over the whole run, for context.
d = np.linalg.norm(np.diff(r.cmd_pose[:, :3], axis=0), axis=1) * 1000.0
top = np.argsort(d)[-8:][::-1]
print("\nlargest commanded position steps in the whole rollout:")
for k in top:
    st = int(r.steps[k + 1])
    print(f"  t={r.cmd_t[k + 1]:6.2f}  step={st:6d}  {d[k]:7.1f} mm  "
          f"chunk {owner.get(st, -1)} (prev step chunk "
          f"{owner.get(st - 1, -1)})")