#!/usr/bin/env python3
"""Split the closure deficit into filter loss and mechanical loss.

The question: the hand executes LESS than commanded, and the later grasp in
rollout_010 shows a smaller gap than the first. Two causes are possible and
they have different fixes:

  filter loss      what control_ur_tesollo_cartesian.py removes before the
                   goal ever reaches the hardware (Butterworth 3 Hz, 0.3 deg
                   deadband, 6 deg/tick clamp). Replayed with viz.filters,
                   so it is the deployed arithmetic, not an estimate.
  mechanical loss  what the hardware does not reach although it was asked:
                   the finger presses on something and the position
                   controller stalls short of its goal.

Reported PER ATTEMPT, because an average over a run with two grasps hides
exactly the difference being asked about.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz import filters, pinch  # noqa: E402
from viz.data import GRASP_LOCAL, Rollout  # noqa: E402

root = pathlib.Path(__file__).resolve().parents[2] / "foil_lifting_wrist_e40"
if not root.exists():
    root = pathlib.Path(__file__).resolve().parents[2] / "epoch=0040-val_action_mse=9.478029"
name = sys.argv[1] if len(sys.argv) > 1 else "rollout_010"
r = Rollout(root / name)
if not r.ok:
    raise SystemExit(r.error)

pm, note = pinch.try_load()
print(f"{name}: {r.notes}")
print(f"  {note}\n")

filt, fstats = filters.run_hand_chain(r.cmd_hand)
cmd_close = r.cmd_hand[:, GRASP_LOCAL].sum(axis=1)
filt_close = filt[:, GRASP_LOCAL].sum(axis=1)
exec_on_cmd = np.interp(r.cmd_t, r.js_t, r.exec_close)

c = cmd_close
thresh = c.min() + 0.45 * (c.max() - c.min())
above = c > thresh
edges = np.diff(above.astype(int))
starts = list(np.flatnonzero(edges == 1) + 1)
ends = list(np.flatnonzero(edges == -1) + 1)
if above[0]:
    starts = [0] + starts
if above[-1]:
    ends = ends + [len(c)]

print(f"{len(starts)} grasp attempt(s) above {thresh:.1f} deg\n")
print("attempt  t_peak  commanded  after-filter  executed  "
      "filter_loss  mech_loss")
print("-" * 76)

for i, (a, b) in enumerate(zip(starts, ends), start=1):
    k = a + int(np.argmax(c[a:b]))
    cmd_pk, fil_pk, exe_pk = cmd_close[k], filt_close[k], exec_on_cmd[k]
    print(f"  {i}    {r.cmd_t[k]:7.2f} {cmd_pk:10.1f} {fil_pk:13.1f} "
          f"{exe_pk:9.1f} {cmd_pk - fil_pk:12.1f} {fil_pk - exe_pk:10.1f}")

    if pm is not None:
        j = int(np.argmin(np.abs(r.js_t - r.cmd_t[k])))
        print(f"       pinch mm {pm.width_mm(r.cmd_hand[k]):10.1f} "
              f"{pm.width_mm(filt[k]):13.1f} "
              f"{pm.width_mm(r.exec_hand[j]):9.1f}")

print(f"\nfilter stage counts over the whole run: {fstats}")
tot = len(c) * r.cmd_hand.shape[1]
print(f"  deadband hits: {fstats['deadbanded'] / tot:.1%} of joint-ticks")
print(f"  clamp hits   : {fstats['clamped'] / tot:.1%} of joint-ticks")