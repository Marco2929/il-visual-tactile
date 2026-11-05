#!/usr/bin/env python3
"""The opening transient, resolved finely, plus a steady-state comparison.

Two questions, and the t<1.5s bucket in damping_check.py answers neither:

  1. The initial lunge is the worst part and has survived every controller
     change so far. Where does its target actually come from? The policy's
     very first chunk is produced from one observation with n_obs_steps=2,
     and the arm starts whereever the previous run left it - so the first
     commanded pose can be a long way from the current one, and the
     controller will chase it at full gain.
  2. Is the steady state after the lunge really faster, or does it only feel
     that way because it is now jerkier?

So: per-100ms slices for the first two seconds, then the settled phase.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.data import Rollout  # noqa: E402

UR = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

ap = argparse.ArgumentParser()
ap.add_argument("rollouts", nargs="+")
ap.add_argument("--ckpt", default="epoch=0040-val_action_mse=9.478029")
ap.add_argument("--urdf", required=True)
ap.add_argument("--tip", default="tool0")
a = ap.parse_args()

import pinocchio as pin  # noqa: E402

model = pin.buildModelFromUrdf(a.urdf)
pdata = model.createData()
qidx = [model.joints[model.getJointId(j)].idx_q for j in UR]
fid = model.getFrameId(a.tip)
q0 = pin.neutral(model)


def fk(ang):
    q = q0.copy()
    for i, v in zip(qidx, ang):
        q[i] = v
    pin.forwardKinematics(model, pdata, q)
    pin.updateFramePlacements(model, pdata)
    return np.array(pdata.oMf[fid].translation)


def smooth(x, n=5):
    k = np.ones(n) / n
    return np.stack([np.convolve(x[:, i], k, "same")
                     for i in range(x.shape[1])], 1)


root = pathlib.Path(__file__).resolve().parents[2] / a.ckpt

for name in a.rollouts:
    r = Rollout(root / name)
    if not r.ok:
        print(f"{name}: {r.error}")
        continue

    xyz = smooth(np.array([fk(p) for p in r.js_arm]))
    t = r.js_t - r.js_t[0]
    dt = np.diff(t)
    g = dt > 1e-6
    v = np.zeros(len(t) - 1)
    v[g] = np.linalg.norm(np.diff(xyz, axis=0)[g], axis=1) / dt[g] * 1000
    tv = 0.5 * (t[1:] + t[:-1])
    av = np.zeros(len(v) - 1)
    dtv = np.diff(tv)
    g2 = dtv > 1e-6
    av[g2] = np.abs(np.diff(v)[g2] / dtv[g2])

    ct = r.cmd_t - r.cmd_t[0]
    j = np.clip(np.searchsorted(ct, tv) - 1, 0, len(ct) - 1)
    err = np.linalg.norm(r.cmd_pose[j, :3] - xyz[:-1], axis=1) * 1000

    # distance from where the arm stood to the first commanded pose
    d0 = np.linalg.norm(r.cmd_pose[0, :3] - xyz[0]) * 1000

    print(f"\n{'=' * 70}")
    print(f"{name}   {r.notes}")
    print(f"{'=' * 70}")
    print(f"first commanded pose is {d0:.0f} mm from the arm's start pose")
    print(f"  arm at start      : {np.round(xyz[0], 3)}")
    print(f"  first commanded   : {np.round(r.cmd_pose[0, :3], 3)}")

    print("\n  t [s]        err mm   speed mm/s   |a| mm/s^2")
    print("  " + "-" * 48)
    for lo in np.arange(0.0, 2.0, 0.1):
        s = (tv >= lo) & (tv < lo + 0.1)
        if s.sum() < 2:
            continue
        sa = s[:len(av)]
        print(f"  {lo:.1f}-{lo + 0.1:.1f}   {np.median(err[s]):8.1f} "
              f"{np.median(v[s]):10.1f} "
              f"{np.median(av[sa]) if sa.sum() else float('nan'):11.0f}")

    for lab, lo, hi in (("settled 3-6s", 3.0, 6.0),
                        ("settled 6s+", 6.0, 1e9)):
        s = (tv >= lo) & (tv < hi)
        if s.sum() < 10:
            continue
        sa = s[:len(av)]
        print(f"\n  {lab}: err {np.median(err[s]):6.1f} mm   "
              f"speed {np.median(v[s]):6.1f} mm/s   "
              f"|a| p50 {np.median(av[sa]):6.0f}  "
              f"p95 {np.percentile(av[sa], 95):7.0f}")