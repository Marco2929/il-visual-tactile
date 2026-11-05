#!/usr/bin/env python3
"""Where is the stutter made: in the command, or in the controller?

With /target_frame now logged there are three stages instead of two, and the
stutter can be attributed instead of inferred:

    cmd_pose  ->  target_frame  ->  js_arm (FK)
    policy        executor out      executed

The commanded stream does NOT stop - longest run of sub-1 mm steps is a
single step over 784 steps (stutter_check.py) - so there is no dwell to
smooth. What it does do is vary by a factor of 4 between p25 and p95. The
controller has no integrator and was measured at v = 0.8 * error, so that
variation should pass straight through to speed.

  SPEED PASSED THROUGH. Coefficient of variation of speed is similar at all
  three stages -> the controller is a faithful follower and the unevenness
  originates in the policy's own step sizes.

  SPEED AMPLIFIED. CV grows from target to executed -> the controller adds
  the pumping, and the fix is feedforward or an integrator, not more gain.

CV is used rather than variance because the stages differ in mean speed.
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
ap.add_argument("--t0", type=float, default=3.0)
a = ap.parse_args()

import pinocchio as pin  # noqa: E402

model = pin.buildModelFromUrdf(a.urdf)
pd = model.createData()
qidx = [model.joints[model.getJointId(j)].idx_q for j in UR]
fid = model.getFrameId(a.tip)
q0 = pin.neutral(model)


def fk(ang):
    q = q0.copy()
    for i, v in zip(qidx, ang):
        q[i] = v
    pin.forwardKinematics(model, pd, q)
    pin.updateFramePlacements(model, pd)
    return np.array(pd.oMf[fid].translation)


def speed(t, xyz, t0):
    """Speed series on a common 20 Hz grid, mm/s, after t0."""
    if len(t) < 3:
        return np.zeros(0)
    grid = np.arange(max(t0, t[0]), t[-1], 1.0 / 20.0)
    if len(grid) < 3:
        return np.zeros(0)
    rs = np.stack([np.interp(grid, t, xyz[:, i]) for i in range(3)], 1)
    return np.linalg.norm(np.diff(rs, axis=0), axis=1) * 1000.0 * 20.0


root = pathlib.Path(__file__).resolve().parents[2] / a.ckpt
print(f"steady state only, t > {a.t0}s. CV = std/mean of speed.\n")
print(f"{'rollout':>9s} {'stage':>14s} {'n':>5s} "
      f"{'mean':>7s} {'p05':>7s} {'p95':>7s} {'CV':>6s}")
print("-" * 60)

rows = {}
for name in a.rollouts:
    r = Rollout(root / name)
    if not r.ok:
        print(f"{name}: {r.error}")
        continue
    if not r.has_target:
        print(f"{name}: no /target_frame logged")
        continue

    stages = [
        ("policy", r.cmd_t, r.cmd_pose[:, :3]),
        ("target_frame", r.tf_t, r.tf_pose[:, :3]),
        ("executed", r.js_t, np.array([fk(p) for p in r.js_arm])),
    ]
    for label, t, xyz in stages:
        v = speed(np.asarray(t), np.asarray(xyz), a.t0)
        if len(v) < 5:
            continue
        cv = float(np.std(v) / max(np.mean(v), 1e-9))
        rows.setdefault(label, []).append(cv)
        print(f"{name[-3:]:>9s} {label:>14s} {len(v):5d} "
              f"{np.mean(v):7.1f} {np.percentile(v, 5):7.1f} "
              f"{np.percentile(v, 95):7.1f} {cv:6.2f}")
    print()

print("mean CV per stage:")
for label in ("policy", "target_frame", "executed"):
    if label in rows:
        print(f"  {label:14s} {np.mean(rows[label]):.2f}")
if "target_frame" in rows and "executed" in rows:
    amp = np.mean(rows["executed"]) / max(np.mean(rows["target_frame"]), 1e-9)
    print(f"\n  executed / target CV = {amp:.2f}")
    print("  ~1.0 -> controller follows faithfully, unevenness is upstream")
    print("  >1.2 -> controller amplifies it")