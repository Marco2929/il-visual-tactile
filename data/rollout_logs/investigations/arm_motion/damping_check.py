#!/usr/bin/env python3
"""Is the arm's slowness velocity damping, or a limit?

The two hypotheses make different predictions, so this is decidable from the
logs alone.

  DAMPING. ForwardDynamicsSolver integrates a virtual mass and then does
      m_current_velocities *= 0.9
  once per solver iteration, i.e. 0.9^10 = 0.35 per control cycle with
  iterations: 10. The fixed point of v <- 0.9 * (v + a * 0.02) is
  v = 0.18 * a, and a is proportional to the pose error. So executed speed
  must be PROPORTIONAL TO THE TRACKING ERROR, a straight line through the
  origin, with no ceiling.

  LIMIT. A saturating stage (velocity clamp, driver cap) gives executed speed
  that RISES AND THEN FLATTENS at a fixed value, independent of error.

The discriminator is the shape of speed-vs-error, not its slope. Binned here
rather than fitted, because a correlation coefficient cannot tell a line from
a saturation curve.

Executed TCP comes from forward kinematics of the arm joints; joint_states
has no TCP pose. Positions are smoothed before differentiating: raw diffs of
a 100-500 Hz joint stream gave 4.2 m/s spikes earlier, which are
differentiation noise, not motion.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.data import Rollout  # noqa: E402

UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

ap = argparse.ArgumentParser()
ap.add_argument("rollouts", nargs="*")
ap.add_argument("--ckpt", default="epoch=0040-val_action_mse=9.478029")
ap.add_argument("--urdf", required=True)
ap.add_argument("--tip", default="tool0")
ap.add_argument("--smooth", type=int, default=5,
                help="moving-average samples before differentiating")
a = ap.parse_args()

import pinocchio as pin  # noqa: E402

model = pin.buildModelFromUrdf(a.urdf)
pdata = model.createData()
qidx = [model.joints[model.getJointId(j)].idx_q for j in UR_JOINTS]
fid = model.getFrameId(a.tip)
q0 = pin.neutral(model)


def fk(angles):
    q = q0.copy()
    for i, v in zip(qidx, angles):
        q[i] = v
    pin.forwardKinematics(model, pdata, q)
    pin.updateFramePlacements(model, pdata)
    return np.array(pdata.oMf[fid].translation)


def movavg(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.stack([np.convolve(x[:, i], k, mode="same")
                     for i in range(x.shape[1])], axis=1)


root = pathlib.Path(__file__).resolve().parents[2] / a.ckpt
names = a.rollouts or sorted(d.name for d in root.glob("rollout_*")
                             if d.is_dir())[:8]

err_all, spd_all, acc_e_all, acc_c_all, t_all = [], [], [], [], []

for name in names:
    r = Rollout(root / name)
    if not r.ok:
        continue

    xyz = movavg(np.array([fk(p) for p in r.js_arm]), a.smooth)
    t = r.js_t
    dt = np.diff(t)
    good = dt > 1e-6
    v = np.zeros(len(t) - 1)
    v[good] = np.linalg.norm(np.diff(xyz, axis=0)[good], axis=1) / dt[good]
    tv = 0.5 * (t[1:] + t[:-1])

    # the target that was active at each js sample
    j = np.searchsorted(r.cmd_t, tv) - 1
    ok = j >= 0
    err = np.full(len(tv), np.nan)
    err[ok] = np.linalg.norm(r.cmd_pose[j[ok], :3] - xyz[:-1][ok], axis=1)

    dtv = np.diff(tv)
    ae = np.zeros(len(v) - 1)
    g2 = dtv > 1e-6
    ae[g2] = np.diff(v)[g2] / dtv[g2]

    dcp = np.diff(r.cmd_pose[:, :3], axis=0)
    dct = np.diff(r.cmd_t)
    gc = dct > 1e-6
    vc = np.zeros(len(dct))
    vc[gc] = np.linalg.norm(dcp[gc], axis=1) / dct[gc]
    ac = np.zeros(len(vc) - 1)
    d2 = np.diff(0.5 * (r.cmd_t[1:] + r.cmd_t[:-1]))
    g3 = d2 > 1e-6
    ac[g3] = np.diff(vc)[g3] / d2[g3]

    err_all.append(err[:-1])
    spd_all.append(v[:-1])
    acc_e_all.append(ae)
    acc_c_all.append(ac)
    t_all.append(tv[:-1] - tv[0])

err = np.concatenate(err_all) * 1000.0
spd = np.concatenate(spd_all) * 1000.0
tt = np.concatenate(t_all)
m = ~np.isnan(err) & (err < 500)
err, spd, tt = err[m], spd[m], tt[m]

print(f"{len(names)} rollouts, {len(err)} samples, "
      f"smoothing {a.smooth} samples\n")

print("TEST: executed speed as a function of tracking error")
print("  damping -> straight line through origin, no ceiling")
print("  limit   -> rises then flattens\n")
print("  error bin mm      n    exec speed mm/s      speed/error")
print("                        median      p90        ratio")
print("-" * 62)
edges = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100, 200]
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (err >= lo) & (err < hi)
    if sel.sum() < 30:
        continue
    e, s = err[sel], spd[sel]
    print(f"  {lo:4d} - {hi:4d}  {sel.sum():7d}  {np.median(s):8.1f} "
          f"{np.percentile(s, 90):8.1f}   "
          f"{np.median(s) / max(np.median(e), 1e-9):8.2f}")

lo_m = (err > 2) & (err < 10)
hi_m = (err > 30) & (err < 120)
if lo_m.sum() > 30 and hi_m.sum() > 30:
    rl = np.median(spd[lo_m]) / np.median(err[lo_m])
    rh = np.median(spd[hi_m]) / np.median(err[hi_m])
    print(f"\n  slope at small error (2-10 mm) : {rl:6.2f} 1/s")
    print(f"  slope at large error (30-120mm): {rh:6.2f} 1/s")
    print(f"  ratio large/small              : {rh / max(rl, 1e-9):6.2f}")
    print("  (damping predicts ~1.0; a limit predicts << 1.0)")

ae = np.concatenate(acc_e_all) * 1000.0
ac = np.concatenate(acc_c_all) * 1000.0
n = min(len(ae), len(ac))
print(f"\nacceleration magnitude, |a| mm/s^2 (p50 / p95)")
print(f"  commanded: {np.median(np.abs(ac)):9.0f} "
      f"{np.percentile(np.abs(ac), 95):9.0f}")
print(f"  executed : {np.median(np.abs(ae)):9.0f} "
      f"{np.percentile(np.abs(ae), 95):9.0f}")

early = tt < 1.5
late = tt > 3.0
if early.sum() > 30 and late.sum() > 30:
    print(f"\nby phase (snap vs steady):")
    print(f"  t < 1.5s : error {np.median(err[early]):7.1f} mm  "
          f"speed {np.median(spd[early]):7.1f} mm/s")
    print(f"  t > 3.0s : error {np.median(err[late]):7.1f} mm  "
          f"speed {np.median(spd[late]):7.1f} mm/s")