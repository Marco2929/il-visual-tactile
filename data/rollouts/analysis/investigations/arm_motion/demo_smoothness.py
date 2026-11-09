#!/usr/bin/env python3
"""Was the stutter demonstrated, or is it introduced at deployment?

pump_check.py attributed it upstream of the controller: speed CV is 0.81 in
the policy output, 0.70 at /target_frame, 0.40 executed - the controller
halves the unevenness rather than adding any. That leaves the policy, and a
policy that imitates faithfully is only as smooth as what it was shown.

The training episodes hold state.npy (6 arm joints in rad + 20 hand joints in
deg) at the same 20 Hz the policy runs at, so the identical metric applies:
forward-kinematics the arm joints to a TCP path, differentiate, take
std/mean of speed. Same URDF, same tip, same steady-state cut as the
rollouts, so the two numbers are comparable.

  demos CV ~= 0.8  -> the jerkiness is in the demonstrations and was learned.
                      No controller tuning can remove it; the fix is upstream
                      (smoother teleop, or resampling before training).
  demos CV << 0.8  -> the demos are smooth and something in the deployment
                      path roughens them - chunking being the obvious
                      suspect, since that is the one thing the demos lack.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.data import Rollout  # noqa: E402

UR = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_DOF = 6

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="/ros2_ws/data/episodes/external")
ap.add_argument("--collections", nargs="*", default=[
    "foil_lifting_collection_tesollo_middle",
    "foil_lifting_collection_tesollo_bottom",
    "foil_lifting_collection_tesollo_top_left",
    "foil_lifting_collection_tesollo_top_right",
])
ap.add_argument("--rollouts", nargs="*",
                default=["rollout_048", "rollout_049", "rollout_050"])
ap.add_argument("--ckpt", default="epoch=0040-val_action_mse=9.478029")
ap.add_argument("--urdf", required=True)
ap.add_argument("--tip", default="tool0")
ap.add_argument("--t0", type=float, default=3.0)
ap.add_argument("--max-per-collection", type=int, default=40)
a = ap.parse_args()

import pinocchio as pin  # noqa: E402

model = pin.buildModelFromUrdf(a.urdf)
pd = model.createData()
qidx = [model.joints[model.getJointId(j)].idx_q for j in UR]
fid = model.getFrameId(a.tip)
q0 = pin.neutral(model)


def fk_path(arm_rad):
    out = np.empty((len(arm_rad), 3))
    for k, ang in enumerate(arm_rad):
        q = q0.copy()
        for i, v in zip(qidx, ang):
            q[i] = v
        pin.forwardKinematics(model, pd, q)
        pin.updateFramePlacements(model, pd)
        out[k] = pd.oMf[fid].translation
    return out


def cv_of_speed(t_s, xyz, t0):
    m = t_s >= t0
    if m.sum() < 10:
        return None, None, None
    t_s, xyz = t_s[m], xyz[m]
    dt = np.diff(t_s)
    ok = dt > 1e-6
    v = np.linalg.norm(np.diff(xyz, axis=0)[ok], axis=1) / dt[ok] * 1000.0
    if len(v) < 10:
        return None, None, None
    return float(np.std(v) / max(np.mean(v), 1e-9)), float(np.mean(v)), len(v)


print(f"speed CV, steady state t > {a.t0}s, same URDF and tip as the "
      f"rollout analysis\n")

root = pathlib.Path(a.episodes)
all_cv, all_mean = [], []
for coll in a.collections:
    cdir = root / coll
    if not cdir.is_dir():
        print(f"  {coll}: not found")
        continue
    eps = sorted(d for d in cdir.iterdir()
                 if d.is_dir() and d.name.startswith("episode_"))
    eps = eps[:a.max_per_collection]
    cvs, means = [], []
    for ep in eps:
        sp, tp = ep / "state.npy", ep / "frame_timestamps_ns.npy"
        if not (sp.is_file() and tp.is_file()):
            continue
        try:
            state = np.load(sp)
            ts = np.load(tp).astype(np.float64)
        except Exception:
            continue
        if state.ndim != 2 or state.shape[1] < ARM_DOF or len(state) < 20:
            continue
        t_s = (ts - ts[0]) / 1e9
        cv, mn, _ = cv_of_speed(t_s, fk_path(state[:, :ARM_DOF]), a.t0)
        if cv is not None:
            cvs.append(cv)
            means.append(mn)
    if cvs:
        print(f"  {coll:46s} n={len(cvs):3d}  "
              f"CV median {np.median(cvs):5.2f}  "
              f"mean speed {np.median(means):6.1f} mm/s")
        all_cv += cvs
        all_mean += means

if not all_cv:
    raise SystemExit("no episodes read")

print(f"\n  DEMOS  n={len(all_cv)}  CV median {np.median(all_cv):.2f}  "
      f"p25 {np.percentile(all_cv, 25):.2f}  "
      f"p75 {np.percentile(all_cv, 75):.2f}")
print(f"         median speed {np.median(all_mean):.1f} mm/s")

ckpt = pathlib.Path(__file__).resolve().parents[2] / a.ckpt
rcv, rmean = [], []
for name in a.rollouts:
    r = Rollout(ckpt / name)
    if not r.ok:
        continue
    cv, mn, _ = cv_of_speed(np.asarray(r.cmd_t) - r.cmd_t[0],
                            np.asarray(r.cmd_pose[:, :3]), a.t0)
    if cv is not None:
        rcv.append(cv)
        rmean.append(mn)

if rcv:
    print(f"\n  POLICY n={len(rcv)}  CV median {np.median(rcv):.2f}")
    print(f"         median speed {np.median(rmean):.1f} mm/s")
    print(f"\n  ratio policy/demos = "
          f"{np.median(rcv) / max(np.median(all_cv), 1e-9):.2f}")
    print("  ~1.0 -> learned from the demos, not a deployment artefact")
    print("  >1.5 -> the deployment path adds it")