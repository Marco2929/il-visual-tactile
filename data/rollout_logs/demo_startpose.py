#!/usr/bin/env python3
"""Where do the DEMOS start, and do the rollouts start there too?

The rollout side established (sensitivity.py) that for the ABSOLUTE policy
the start pose predicts the outcome: all four successes are the four runs
closest to the successes' median start pose, failures sit further out with no
overlap. That is consistent with the obvious reading - the absolute policy
conditions on the measured EEF pose, so a start pose it never saw in
training puts it out of distribution immediately, while the relative policy
never receives an absolute pose and cannot be sensitive to it in that way.

That reading makes a checkable prediction, which is the point of this
script: the rollouts that SUCCEED should start inside the demonstrations'
start-pose distribution, and the ones that FAIL should start outside it. If
instead the demos cover everything the rollouts did, the start pose is not
an out-of-distribution problem and the correlation found on the rollout side
needs a different explanation.

The demos hold `state.npy` (6 arm joints in rad, then 20 hand joints), not a
TCP pose, so the arm joints are forward-kinematicked with the same URDF and
tip as the rollout side. Approach taken from
investigations/arm_motion/demo_smoothness.py rather than reinvented.

Usage (inside the container, pinocchio is there):
    python data/rollout_logs/demo_startpose.py \\
        --urdf /ros2_ws/src/moveit_ur_hand_config/urdf/tesollo_hand_ur10e_flange_workcell.urdf
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from viz import labels as lab  # noqa: E402
from viz.data import Rollout  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
DEMO_ROOT = pathlib.Path("/ros2_ws/data/episodes/external")
COLLECTIONS = ["foil_lifting_collection_tesollo_bottom",
               "foil_lifting_collection_tesollo_middle",
               "foil_lifting_collection_tesollo_top_left",
               "foil_lifting_collection_tesollo_top_right"]
CKPTS = ["foil_lifting_wrist_e40", "foil_lifting_relative_e40"]

# Joint order as in the recordings; state.npy is [6 arm | 20 hand].
UR = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_DOF = 6


def make_fk(urdf, tip):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(urdf)
    pd = model.createData()
    qidx = [model.joints[model.getJointId(j)].idx_q for j in UR]
    fid = model.getFrameId(tip)
    q0 = pin.neutral(model)

    def fk(ang):
        q = q0.copy()
        for i, v in zip(qidx, ang):
            q[i] = v
        pin.forwardKinematics(model, pd, q)
        pin.updateFramePlacements(model, pd)
        return np.array(pd.oMf[fid].translation)

    return fk


def demo_starts(fk, max_per=None):
    """-> {collection: (N, 3)} first-frame TCP position of each episode."""
    out = {}
    for coll in COLLECTIONS:
        d = DEMO_ROOT / coll
        if not d.is_dir():
            print(f"  !! missing collection {coll}")
            continue
        pts = []
        eps = sorted(p for p in d.iterdir() if p.name.startswith("episode_"))
        if max_per:
            eps = eps[:max_per]
        for ep in eps:
            f = ep / "state.npy"
            if not f.is_file():
                continue
            try:
                st = np.load(f)
            except ValueError:
                continue
            if st.ndim != 2 or st.shape[0] < 1:
                continue
            pts.append(fk(st[0, :ARM_DOF]))
        if pts:
            out[coll] = np.array(pts)
            print(f"    {coll.replace('foil_lifting_collection_tesollo_', ''):>11s}"
                  f"  {len(pts):3d} episodes")
    return out


def rollout_starts(ckpt):
    """-> [(name, xyz, ok, failure_mode)] first COMMANDED TCP pose."""
    d = ROOT / ckpt
    labels = lab.load(d)
    out = []
    for name in sorted(labels):
        r = Rollout(d / name)
        if not r.ok:
            continue
        out.append((name, np.asarray(r.cmd_pose[0, :3], float),
                    labels[name].ok, labels[name].failure_mode))
    return out


def report(demos, ckpt, rows):
    allp = np.vstack(list(demos.values()))
    c = allp.mean(axis=0)
    rad = np.linalg.norm(allp - c, axis=1)

    print(f"\n  demo start poses: n={len(allp)}")
    print(f"    centroid {c[0]:+.4f} {c[1]:+.4f} {c[2]:+.4f} m")
    print(f"    distance from centroid: median {np.median(rad) * 1000:.1f} mm,"
          f" p95 {np.percentile(rad, 95) * 1000:.1f} mm,"
          f" max {rad.max() * 1000:.1f} mm")
    print("    per collection centroid:")
    for coll, p in demos.items():
        short = coll.replace("foil_lifting_collection_tesollo_", "")
        m = p.mean(axis=0)
        sp = np.linalg.norm(p - m, axis=1) * 1000
        print(f"      {short:>11s} {m[0]:+.4f} {m[1]:+.4f} {m[2]:+.4f}"
              f"   spread {sp.mean():5.1f} mm")

    # Nearest demo start, which is the honest in-distribution question: the
    # demos are four separate pallet positions, so distance to the pooled
    # centroid would punish a rollout for starting near one cluster.
    print(f"\n  {ckpt}")
    print(f"    {'run':>5s} {'verdict':>8s} {'d_near':>8s} {'d_cent':>8s}"
          f"  nearest collection      failure")
    print("    " + "-" * 76)

    near_ok, near_bad = [], []
    for name, p, ok, fm in rows:
        best, bestd = None, 1e9
        for coll, pts in demos.items():
            dd = float(np.linalg.norm(pts - p, axis=1).min())
            if dd < bestd:
                best, bestd = coll, dd
        dcent = float(np.linalg.norm(p - c))
        (near_ok if ok else near_bad).append(bestd * 1000)
        print(f"    {name[-3:]:>5s} {'success' if ok else 'fail':>8s} "
              f"{bestd * 1000:>7.1f}m {dcent * 1000:>7.1f}m  "
              f"{best.replace('foil_lifting_collection_tesollo_', ''):<22s}"
              f"  {fm}")

    if near_ok and near_bad:
        mo, mb = np.median(near_ok), np.median(near_bad)
        print(f"\n    nearest-demo distance: successes median {mo:.1f} mm "
              f"(n={len(near_ok)}), failures {mb:.1f} mm "
              f"(n={len(near_bad)})")
        inside = np.percentile(rad, 95) * 1000
        n_out_ok = sum(1 for d in near_ok if d > inside)
        n_out_bad = sum(1 for d in near_bad if d > inside)
        print(f"    beyond the demos' own p95 spread ({inside:.0f} mm): "
              f"{n_out_ok}/{len(near_ok)} successes, "
              f"{n_out_bad}/{len(near_bad)} failures")
        if mb > 1.5 * max(mo, 1.0):
            print("    -> failures start further from any demonstrated start")
            print("       pose: out-of-distribution start is a real factor.")
        else:
            print("    -> successes and failures are equally far from the")
            print("       demos, so distance to the demo start poses does not")
            print("       by itself explain the outcome.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--tip", default="tool0")
    ap.add_argument("--ckpt", action="append", default=None)
    ap.add_argument("--max-per-collection", type=int, default=None)
    a = ap.parse_args()

    fk = make_fk(a.urdf, a.tip)
    print("demo collections:")
    demos = demo_starts(fk, a.max_per_collection)
    if not demos:
        raise SystemExit("no demo start poses found")

    for ckpt in (a.ckpt or CKPTS):
        rows = rollout_starts(ckpt)
        if rows:
            report(demos, ckpt, rows)


if __name__ == "__main__":
    main()
