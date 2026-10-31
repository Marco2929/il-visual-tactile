#!/usr/bin/env python3
"""Do the missed grasps approach at a different ANGLE than the demos?

Open point from the relative findings (section 6): "Anfahrwinkel der
verfehlten Griffe gegen die Demos stellen", with the rig observation
"Tendenz, zu steil zu greifen" behind it (section 2). Everything measured so
far used positions only; orientation has been ignored completely, so a
systematically wrong wrist angle would not have shown up anywhere.

The reference is the DEMONSTRATED grasp orientation. The demos do not label
when the grasp happens, so it is located the same way as on the rollout side
- by the hand closing. The pinch joints are in the state vector (columns
7:15, degrees), and the frame where their sum peaks is the grasp.

Reported per rollout, split by verdict:

  TILT       angle between the tool's approach axis and vertical. This is
             the "too steep" claim in a single number.
  FULL       total rotation angle to the nearest demonstrated grasp
             orientation - catches a wrist that is rotated about the
             approach axis, which TILT cannot see.
  DESCENT    direction of travel over the last 300 ms before closing,
             against vertical. A policy can hold the right tool angle and
             still arrive sideways.

Why both TILT and FULL: tilt alone treats a 90 deg wrist roll as perfect,
and the full angle alone cannot distinguish "steep" from "rolled". The two
together localise which part of the orientation is off.

Usage (in the container):
    python data/rollout_logs/approach_angle.py
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, "/ros2_ws/third_party/ManiFlow_Policy/ManiFlow")

from viz import attempts, labels as lab  # noqa: E402
from viz.data import Rollout  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
ZARR = "/ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr"
CKPTS = ["foil_lifting_wrist_e40", "foil_lifting_relative_e40"]

# Columns 7:15 of the state vector are the pinch joints in degrees. Same
# definition of "closure" as viz/data.py uses for the rollouts, so the demo
# grasp instant and the rollout grasp instant are found the same way.
HAND_LO, HAND_HI = 7, 15


def quat_to_matrix(q):
    from maniflow.dataset.relative_pose import quat_to_matrix as f
    return f(q)


def approach_axis(R):
    """Tool z-axis: the direction the hand points.

    Third column of the rotation matrix. For a gripper reaching down at a
    pallet this is close to -world_z, so the angle against DOWN is the
    quantity the 'too steep' observation refers to.
    """
    return R[..., :, 2]


def tilt_deg(R):
    """Angle between the tool axis and straight down, in degrees."""
    a = approach_axis(np.asarray(R, dtype=np.float64))
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    down = np.array([0.0, 0.0, -1.0])
    return np.degrees(np.arccos(np.clip(a @ down, -1.0, 1.0)))


def rot_angle_deg(R1, R2):
    """Total rotation angle between two orientations."""
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    M = np.einsum("...ij,...jk->...ik", np.swapaxes(R1, -1, -2), R2)
    tr = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def demo_grasps():
    """-> (R_grasp, tilt, descent_tilt) at the demonstrated grasp instants.

    Episodes are delimited by the replay buffer's episode_ends, and the grasp
    is the frame where the pinch sum peaks within each episode.
    """
    from maniflow.common.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.copy_from_path(ZARR, keys=["state"])
    st = np.asarray(rb["state"], dtype=np.float64)
    ends = np.asarray(rb.episode_ends, dtype=int)

    Rs, tilts, desc = [], [], []
    start = 0
    for end in ends:
        ep = st[start:end]
        start = end
        if len(ep) < 12:
            continue
        close = ep[:, HAND_LO:HAND_HI].sum(axis=1)
        k = int(np.argmax(close))
        R = quat_to_matrix(ep[k, 3:7])
        Rs.append(R)
        tilts.append(float(tilt_deg(R)))

        # Direction of travel into the grasp: 6 frames back at 20 Hz = 300 ms.
        j = max(0, k - 6)
        d = ep[k, 0:3] - ep[j, 0:3]
        n = np.linalg.norm(d)
        if n > 1e-4:
            desc.append(float(np.degrees(np.arccos(
                np.clip(d / n @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0)))))

    return np.array(Rs), np.array(tilts), np.array(desc)


def rollout_grasps(ckpt):
    """-> [(name, ok, mode, R, tilt, descent)] at the best attempt."""
    d = ROOT / ckpt
    labels = lab.load(d)
    out = []
    for name in sorted(labels):
        r = Rollout(d / name)
        if not r.ok:
            continue
        b = attempts.best(r)
        if b is None:
            continue
        q = np.asarray(r.cmd_pose[b.peak, 3:7], dtype=np.float64)
        R = quat_to_matrix(q)

        t = np.asarray(r.cmd_t, dtype=np.float64)
        j = int(np.argmin(np.abs(t - (t[b.peak] - 0.3))))
        dvec = np.asarray(r.cmd_pose[b.peak, :3], float) - np.asarray(
            r.cmd_pose[j, :3], float)
        n = np.linalg.norm(dvec)
        desc = (float(np.degrees(np.arccos(np.clip(
            dvec / n @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))))
            if n > 1e-4 else float("nan"))

        out.append((name, labels[name].ok, labels[name].failure_mode,
                    R, float(tilt_deg(R)), desc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=None)
    a = ap.parse_args()

    print("=== approach angle: rollouts vs. demonstrations ===")
    print("grasp instant = peak of the pinch-joint sum, same rule on both")
    print("sides. Dataset is shared; the dataloader relativises.")

    R_demo, tilt_demo, desc_demo = demo_grasps()
    print(f"\n  demo grasps: n={len(R_demo)}")
    print(f"    tool tilt from vertical: median {np.median(tilt_demo):5.1f} "
          f"deg, p5 {np.percentile(tilt_demo, 5):5.1f}, "
          f"p95 {np.percentile(tilt_demo, 95):5.1f}")
    if len(desc_demo):
        print(f"    descent direction      : median "
              f"{np.median(desc_demo):5.1f} deg, "
              f"p95 {np.percentile(desc_demo, 95):5.1f}")
    print("    (0 deg = straight down; larger = more sideways/steeper)")

    for ckpt in (a.ckpt or CKPTS):
        rows = rollout_grasps(ckpt)
        if not rows:
            continue
        print(f"\n  {ckpt}")
        print(f"    {'run':>5s} {'ok':>3s} {'tilt':>7s} {'d_tilt':>8s} "
              f"{'to_demo':>8s} {'descent':>8s}  failure")
        print("    " + "-" * 66)

        agg = {True: [], False: []}
        for name, ok, mode, R, tl, desc in rows:
            # Nearest demonstrated orientation, not the mean: the demos cover
            # four pallet positions and their mean is not a pose anyone
            # actually demonstrated.
            full = float(rot_angle_deg(R_demo, R[None, ...]).min())
            dt = tl - float(np.median(tilt_demo))
            agg[ok].append((tl, dt, full, desc))
            print(f"    {name[-3:]:>5s} {'OK' if ok else '--':>3s} "
                  f"{tl:>6.1f}d {dt:>+7.1f}d {full:>7.1f}d "
                  f"{desc:>7.1f}d  {mode}")

        for ok in (True, False):
            if not agg[ok]:
                continue
            arr = np.array([[x[0], x[1], x[2],
                             x[3] if np.isfinite(x[3]) else np.nan]
                            for x in agg[ok]])
            lbl = "successes" if ok else "failures "
            print(f"    {lbl}: tilt {np.median(arr[:, 0]):5.1f}d "
                  f"(demo {np.median(arr[:, 1]):+5.1f}d), "
                  f"to_demo {np.median(arr[:, 2]):5.1f}d, "
                  f"descent {np.nanmedian(arr[:, 3]):5.1f}d "
                  f"(n={len(arr)})")

        if agg[True] and agg[False]:
            o = np.median([x[0] for x in agg[True]])
            f = np.median([x[0] for x in agg[False]])
            print(f"    tilt difference failure minus success: "
                  f"{f - o:+.1f} deg")


if __name__ == "__main__":
    main()
