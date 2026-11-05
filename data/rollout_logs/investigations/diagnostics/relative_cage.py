#!/usr/bin/env python3
"""Numbers for the relative-policy cage (relative findings.md section 7.3).

The gate in section 7.1 is GREEN: maniflow/dataset/tests/
test_deploy_relative_roundtrip.py feeds recorded relative actions and the
recorded base through deploy_relative.py's own encode/decode and gets the
recorded absolute poses back to < 1e-5 m, with a deliberate one-frame base
error failing by >100x so the tolerance is known to be sensitive. The
transform is therefore not the explanation, and section 7.2 does not apply.

That leaves the behaviour itself, which section 7.3 proposes to bound. All
three items there need numbers that nobody has measured yet, and this script
produces them:

  1. DELTA CLAMP      per-step ||T_rel[j+1] - T_rel[j]|| and rotation angle,
                      99.9th percentile over the training windows. Anything
                      beyond that is a step the demonstrations never took.
  2. WORKSPACE BOX    min/max of the absolute training poses. The decoded
                      target can be checked against this before it is sent.
  3. OOD DISTANCE     for every rollout, the distance from each commanded
                      pose to the NEAREST training pose, over time. This is
                      the direct test of the accumulation hypothesis
                      (section 9.4): if the relative policy drifts, the
                      distance must RISE during the run, and it must rise
                      more than the absolute policy's does.

The third one is the point. A clean decoder plus a rising OOD distance means
the policy is walking out of its training distribution on its own, which is
what "die Policy summiert auf" predicts and what a clamp would stop.

Usage (in the container):
    python data/rollout_logs/relative_cage.py
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/ros2_ws/third_party/ManiFlow_Policy/ManiFlow")

from viz import labels as lab  # noqa: E402
from viz.data import Rollout  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZARR = "/ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr"
CKPTS = ["foil_lifting_wrist_e40", "foil_lifting_relative_e40"]


def training_poses():
    """-> (N, 3) absolute EE positions and (N, 4) quats from the zarr.

    Read straight off the replay buffer rather than through the dataset: the
    workspace box and the OOD reference are properties of the DATA, not of
    any particular windowing.
    """
    from maniflow.common.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.copy_from_path(ZARR, keys=["state"])
    st = np.asarray(rb["state"], dtype=np.float64)
    return st[:, 0:3], st[:, 3:7]


def delta_stats():
    """Per-step relative deltas over training windows -> clamp values."""
    from maniflow.dataset.relative_dataset import RelativeRealDexDataset
    from maniflow.dataset.relative_pose import rot6d_to_matrix

    n_obs, horizon = 2, 64
    ds = RelativeRealDexDataset(
        zarr_path=ZARR, horizon=horizon, pad_before=n_obs - 1,
        pad_after=horizon - n_obs, n_obs_steps=n_obs, val_ratio=0.0,
        normalizer_samples=50)

    rng = np.random.default_rng(0)
    n = min(1500, len(ds.sampler))
    idx = rng.choice(len(ds.sampler), size=n, replace=False)

    pos_d, rot_d = [], []
    for i in idx:
        rel = ds[int(i)]["action"].numpy().astype(np.float64)
        T = rel[:, 0:3]
        pos_d.append(np.linalg.norm(np.diff(T, axis=0), axis=1))
        R = rot6d_to_matrix(rel[:, 3:9])
        # Angle of R[j]^T R[j+1], the step rotation.
        rel_rot = np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])
        tr = np.clip((np.trace(rel_rot, axis1=1, axis2=2) - 1) / 2, -1, 1)
        rot_d.append(np.degrees(np.arccos(tr)))

    p = np.concatenate(pos_d) * 1000.0
    r = np.concatenate(rot_d)
    print(f"\n  1. DELTA CLAMP  ({n} windows, {len(p)} steps)")
    print(f"     position step (mm): p50 {np.percentile(p, 50):6.2f}  "
          f"p99 {np.percentile(p, 99):6.2f}  "
          f"p99.9 {np.percentile(p, 99.9):6.2f}  max {p.max():6.2f}")
    print(f"     rotation step (deg): p50 {np.percentile(r, 50):6.2f}  "
          f"p99 {np.percentile(r, 99):6.2f}  "
          f"p99.9 {np.percentile(r, 99.9):6.2f}  max {r.max():6.2f}")
    print(f"\n     DELTA_CLAMP_POS_M = {np.percentile(p, 99.9) / 1000:.5f}")
    print(f"     DELTA_CLAMP_ROT_DEG = {np.percentile(r, 99.9):.3f}")
    return np.percentile(p, 99.9), np.percentile(r, 99.9)


def workspace_box(T):
    print("\n  2. WORKSPACE BOX (absolute training poses)")
    lo, hi = T.min(axis=0), T.max(axis=0)
    for ax, name in enumerate("xyz"):
        print(f"     {name}: [{lo[ax]:+.4f}, {hi[ax]:+.4f}] m "
              f"(span {(hi[ax] - lo[ax]) * 1000:.0f} mm)")
    m = 0.05
    print(f"     with {m * 1000:.0f} mm margin:")
    print(f"     WORKSPACE_LO = [{', '.join(f'{v:.4f}' for v in lo - m)}]")
    print(f"     WORKSPACE_HI = [{', '.join(f'{v:.4f}' for v in hi + m)}]")
    return lo, hi


def ood_profile(T_train, ckpt, lo, hi):
    """Distance from each commanded pose to the nearest training pose.

    Subsampled reference: 50k training poses against ~700 commanded poses
    per rollout is 35M distances, and the nearest-neighbour distance is not
    sensitive to thinning a dense trajectory set.
    """
    ref = T_train[::5]
    d = ROOT / ckpt
    labels = lab.load(d)
    if not labels:
        return

    print(f"\n  3. OOD DISTANCE - {ckpt}")
    print(f"     reference: {len(ref)} training poses")
    print(f"     {'run':>5s} {'ok':>3s} {'start':>8s} {'end':>8s} "
          f"{'max':>8s} {'trend':>9s} {'out_box':>8s}")
    print("     " + "-" * 60)

    trends = {True: [], False: []}
    for name in sorted(labels):
        r = Rollout(d / name)
        if not r.ok:
            continue
        P = np.asarray(r.cmd_pose[:, :3], float)
        # Nearest training pose per commanded pose, in chunks to bound memory.
        dist = np.empty(len(P))
        for a in range(0, len(P), 64):
            b = min(a + 64, len(P))
            dd = np.linalg.norm(P[a:b, None, :] - ref[None, :, :], axis=2)
            dist[a:b] = dd.min(axis=1)
        dist *= 1000.0

        k = max(5, len(dist) // 10)
        start, end = dist[:k].mean(), dist[-k:].mean()
        out = int(((P < lo - 0.05) | (P > hi + 0.05)).any(axis=1).sum())
        ok = labels[name].ok
        trends[ok].append(end - start)
        print(f"     {name[-3:]:>5s} {'OK' if ok else '--':>3s} "
              f"{start:>7.1f}m {end:>7.1f}m {dist.max():>7.1f}m "
              f"{end - start:>+8.1f}m {out:>8d}")

    for ok in (True, False):
        if trends[ok]:
            t = np.array(trends[ok])
            print(f"     {'successes' if ok else 'failures ':>10s}: "
                  f"median trend {np.median(t):+.1f} mm "
                  f"(n={len(t)})")
    print("     (trend = last 10% minus first 10% of the run; rising means")
    print("      the commanded pose is drifting away from the demos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-deltas", action="store_true")
    a = ap.parse_args()

    print("=== relative-policy cage, section 7.3 ===")
    print("gate 7.1 is green (test_deploy_relative_roundtrip.py), so the")
    print("decoder is ruled out and these bound the BEHAVIOUR.")

    if not a.skip_deltas:
        delta_stats()

    T, _ = training_poses()
    print(f"\n  training poses: {len(T)}")
    lo, hi = workspace_box(T)

    for ckpt in CKPTS:
        ood_profile(T, ckpt, lo, hi)


if __name__ == "__main__":
    main()
