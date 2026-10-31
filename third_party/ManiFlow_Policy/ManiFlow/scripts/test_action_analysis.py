#!/usr/bin/env python3
"""Where does the policy actually want to drive? Offline, no robot.

Purpose
-------
Before putting the arm back under policy control, answer three questions that
the live logs could not answer because every chunk was being rejected by the
wrong executor (action_dim 13 vs 27):

  1. Are the predicted POSES inside the executor's WORKSPACE box? Anything
     outside is silently turned into "Target out of bounds. Holding."
  2. Are the predicted HAND joints inside the DG-5F limits, and do they
     actually MOVE? The complaint was that the fingers never closed.
  3. Does the wire format match what control_ur_tesollo_cartesian expects,
     i.e. 27 values per step with the pinch joints in the right slots?

Everything runs from the zarr, so it is safe to execute with the robot
powered down.

Run
---
    python3 test_action_analysis.py
    python3 test_action_analysis.py --n-samples 24 --from-episode 3
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_serve import init_model
from deploy import (expand_hand_action, FULL_HAND_JOINTS, POLICY_HAND_JOINTS,
                    POSE_DIM, WIRE_ACTION_DIM)

DEFAULT_RUN_DIR = ("/ros2_ws/data/outputs/"
                   "foil_lifting_wrist-maniflow_multimodal_policy-0113_seed0")
DEFAULT_CKPT = "epoch=0070-val_action_mse=9.184398"
DEFAULT_ZARR = "/ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr"

# Mirrors control_ur_tesollo_cartesian.WORKSPACE exactly. Duplicated rather
# than imported because that module pulls in rclpy/tf2 and this script must
# run without a ROS graph.
WORKSPACE_MARGIN_M = 0.25
WORKSPACE = {
    "x": (-0.523 - WORKSPACE_MARGIN_M, 0.123 + WORKSPACE_MARGIN_M),
    "y": (-1.230 - WORKSPACE_MARGIN_M, -0.535 + WORKSPACE_MARGIN_M),
    "z": (-0.080 - WORKSPACE_MARGIN_M, 0.587 + WORKSPACE_MARGIN_M),
}


def build_obs(z, idx, n_obs, device):
    lo = idx - n_obs + 1
    to = lambda a, dt: torch.from_numpy(
        np.ascontiguousarray(a[np.newaxis, ...], dtype=dt)).to(device)
    return {
        "point_cloud": to(z["point_cloud"][lo:idx + 1], np.float32),
        "agent_pos": to(z["state"][lo:idx + 1], np.float32),
        "wrist_rgb": to(z["wrist_rgb"][lo:idx + 1], np.uint8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default=DEFAULT_ZARR)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--task", default="foil_lifting_wrist")
    ap.add_argument("--alg", default="maniflow_multimodal_policy")
    ap.add_argument("--addition-info", type=int, default=1211)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--from-episode", type=int, default=0,
                    help="walk this episode start-to-end instead of sampling")
    ap.add_argument("--num-inference-steps", type=int, default=10)
    args = ap.parse_args()

    import zarr
    z = zarr.open(args.zarr, "r")
    ep_ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ep_ends[:-1]])

    cfg, policy = init_model(args.task, args.alg, args.addition_info,
                             args.run_dir, args.ckpt)
    policy.num_inference_steps = args.num_inference_steps
    device = torch.device(cfg.training.device)
    n_obs, n_act = cfg.n_obs_steps, cfg.n_action_steps

    ep = args.from_episode
    lo, hi = int(starts[ep]) + n_obs - 1, int(ep_ends[ep]) - n_act
    idxs = np.linspace(lo, hi, min(args.n_samples, max(1, hi - lo))).astype(int)
    print(f"[data] episode {ep}: frames {starts[ep]}..{ep_ends[ep]}, "
          f"probing {len(idxs)} points")

    _ = policy.predict_action(build_obs(z, idxs[0], n_obs, device))  # warmup

    all_wire, all_pred, all_gt = [], [], []
    for i in idxs:
        torch.manual_seed(0)
        with torch.no_grad():
            out = policy.predict_action(build_obs(z, i, n_obs, device))
        pred = out["action"][0].cpu().numpy().astype(np.float64)
        q = pred[:, 3:7]
        n = np.linalg.norm(q, axis=1, keepdims=True); n[n < 1e-9] = 1.0
        pred[:, 3:7] = q / n
        all_pred.append(pred)
        all_gt.append(z["action"][i:i + n_act].astype(np.float64))
        all_wire.append(expand_hand_action(pred))

    pred = np.concatenate(all_pred)
    gt = np.concatenate(all_gt)
    wire = np.concatenate(all_wire)

    # --- 1. wire format ---
    print("\n" + "=" * 72)
    print("WIRE FORMAT")
    print("=" * 72)
    print(f"  policy action_dim      {pred.shape[1]}")
    print(f"  wire action_dim        {wire.shape[1]}  "
          f"(executor expects {WIRE_ACTION_DIM})")
    ok = wire.shape[1] == WIRE_ACTION_DIM
    print(f"  {'[PASS]' if ok else '[FAIL]'} divisible by executor ACTION_DIM")

    # --- 2. pose vs workspace ---
    print("\n" + "=" * 72)
    print("PREDICTED POSE vs EXECUTOR WORKSPACE")
    print("=" * 72)
    viol_total = 0
    for ax, k in enumerate("xyz"):
        v, g = pred[:, ax], gt[:, ax]
        bl, bh = WORKSPACE[k]
        bad = int(((v < bl) | (v > bh)).sum())
        viol_total += bad
        print(f"  {k}: pred {v.min():+.3f}..{v.max():+.3f}   "
              f"gt {g.min():+.3f}..{g.max():+.3f}   "
              f"bounds {bl:+.3f}..{bh:+.3f}   "
              f"outside {bad}/{len(v)}"
              + ("  <<< CLAMPED" if bad else ""))
    print(f"  {'[PASS]' if not viol_total else '[FAIL]'} "
          f"{viol_total} of {len(pred)} predicted poses outside the box")

    # --- 3. hand joints ---
    print("\n" + "=" * 72)
    print("PREDICTED HAND JOINTS (deg)")
    print("=" * 72)
    print(f"  policy predicts {len(POLICY_HAND_JOINTS)} joints: "
          f"{', '.join(POLICY_HAND_JOINTS)}")
    moved = 0
    for src, name in enumerate(POLICY_HAND_JOINTS):
        col = POSE_DIM + src
        v = pred[:, col]
        g = gt[:, col]
        rng = v.max() - v.min()
        if rng > 2.0:
            moved += 1
        print(f"    {name:12} pred {v.min():+7.2f}..{v.max():+7.2f} "
              f"(span {rng:6.2f})   gt {g.min():+7.2f}..{g.max():+7.2f} "
              f"(span {g.max() - g.min():6.2f})")
    print(f"  {moved}/{len(POLICY_HAND_JOINTS)} predicted joints span > 2 deg")
    if moved == 0:
        print("  [FAIL] hand is static -> fingers would never close")
    else:
        print("  [PASS] hand moves")

    held = [n for n in FULL_HAND_JOINTS if n not in POLICY_HAND_JOINTS]
    hcols = [POSE_DIM + FULL_HAND_JOINTS.index(n) for n in held]
    print(f"  {len(held)} joints held at a constant "
          f"{wire[:, hcols].min():.1f}..{wire[:, hcols].max():.1f} deg "
          f"(HELD_JOINT_POSE_DEG)")

    # --- 4. per-tick step size ---
    print("\n" + "=" * 72)
    print("PER-TICK MOTION (within one chunk)")
    print("=" * 72)
    for k, chunk in enumerate(all_pred):
        if k:
            break
        d = np.linalg.norm(np.diff(chunk[:, :3], axis=0), axis=1) * 1000.0
        print(f"  chunk 0: mean {d.mean():.1f} mm/tick, max {d.max():.1f} "
              f"-> {d.mean() * 20:.0f} mm/s at 20 Hz")

    print("\n" + "=" * 72)
    print("TRAJECTORY (chunk starts, every probe)")
    print("=" * 72)
    print(f"  {'frame':>7} {'x':>8} {'y':>8} {'z':>8}  "
          f"{'gt_x':>8} {'gt_y':>8} {'gt_z':>8}")
    for i, p, g in zip(idxs, all_pred, all_gt):
        print(f"  {i:7d} {p[0,0]:8.3f} {p[0,1]:8.3f} {p[0,2]:8.3f}  "
              f"{g[0,0]:8.3f} {g[0,1]:8.3f} {g[0,2]:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())