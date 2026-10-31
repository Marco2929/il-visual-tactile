#!/usr/bin/env python3
"""Offline replay: does the checkpoint reproduce its own training actions?

Why this exists
---------------
Live deployment showed ~70 mm of chunk-to-chunk disagreement in the RTC
continuity metric on a PARKED robot. Three preprocessing bugs were fixed
(missing RGB channels, metre/millimetre filter domain, HD1080-vs-HD720
intrinsics) and none of them moved that number. Raising the flow steps from
6 to 10 did not move it either.

That leaves two very different explanations, which the live log cannot
separate because both produce the same symptom:

  (A) distribution shift / residual preprocessing skew - the policy is fine
      but the observations it gets live are not the ones it trained on, or
  (B) the policy itself is noisy - undertrained, overfit, or genuinely
      multi-modal, in which case more data is the answer.

This script feeds TRAINING observations straight from the zarr into the same
policy.predict_action() the deploy loop calls, and compares against the
ground-truth actions stored next to them. No ROS, no cameras, no executor,
no RTC. So:

  small error here  -> the policy reproduces its training data; the live
                       problem is (A), keep debugging the deploy path.
  large error here  -> the problem is (B), and it is a training decision.

It also measures two variance components that the live metric conflates:

  seed variance        same observation, many noise draws. This is the
                       sampling noise floor; the RTC metric can never be
                       better than this.
  observation variance consecutive observations, one fixed seed. How much
                       the output moves because the INPUT moved.

If seed variance alone already accounts for ~70 mm, the live metric was
never measuring a bug at all.

Note on normalisation: the zarr holds RAW point clouds / state / actions.
predict_action() applies the policy's own LinearNormalizer internally, which
is exactly what deploy.py relies on too, so feeding raw values is correct.

Run
---
    python3 test_offline_replay.py
    python3 test_offline_replay.py --n-samples 64 --seed-trials 16
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_serve import init_model

# Same checkpoint the deploy entry point uses.
DEFAULT_RUN_DIR = ("/ros2_ws/data/outputs/"
                   "foil_lifting_wrist-maniflow_multimodal_policy-0113_seed0")
DEFAULT_CKPT = "epoch=0070-val_action_mse=9.184398"
DEFAULT_ZARR = "/ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr"


def quat_normalize(actions: np.ndarray) -> np.ndarray:
    """Match model_serve.normalize(): unit-norm the quaternion block.

    Without this the comparison would charge the policy for quaternion scale
    that the deploy path fixes up anyway.
    """
    out = actions.copy()
    quats = out[:, 3:7]
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    out[:, 3:7] = quats / norms
    return out


def build_obs(zroot, idx, n_obs_steps, device):
    """Observation window ending at idx, in the layout deploy.py produces.

    Shapes and dtypes mirror RealDexEnv.snapshot_obs() exactly:
      point_cloud (1, T, 1024, 6) float32
      agent_pos   (1, T, 15)      float32
      wrist_rgb   (1, T, H, W, 3) uint8, channels-LAST
    The policy's _encode_wrist() is the single HWC->CHW boundary, so handing
    it CHW here would double-permute.
    """
    lo = idx - n_obs_steps + 1
    pc = zroot["point_cloud"][lo:idx + 1]
    state = zroot["state"][lo:idx + 1]
    wrist = zroot["wrist_rgb"][lo:idx + 1]

    to = lambda a, dt: torch.from_numpy(
        np.ascontiguousarray(a[np.newaxis, ...], dtype=dt)).to(device)
    return {
        "point_cloud": to(pc, np.float32),
        "agent_pos": to(state, np.float32),
        "wrist_rgb": to(wrist, np.uint8),
    }


def predict(policy, obs, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = policy.predict_action(obs)
    return quat_normalize(out["action"][0].cpu().numpy().astype(np.float64))


def stats(name, values, unit="mm", scale=1000.0):
    v = np.asarray(values, dtype=np.float64) * scale
    print(f"  {name:<34} n={len(v):<5} mean={v.mean():7.2f}{unit}  "
          f"sd={v.std():6.2f}  p95={np.percentile(v, 95):7.2f}  "
          f"max={v.max():7.2f}")
    return v.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default=DEFAULT_ZARR)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--task", default="foil_lifting_wrist")
    ap.add_argument("--alg", default="maniflow_multimodal_policy")
    ap.add_argument("--addition-info", type=int, default=1211)
    ap.add_argument("--n-samples", type=int, default=48,
                    help="random frames drawn for the accuracy test")
    ap.add_argument("--seed-trials", type=int, default=12,
                    help="noise draws per frame for the variance test")
    ap.add_argument("--num-inference-steps", type=int, default=10)
    args = ap.parse_args()

    import zarr
    zroot = zarr.open(args.zarr, "r")
    n_frames = zroot["action"].shape[0]
    ep_ends = zroot["meta/episode_ends"][:]
    print(f"[data] {args.zarr}")
    print(f"[data] {n_frames} frames, {len(ep_ends)} episodes")

    cfg, policy = init_model(args.task, args.alg, args.addition_info,
                             args.run_dir, args.ckpt)
    policy.num_inference_steps = args.num_inference_steps
    device = torch.device(cfg.training.device)
    n_obs = cfg.n_obs_steps
    n_act = cfg.n_action_steps
    print(f"[cfg] n_obs_steps={n_obs} n_action_steps={n_act} "
          f"num_inference_steps={args.num_inference_steps}")

    # Episode starts, so an observation window never straddles a cut (which
    # would pair a cloud with state from a different episode).
    starts = np.concatenate([[0], ep_ends[:-1]])
    def episode_of(i):
        return int(np.searchsorted(ep_ends, i, side="right"))
    def window_ok(i):
        return i - n_obs + 1 >= starts[episode_of(i)] and \
               i + n_act <= ep_ends[episode_of(i)]

    rng = np.random.default_rng(0)
    candidates = [i for i in rng.choice(n_frames, size=min(n_frames, 4000),
                                        replace=False) if window_ok(i)]
    if not candidates:
        print("[FAIL] no valid observation windows")
        return 1

    # --- warmup: torch.compile pays its JIT cost on the first call ---
    _ = predict(policy, build_obs(zroot, candidates[0], n_obs, device), seed=0)

    # ---------------------------------------------------------------
    # 1. ACCURACY vs ground truth
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ACCURACY ON TRAINING DATA (policy vs stored actions)")
    print("=" * 72)
    xyz_step0, xyz_mean, xyz_max = [], [], []
    for i in candidates[:args.n_samples]:
        pred = predict(policy, build_obs(zroot, i, n_obs, device), seed=0)
        gt = quat_normalize(
            zroot["action"][i:i + n_act].astype(np.float64))
        k = min(len(pred), len(gt))
        d = np.linalg.norm(pred[:k, :3] - gt[:k, :3], axis=1)
        xyz_step0.append(d[0])
        xyz_mean.append(d.mean())
        xyz_max.append(d.max())
    stats("position error, step 0", xyz_step0)
    acc_mean = stats("position error, chunk mean", xyz_mean)
    stats("position error, chunk max", xyz_max)

    # ---------------------------------------------------------------
    # 2. SEED VARIANCE - the sampling noise floor
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SEED VARIANCE (identical observation, different noise draws)")
    print("=" * 72)
    spread_step0, spread_mean = [], []
    for i in candidates[:max(4, args.n_samples // 4)]:
        obs = build_obs(zroot, i, n_obs, device)
        preds = np.stack([predict(policy, obs, seed=s)[:, :3]
                          for s in range(args.seed_trials)])
        centroid = preds.mean(axis=0)
        dev = np.linalg.norm(preds - centroid, axis=2)  # (trials, T)
        spread_step0.append(dev[:, 0].mean())
        spread_mean.append(dev.mean())
    stats("spread about centroid, step 0", spread_step0)
    seed_mean = stats("spread about centroid, chunk mean", spread_mean)

    # ---------------------------------------------------------------
    # 3. OBSERVATION SENSITIVITY - fixed seed, consecutive frames
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("OBSERVATION SENSITIVITY (consecutive frames, fixed seed)")
    print("=" * 72)
    print("  Compares chunk t+1 against the time-shifted tail of chunk t,")
    print("  i.e. the offline analogue of the live [RTC] metric at delay=1.")
    cont_step0, cont_mean = [], []
    for i in candidates[:max(4, args.n_samples // 4)]:
        if not window_ok(i + 1):
            continue
        a = predict(policy, build_obs(zroot, i, n_obs, device), seed=0)
        b = predict(policy, build_obs(zroot, i + 1, n_obs, device), seed=0)
        k = min(len(a) - 1, len(b))
        d = np.linalg.norm(b[:k, :3] - a[1:k + 1, :3], axis=1)
        cont_step0.append(d[0])
        cont_mean.append(d.mean())
    obs_mean = stats("successor disagreement, step 0", cont_step0)
    stats("successor disagreement, chunk mean", cont_mean)

    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print(f"  accuracy (chunk mean)      {acc_mean:7.2f} mm")
    print(f"  seed noise floor           {seed_mean:7.2f} mm")
    print(f"  observation sensitivity    {obs_mean:7.2f} mm")
    print(f"  live [RTC] step0 observed   ~70.00 mm (parked robot)")
    print()
    if seed_mean > 40.0:
        print("  => Sampling noise ALONE is the same order as the live RTC")
        print("     figure. The live metric was largely measuring the flow")
        print("     sampler's own variance, not a deploy bug. Fix the seed or")
        print("     average samples before reading continuity.")
    if acc_mean > 50.0:
        print("  => Large error on its OWN TRAINING DATA. This is not")
        print("     distribution shift; it points at the checkpoint itself")
        print("     (undertrained / overfit / too few episodes).")
    elif acc_mean < 20.0:
        print("  => Reproduces training actions well. The checkpoint is")
        print("     sound, so the live behaviour is distribution shift or a")
        print("     remaining deploy-path skew - NOT a training-data problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())