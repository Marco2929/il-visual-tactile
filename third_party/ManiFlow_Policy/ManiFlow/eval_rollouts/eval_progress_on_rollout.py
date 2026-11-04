#!/usr/bin/env python3
"""Offline Progress Head Evaluation on Real-World Rollout Logs.

Evaluates a trained Spatio-Temporal ProgressHeadST checkpoint directly on
saved rollout logs (MCAP wrist bags, joint states, outcomes) without requiring
access to physical robot hardware.

Usage:
  # Evaluate a single rollout:
  python eval_rollouts/eval_progress_on_rollout.py \
      --rollout /ros2_ws/data/rollout_logs/foil_lifting_wrist_e40/rollout_000 \
      --checkpoint /ros2_ws/checkpoints/progress_head/best_progress_head.pt \
      --output_dir /ros2_ws/data/rollout_logs/eval_progress

  # Evaluate all rollouts in a folder:
  python eval_rollouts/eval_progress_on_rollout.py \
      --rollout_dir /ros2_ws/data/rollout_logs/foil_lifting_wrist_e40 \
      --checkpoint /ros2_ws/checkpoints/progress_head/best_progress_head.pt \
      --output_dir /ros2_ws/data/rollout_logs/eval_progress
"""

import argparse
import glob
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Ensure maniflow & rollout loader are importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import viz.wrist loader from data/rollout_logs
workspace_root = os.path.abspath(os.path.join(repo_root, "..", "..", ".."))
rollout_logs_dir = os.path.join(workspace_root, "data", "rollout_logs")
if rollout_logs_dir not in sys.path:
    sys.path.insert(0, rollout_logs_dir)

try:
    from viz.wrist import load_frames, bag_path
except ImportError:
    load_frames = None
    bag_path = None

from maniflow.model.progress.progress_head_st import ProgressHeadST
from maniflow.model.vision_2d.dinov3_obs_encoder import DINOv3ObsEncoder


def load_rollout_meta_and_outcome(rollout_dir: str) -> Tuple[Dict, Dict]:
    """Load meta.json and outcome.json if present."""
    r_path = pathlib.Path(rollout_dir)
    meta, outcome = {}, {}

    meta_file = r_path / "meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {meta_file}: {e}")

    outcome_file = r_path / "outcome.json"
    if outcome_file.exists():
        try:
            with open(outcome_file, "r") as f:
                outcome = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {outcome_file}: {e}")

    return meta, outcome


def load_rollout_npz(rollout_dir: str) -> Optional[Dict]:
    """Load rollout.npz if present (provides tf_pose and js_pos)."""
    npz_path = pathlib.Path(rollout_dir) / "rollout.npz"
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path)
        return {k: data[k] for k in data.files}
    except Exception as e:
        print(f"Warning: could not read {npz_path}: {e}")
        return None


def extract_wrist_tokens(
    frames: List[np.ndarray],
    encoder: DINOv3ObsEncoder,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract DINOv3 patch tokens for all frames in batches.

    Returns:
        tokens: (T, N_p, D) torch.Tensor on CPU.
    """
    all_tokens = []
    n_frames = len(frames)

    with torch.no_grad():
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch_np = np.stack(frames[start:end])  # (B, H, W, 3)
            batch_t = torch.from_numpy(batch_np).to(device)

            # extract_patch_tokens accepts channels-last (B, H, W, 3) uint8
            patches = encoder.extract_patch_tokens("wrist_rgb", batch_t)
            all_tokens.append(patches.cpu())

    return torch.cat(all_tokens, dim=0)


def predict_progress_sequence(
    tokens: torch.Tensor,
    model: ProgressHeadST,
    device: torch.device,
    stride: int = 4,
    batch_size: int = 128,
) -> np.ndarray:
    """Predict progress y_hat in [0, 1] for each frame using temporal window (t, t-s, t-2s, t-3s)."""
    T = tokens.size(0)
    preds = np.zeros(T, dtype=np.float32)

    # Build windows (T, 4, N_p, D)
    windows = []
    for t in range(T):
        t0 = t
        t1 = max(0, t - stride)
        t2 = max(0, t - 2 * stride)
        t3 = max(0, t - 3 * stride)
        # 4 frames: (4, N_p, D)
        win = torch.stack([tokens[t0], tokens[t1], tokens[t2], tokens[t3]], dim=0)
        windows.append(win)

    windows = torch.stack(windows, dim=0)  # (T, 4, N_p, D)

    model.eval()
    with torch.no_grad():
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_win = windows[start:end].to(device)
            out = model(batch_win)  # (B, 1)
            preds[start:end] = out.squeeze(-1).cpu().numpy()

    return preds


def evaluate_single_rollout(
    rollout_dir: str,
    encoder: DINOv3ObsEncoder,
    head: ProgressHeadST,
    device: torch.device,
    output_dir: str,
    stride: Optional[int] = None,
    max_frames: int = 0,
    gate_z: bool = False,
) -> Optional[Dict]:
    """Evaluate progress predictions on a single rollout directory."""
    rollout_name = os.path.basename(os.path.normpath(rollout_dir))
    print(f"\n--- Evaluating {rollout_name} ---")

    meta, outcome = load_rollout_meta_and_outcome(rollout_dir)
    started_unix = meta.get("started_unix", 0.0)

    if load_frames is None:
        raise ImportError("Could not import viz.wrist.load_frames. Make sure data/rollout_logs is available.")

    times, frames, note = load_frames(rollout_dir, started_unix, max_frames=max_frames)
    if not frames:
        print(f"Skipping {rollout_name}: {note}")
        return None

    times = np.array(times)
    fps = float(len(times) - 1) / max(float(times[-1] - times[0]), 1e-4) if len(times) > 1 else 30.0
    print(f"Loaded {len(frames)} wrist frames ({note}) | Measured FPS: {fps:.1f}")

    # Determine time-matched stride (matching 200 ms spacing from 20 Hz training data: 4 steps * 50 ms = 200 ms)
    if stride is not None and stride > 0:
        eff_stride = stride
    else:
        eff_stride = max(1, int(round(fps * 0.20)))
    print(f"Using temporal stride: {eff_stride} frames ({eff_stride / fps * 1000:.0f} ms per step, total window {3 * eff_stride / fps * 1000:.0f} ms)")

    # Extract tokens & predict progress
    tokens = extract_wrist_tokens(frames, encoder, device)
    preds = predict_progress_sequence(tokens, head, device, stride=eff_stride)

    # Optional: load robot pose / height from rollout.npz
    npz_data = load_rollout_npz(rollout_dir)
    ee_z, tf_t = None, None
    raw_preds = preds.copy()
    if npz_data and "tf_pose" in npz_data and "tf_t" in npz_data:
        tf_t = npz_data["tf_t"]
        ee_z = npz_data["tf_pose"][:, 2]  # z height

        if gate_z and len(tf_t) > 0 and len(ee_z) > 0:
            # Interpolate z onto wrist timestamps
            z_on_times = np.interp(times, tf_t, ee_z)
            min_z = float(np.min(ee_z))
            # Find first time robot descends to near lowest table height (+3cm)
            reach_table_mask = (z_on_times <= min_z + 0.03)
            first_contact_idx = np.where(reach_table_mask)[0][0] if np.any(reach_table_mask) else len(times)
            # Before first reaching the table, progress cannot exceed approach limit (0.30)
            descent_mask = np.arange(len(preds)) < first_contact_idx
            preds[descent_mask] = np.minimum(preds[descent_mask], 0.30)
            print(f"  Applied kinematic descent gate: capped {np.sum(descent_mask)} pre-contact frames to max 0.30")

    # Metrics & 80% threshold tracking
    max_progress = float(np.max(preds))
    final_progress = float(preds[-1])
    over_80_mask = (preds >= 0.80)
    reached_080 = bool(np.any(over_80_mask))
    reached_090 = bool(np.any(preds >= 0.90))

    frames_over_80 = int(np.sum(over_80_mask))
    first_t_over_80 = None
    duration_over_80 = 0.0
    peak_idx = int(np.argmax(preds))
    peak_t = float(times[peak_idx])

    if reached_080:
        time_indices = np.where(over_80_mask)[0]
        first_t_over_80 = float(times[time_indices[0]])
        duration_over_80 = float(times[time_indices[-1]] - times[time_indices[0]])

    # Plot
    verdict = outcome.get("verdict", "unknown")
    fail_mode = outcome.get("failure_mode", "")
    is_success = outcome.get("success", (verdict == "pass" or verdict == "success"))

    status_str = f"Outcome: {verdict.upper()}"
    if fail_mode:
        status_str += f" ({fail_mode})"

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Color progress curve based on success/fail
    p_color = "#2ca02c" if is_success else "#d62728"
    if gate_z and np.any(preds != raw_preds):
        ax1.plot(times, raw_preds, label="Raw Ungated Vision Progress", color="#888888", linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.plot(times, preds, label=f"ProgressHeadST ({'Gated' if gate_z else 'Raw'})", color=p_color, linewidth=2.2)
    ax1.axhline(0.90, color="#1b9e77", linestyle=":", linewidth=1.5, label="Completion Gate (0.90)")
    ax1.axhline(0.80, color="#d95f02", linestyle="--", linewidth=1.5, label="User Alert Threshold (0.80)")
    ax1.axhline(0.70, color="#7570b3", linestyle=":", linewidth=1.0, label="Pinch Grasp Target (0.70)")

    # Highlight region >= 0.80 if present
    if reached_080:
        ax1.fill_between(times, 0.80, preds, where=over_80_mask, color="#d95f02", alpha=0.3, label="Region >= 0.80")

    ax1.set_xlabel("Rollout Time [s]", fontsize=11)
    ax1.set_ylabel("Progress Estimate", fontsize=11, color=p_color)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis='y', labelcolor=p_color)
    ax1.grid(True, alpha=0.3)

    # Plot EE height on secondary axis if available
    if ee_z is not None and tf_t is not None:
        ax2 = ax1.twinx()
        ax2.plot(tf_t, ee_z, label="EE z-Height [m]", color="#1f77b4", linestyle="--", alpha=0.6)
        ax2.set_ylabel("EE Height [m]", fontsize=11, color="#1f77b4")
        ax2.tick_params(axis='y', labelcolor="#1f77b4")

    # Title & Sub-annotation
    alert_badge = f" [ALERT: Peak {max_progress:.1%} >= 80%]" if reached_080 else f" [Max: {max_progress:.1%}]"
    title = f"Progress Head: {rollout_name} | {status_str}{alert_badge}"
    plt.title(title, fontsize=11, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ee_z is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    else:
        ax1.legend(lines1, labels1, loc="upper left", fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_img = os.path.join(output_dir, f"{rollout_name}_progress.png")
    plt.savefig(out_img, dpi=150)
    plt.close(fig)
    print(f"Saved evaluation plot: {out_img}")

    # Save progress.npz directly inside the rollout directory for instant caching & viz
    npz_out = os.path.join(rollout_dir, "progress.npz")
    np.savez_compressed(
        npz_out,
        t=times.astype(np.float32),
        progress=preds.astype(np.float32),
        max_progress=np.float32(max_progress),
        final_progress=np.float32(final_progress),
        reached_080=bool(reached_080),
        peak_time_s=np.float32(peak_t),
    )

    # Console feedback for this rollout
    if reached_080:
        fp_note = " *** [FALSE POSITIVE WARNING] Failed rollout exceeded 80%!" if not is_success else ""
        print(f"  >>> [ALERT > 80%] Peak: {max_progress:.3f} at t={peak_t:.2f}s | >80% for {frames_over_80} frames (~{duration_over_80:.2f}s){fp_note}")
    else:
        print(f"  [OK] Peak: {max_progress:.3f} (stayed below 80%) | Final: {final_progress:.3f}")

    return {
        "rollout": rollout_name,
        "is_success": is_success,
        "verdict": verdict,
        "failure_mode": fail_mode,
        "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "num_frames": len(frames),
        "max_progress": max_progress,
        "final_progress": final_progress,
        "peak_time_s": peak_t,
        "reached_080": reached_080,
        "first_t_over_80": first_t_over_80,
        "frames_over_80": frames_over_80,
        "duration_over_80_s": duration_over_80,
        "reached_completion_gate": reached_090,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProgressHeadST on rollout logs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rollout", type=str, help="Path to a single rollout directory (e.g. rollout_000).")
    group.add_argument("--rollout_dir", type=str, help="Path to directory containing multiple rollout_* folders.")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_progress_head.pt checkpoint.")
    parser.add_argument("--output_dir", type=str, default="eval_rollouts/plots", help="Directory to save evaluation plots.")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--stride", type=int, default=0, help="Frame stride for 4-frame window. If 0 (default), auto-computed to match 200ms spacing (s=6 at 30Hz).")
    parser.add_argument("--gate_z", action="store_true", help="Apply kinematic descent gate (cap progress at 0.30 before first table contact).")
    parser.add_argument("--max_frames", type=int, default=0, help="Max wrist frames to decode (0 = all).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Init] Loading ProgressHeadST from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    feature_dim = ckpt.get('feature_dim', 384)

    head = ProgressHeadST(
        feature_dim=feature_dim,
        patch_dropout=0.0,
        noise_ratio=0.0,
    ).to(device).eval()
    head.load_state_dict(state_dict)

    print(f"[Init] Initializing DINOv3 ObsEncoder ({args.model_name})...")
    shape_meta = {'obs': {'wrist_rgb': {'shape': [3, 192, 256], 'type': 'rgb', 'horizon': 1}}}
    encoder = DINOv3ObsEncoder(
        shape_meta=shape_meta,
        model_name=args.model_name,
        pretrained=True,
        frozen=True,
        share_rgb_model=False,
    ).to(device).eval()

    # Collect rollouts to process
    if args.rollout:
        rollout_dirs = [args.rollout]
    else:
        # Search for rollout_* subdirs
        rollout_dirs = sorted([
            p for p in glob.glob(os.path.join(args.rollout_dir, "rollout_*"))
            if os.path.isdir(p)
        ])
        if not rollout_dirs:
            # Maybe the passed dir is itself a rollout?
            if os.path.exists(os.path.join(args.rollout_dir, "wrist_bag")):
                rollout_dirs = [args.rollout_dir]
            else:
                raise ValueError(f"No rollout_* folders found in {args.rollout_dir}")

    print(f"Found {len(rollout_dirs)} rollout(s) to evaluate.")

    results = []
    for r_dir in rollout_dirs:
        res = evaluate_single_rollout(
            rollout_dir=r_dir,
            encoder=encoder,
            head=head,
            device=device,
            output_dir=args.output_dir,
            stride=args.stride if args.stride > 0 else None,
            max_frames=args.max_frames,
            gate_z=args.gate_z,
        )
        if res:
            results.append(res)

    if results:
        # Save summary JSON
        summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Summary] Wrote summary of {len(results)} rollouts to {summary_path}")

        # Print statistics table
        print("\n" + "=" * 90)
        print(f"{'Rollout':<14} | {'Verdict':<8} | {'Failure Mode':<14} | {'Max Prog':<9} | {'Final Prog':<10} | {'>=80%?':<16} | {'>=90%?'}")
        print("-" * 90)
        for r in results:
            v_str = "SUCCESS" if r["is_success"] else "FAIL"
            mode = r["failure_mode"] if r["failure_mode"] else "-"
            over_80_str = f"YES ({r['max_progress']:.1%}, {r['duration_over_80_s']:.1f}s)" if r["reached_080"] else f"no ({r['max_progress']:.1%})"
            over_90_str = "YES" if r["reached_completion_gate"] else "no"
            print(f"{r['rollout']:<14} | {v_str:<8} | {mode:<14} | {r['max_progress']:<9.3f} | {r['final_progress']:<10.3f} | {over_80_str:<16} | {over_90_str}")
        print("=" * 90)

        successes = [r for r in results if r["is_success"]]
        failures = [r for r in results if not r["is_success"]]

        success_80 = sum(1 for r in successes if r["reached_080"])
        success_90 = sum(1 for r in successes if r["reached_completion_gate"])

        false_80 = sum(1 for r in failures if r["reached_080"])
        false_90 = sum(1 for r in failures if r["reached_completion_gate"])

        print(f"\n--- Evaluation Summary ---")
        print(f"Total Rollouts Evaluated: {len(results)}")
        print(f"SUCCESS Rollouts ({len(successes)}):")
        print(f"  Reached >= 0.80: {success_80}/{len(successes)} ({success_80/max(1, len(successes)):.1%})")
        print(f"  Reached >= 0.90: {success_90}/{len(successes)} ({success_90/max(1, len(successes)):.1%})")
        if successes:
            print(f"  Mean Peak Progress: {np.mean([r['max_progress'] for r in successes]):.3f}")

        print(f"FAILURE Rollouts ({len(failures)}):")
        print(f"  False Positives (>= 0.80): {false_80}/{len(failures)} ({false_80/max(1, len(failures)):.1%})")
        print(f"  False Positives (>= 0.90): {false_90}/{len(failures)} ({false_90/max(1, len(failures)):.1%})")
        if failures:
            print(f"  Mean Peak Progress: {np.mean([r['max_progress'] for r in failures]):.3f}")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

