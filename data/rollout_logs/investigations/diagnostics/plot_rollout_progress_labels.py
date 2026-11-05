#!/usr/bin/env python3
"""Plot how the dataset builder's progress labeling tool responds to actual rollout data.

Loads rollout logs (joint states, TF poses, outcomes, and wrist camera frames),
runs the progress labeling pipeline (extract_grasp_signal, segment_grasp_attempts,
build_progress_curve), and plots:
  1. Wrist camera frames at key task moments (Start, Grasp Closing, Lock/Peel, Lift)
  2. Detected grasp signal & attempt segmentation (threshold, failed vs final attempt)
  3. Continuous progress curve y(t) with peel threshold theta

Usage:
  python plot_rollout_progress_labels.py [--rollouts rollout_000 rollout_013 rollout_021]
"""

import argparse
import json
import os
import pathlib
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project paths
_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "data" / "dataset_builder"))
sys.path.insert(0, str(_REPO / "data" / "rollout_logs"))

import viz.wrist as wrist
from processing.progress import (
    CONSERVATIVE,
    build_progress_curve,
    extract_grasp_signal,
    segment_grasp_attempts,
    smooth,
)

DEFAULT_ROLLOUT_DIR = _REPO / "data" / "rollout_logs" / "foil_lifting_wrist_e40"
OUT_PLOT = _REPO / "data" / "rollout_logs" / "investigations" / "modality_ablation" / "plots" / "rollout_progress_label_verification.png"
ARTIFACT_PLOT = pathlib.Path("/home/mm/.gemini/antigravity/brain/93b2b22b-696f-4543-8358-70a30cd58288/plots/rollout_progress_label_verification.png")


def evaluate_rollout_progress(ep_dir, params=CONSERVATIVE):
    """Load rollout and apply dataset builder progress labeling."""
    npz_path = ep_dir / "rollout.npz"
    if not npz_path.exists():
        return None

    z_data = np.load(npz_path)
    meta = json.loads((ep_dir / "meta.json").read_text()) if (ep_dir / "meta.json").exists() else {}
    outcome = json.loads((ep_dir / "outcome.json").read_text()) if (ep_dir / "outcome.json").exists() else {}

    verdict = outcome.get("verdict", "fail")
    is_success = verdict in ("pass", "success")
    fail_mode = outcome.get("failure_mode", "none")

    started_unix = meta.get("started_unix", 0.0)

    # Load wrist frames from bag
    times, frames = [], []
    if (ep_dir / "wrist_bag").exists():
        try:
            times, frames, _ = wrist.load_frames(ep_dir, started_unix, max_frames=0)
        except Exception as e:
            print(f"Warning loading wrist bag for {ep_dir.name}: {e}")

    js_t = z_data["js_t"]
    js_pos = z_data["js_pos"]

    if frames and len(times) > 0:
        n = len(frames)
        resampled_js = np.zeros((n, js_pos.shape[1]), dtype=np.float32)
        for j in range(js_pos.shape[1]):
            resampled_js[:, j] = np.interp(times, js_t, js_pos[:, j])
        raw = resampled_js
        if "tf_pose" in z_data and "tf_t" in z_data and len(z_data["tf_t"]) > 0:
            tf_t = z_data["tf_t"]
            ee_z = z_data["tf_pose"][:, 2]
            z = np.interp(times, tf_t, ee_z)
        else:
            z = raw[:, 2]
    else:
        raw = js_pos
        n = len(raw)
        z = z_data["tf_pose"][:, 2] if "tf_pose" in z_data and len(z_data["tf_pose"]) == n else raw[:, 2]
        times = js_t - js_t[0]

    # Run dataset builder labeling functions
    sig = extract_grasp_signal(joint_states=raw)
    g = smooth(sig, 5)
    seg = segment_grasp_attempts(sig)

    if seg["has_grasp"]:
        prog = build_progress_curve(
            num_frames=n,
            attempts=seg["attempts"],
            t_first=seg["t_first"],
            t_last=seg["t_last"],
            t_lock=seg["t_lock"],
            z=z,
            params=params,
            is_success=is_success,
        )
    else:
        prog = np.zeros(n, dtype=np.float32)

    return {
        "name": ep_dir.name,
        "n": n,
        "frames": frames,
        "times": times,
        "raw": raw,
        "z": z,
        "sig": sig,
        "g": g,
        "seg": seg,
        "prog": prog,
        "verdict": verdict,
        "is_success": is_success,
        "fail_mode": fail_mode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", type=pathlib.Path, default=DEFAULT_ROLLOUT_DIR)
    ap.add_argument(
        "--rollouts",
        nargs="*",
        default=["rollout_000", "rollout_011", "rollout_013", "rollout_021"],
        help="Rollout IDs to plot (representing fail miss, fail edge, success)",
    )
    args = ap.parse_args()

    rollout_dirs = [args.rollout_dir / r for r in args.rollout_dir.glob("rollout_*") if r.name in args.rollouts]
    rollout_dirs = sorted(rollout_dirs, key=lambda p: p.name)

    results = []
    for rdir in rollout_dirs:
        res = evaluate_rollout_progress(rdir)
        if res is not None:
            results.append(res)

    if not results:
        print("No rollouts found to evaluate.")
        return

    n_rollouts = len(results)
    fig = plt.figure(figsize=(18, 4.5 * n_rollouts))
    gs = fig.add_gridspec(n_rollouts, 2, width_ratios=[1.2, 1.8], hspace=0.35, wspace=0.20)

    for row_idx, res in enumerate(results):
        n = res["n"]
        t = np.arange(n)
        seg = res["seg"]
        frames = res["frames"]

        # Left Column: Filmstrip of 4 key frames
        gs_left = gs[row_idx, 0].subgridspec(1, 4, wspace=0.05)
        # Select 4 key frames: t=0, t=first_attempt, t=peel_lock, t=end
        t_peel = seg.get("t_lock", seg.get("t_last", int(0.6 * n)))
        t_first = seg.get("t_first", int(0.3 * n))
        key_frames_idx = [0, max(0, min(t_first, n - 1)), max(0, min(t_peel, n - 1)), n - 1]
        frame_labels = ["Start", "Grasp Closing", "Contact / Grip", "Lift / Outcome"]

        for col_f, (f_idx, f_lbl) in enumerate(zip(key_frames_idx, frame_labels)):
            ax_f = fig.add_subplot(gs_left[0, col_f])
            ax_f.axis("off")
            if frames and len(frames) > f_idx:
                ax_f.imshow(frames[f_idx])
            else:
                ax_f.text(0.5, 0.5, "No Image", ha="center", va="center")
            ax_f.set_title(f"{f_lbl}\n(f={f_idx})", fontsize=8, fontweight="bold")

        # Right Column: Grasp Signal (Top) and Progress Curve (Bottom)
        gs_right = gs[row_idx, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.25)
        ax_g = fig.add_subplot(gs_right[0, 0])
        ax_y = fig.add_subplot(gs_right[1, 0], sharex=ax_g)

        # Plot Grasp Signal
        ax_g.plot(t, res["g"], color="#222222", lw=1.5, label="Grasp Signal (deg)")
        thr = seg.get("threshold")
        if thr is not None and np.isfinite(thr):
            ax_g.axhline(thr, color="#ff7f0e", ls="--", lw=1.2, label=f"Threshold ({thr:.1f}°)")

        for a, b in seg["attempts"]:
            final = a >= seg["t_last"]
            c = "#2ca02c" if (final and res["is_success"]) else "#d62728"
            lbl = "Final Grasp" if final else "Failed Attempt"
            ax_g.axvspan(a, b, color=c, alpha=0.20, label=lbl if a == seg["attempts"][0][0] else "")

        v_col = "#2ca02c" if res["is_success"] else "#d62728"
        status_txt = f"{res['name']}: {res['verdict'].upper()} (fail={res['fail_mode']})"
        ax_g.set_title(status_txt, fontsize=10, fontweight="bold", color=v_col, loc="left")
        ax_g.set_ylabel("Grasp Angle (°)", fontsize=9)
        ax_g.grid(True, alpha=0.3)
        ax_g.legend(loc="upper right", fontsize=8)

        # Plot Progress Curve
        ax_y.plot(t, res["prog"], color="#1f77b4", lw=2.0, label="Labeling Tool Progress y(t)")
        ax_y.axhline(CONSERVATIVE.theta, color="#888888", ls=":", lw=1.0, label=f"Theta ({CONSERVATIVE.theta:.2f})")

        if seg["has_grasp"]:
            if res["is_success"]:
                ax_y.axvspan(t_peel, n - 1, color="#2ca02c", alpha=0.08, label="Peel Region")
            else:
                ax_y.text(0.5, 0.4, "Failed grasp -> Progress clamped / penalised", transform=ax_y.transAxes,
                          ha="center", fontsize=9, color="#d62728", style="italic")

        ax_y.set_ylim(-0.05, 1.05)
        ax_y.set_ylabel("Progress Label [0..1]", fontsize=9)
        ax_y.set_xlabel("Video Frame Index (20 Hz)", fontsize=9)
        ax_y.grid(True, alpha=0.3)
        ax_y.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Response of Dataset Builder Progress Labeling Tool on Real-Robot Rollouts\n"
        "(Grasp Attempt Segmentation, Physical Angle Thresholding, and Computed Progress Curves)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    os.makedirs(OUT_PLOT.parent, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=200, bbox_inches="tight")
    os.makedirs(ARTIFACT_PLOT.parent, exist_ok=True)
    plt.savefig(ARTIFACT_PLOT, dpi=200, bbox_inches="tight")
    print(f"Saved rollout progress verification plot to {OUT_PLOT} and {ARTIFACT_PLOT}")


if __name__ == "__main__":
    main()

