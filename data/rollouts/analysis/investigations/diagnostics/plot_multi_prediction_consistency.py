#!/usr/bin/env python3
"""Plot repeated trajectory predictions (K=10 samples) on the exact same input observation.

Demonstrates whether the policy outputs similar trajectories across multiple
stochastic flow samples or exhibits mode collapse / erratic variance.
"""

import os
import pathlib
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_MANIFLOW_SCRIPTS = pathlib.Path("/ros2_ws/third_party/ManiFlow_Policy/ManiFlow/scripts")
if str(_MANIFLOW_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_MANIFLOW_SCRIPTS))

from ablate_modalities import build_val_dataset, load_policy
from maniflow.common.phase_detection import PHASE_NAMES, phase_of_windows
from maniflow.common.pytorch_util import dict_apply

CKPT = "/ros2_ws/data/outputs/foil_lifting_wrist_recorded_action-maniflow_multimodal_policy-0016_abl_wrist_only_seed0/checkpoints/epoch=0060-val_action_mse=0.026752.ckpt"
OUT_PLOT = "/ros2_ws/data/rollout_logs/investigations/modality_ablation/plots/prediction_consistency_trajectories.png"


@torch.no_grad()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading policy from {CKPT}...")
    policy, cfg = load_policy(CKPT, device, use_ema=True)
    policy.eval()

    dataset = build_val_dataset(cfg)
    phases, _ = phase_of_windows(dataset.replay_buffer, dataset.sampler.indices)

    norm = policy.normalizer["action"]

    # Select 1 representative window for each phase: approach (0), search (1), hold (3)
    target_phases = [0, 1, 3]
    phase_labels = ["Approach (Anfahrt)", "Search (Entscheidung / Greifpunkt)", "Hold (Heben / Abzug)"]

    rng = np.random.default_rng(42)
    selected_indices = []
    for p in target_phases:
        pool = np.flatnonzero(phases == p)
        idx = rng.choice(pool)
        selected_indices.append(idx)

    # 4 rows: X (mm), Y (mm), Z (mm), Gripper Close (deg)
    # 3 cols: Approach, Search, Hold
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    plt.subplots_adjust(hspace=0.25, wspace=0.25)

    K = 10
    timesteps = np.arange(32) * (1.0 / 20.0)  # 20 Hz = seconds

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, K))

    for col_idx, (p_idx, p_name, win_idx) in enumerate(zip(target_phases, phase_labels, selected_indices)):
        sample = dataset[win_idx]
        batch = dict_apply(sample, lambda x: torch.as_tensor(x, device=device).unsqueeze(0))

        # Ground truth unnormalized
        start = cfg.n_obs_steps - 1
        end = start + cfg.n_action_steps
        gt_norm = batch["action"].squeeze(0)  # [horizon, 15]
        gt_unnorm = norm.unnormalize(gt_norm).cpu().numpy()[start:end]  # take executed chunk

        # Run K predictions
        preds_unnorm = []
        for k in range(K):
            out = policy.predict_action(batch["obs"])
            pred_un = out["action"].squeeze(0).cpu().numpy()  # [n_action_steps, 15]
            preds_unnorm.append(pred_un)

        preds_unnorm = np.stack(preds_unnorm, axis=0)  # [K, 32, 15]

        # In 0016: actions are [x, y, z, qx, qy, qz, qw, hand_0..hand_7]
        # x, y, z in meters -> convert to mm relative to start pose for clarity, or absolute mm
        x_gt = gt_unnorm[:, 0] * 1000.0
        y_gt = gt_unnorm[:, 1] * 1000.0
        z_gt = gt_unnorm[:, 2] * 1000.0
        # Finger 2 index grasp sum:
        g_gt = gt_unnorm[:, 7:11].sum(axis=-1)

        x_preds = preds_unnorm[:, :, 0] * 1000.0
        y_preds = preds_unnorm[:, :, 1] * 1000.0
        z_preds = preds_unnorm[:, :, 2] * 1000.0
        g_preds = preds_unnorm[:, :, 7:11].sum(axis=-1)

        # Plot X
        ax_x = axes[0, col_idx]
        for k in range(K):
            ax_x.plot(timesteps, x_preds[k], color=colors[k], alpha=0.6, lw=1.5, label="Samples" if k == 0 else "")
        ax_x.plot(timesteps, x_preds.mean(axis=0), color="blue", lw=2.2, label="Sample Mean")
        ax_x.plot(timesteps, x_gt, color="black", ls="--", lw=2.0, label="Ground Truth")
        x_spread = (x_preds.max(axis=0) - x_preds.min(axis=0)).mean()
        ax_x.set_title(f"{p_name}\nX Spread: ±{x_spread/2:.1f} mm", fontsize=11, fontweight="bold")
        ax_x.set_ylabel("EE X (mm)")
        ax_x.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_x.legend(fontsize=8, loc="upper right")

        # Plot Y
        ax_y = axes[1, col_idx]
        for k in range(K):
            ax_y.plot(timesteps, y_preds[k], color=colors[k], alpha=0.6, lw=1.5)
        ax_y.plot(timesteps, y_preds.mean(axis=0), color="blue", lw=2.2)
        ax_y.plot(timesteps, y_gt, color="black", ls="--", lw=2.0)
        y_spread = (y_preds.max(axis=0) - y_preds.min(axis=0)).mean()
        ax_y.set_title(f"Y Spread: ±{y_spread/2:.1f} mm", fontsize=10)
        ax_y.set_ylabel("EE Y (mm)")
        ax_y.grid(True, alpha=0.3)

        # Plot Z
        ax_z = axes[2, col_idx]
        for k in range(K):
            ax_z.plot(timesteps, z_preds[k], color=colors[k], alpha=0.6, lw=1.5)
        ax_z.plot(timesteps, z_preds.mean(axis=0), color="blue", lw=2.2)
        ax_z.plot(timesteps, z_gt, color="black", ls="--", lw=2.0)
        z_spread = (z_preds.max(axis=0) - z_preds.min(axis=0)).mean()
        ax_z.set_title(f"Z Spread: ±{z_spread/2:.1f} mm", fontsize=10)
        ax_z.set_ylabel("EE Z (mm)")
        ax_z.grid(True, alpha=0.3)

        # Plot Gripper
        ax_g = axes[3, col_idx]
        for k in range(K):
            ax_g.plot(timesteps, g_preds[k], color=colors[k], alpha=0.6, lw=1.5)
        ax_g.plot(timesteps, g_preds.mean(axis=0), color="blue", lw=2.2)
        ax_g.plot(timesteps, g_gt, color="black", ls="--", lw=2.0)
        g_spread = (g_preds.max(axis=0) - g_preds.min(axis=0)).mean()
        ax_g.set_title(f"Grasp Spread: ±{g_spread/2:.1f}°", fontsize=10)
        ax_g.set_ylabel("Finger Angle (deg)")
        ax_g.set_xlabel("Time in Chunk (s)")
        ax_g.grid(True, alpha=0.3)

    fig.suptitle(
        "ManiFlow Trajectory Repeatability: K=10 Repeated Predictions on Same Observation\n"
        "(Run 0016: Abs Wrist Only, Epoche 60 - Identical Input, Differing Initial Noise x0 ~ N(0, I))",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=200, bbox_inches="tight")
    print(f"Saved trajectory consistency plot to {OUT_PLOT}")


if __name__ == "__main__":
    main()
