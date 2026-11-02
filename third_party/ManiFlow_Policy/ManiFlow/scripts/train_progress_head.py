#!/usr/bin/env python3
"""Standalone training script for Spatio-Temporal Progress Head (ProgressHeadST).

Trains on cached raw DINOv3 wrist patch tokens with PCHIP progress targets.
Keeps policy training and progress head completely separate.
"""

import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import zarr

# Ensure maniflow is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from maniflow.model.progress.progress_head_st import ProgressHeadST


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 42) -> np.ndarray:
    """Split episodes into train and val masks (whole episode level)."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class CachedProgressDataset(Dataset):
    """Dataset sampling (t, t-s, t-2s, t-3s) from cached DINOv3 token arrays."""

    def __init__(
        self,
        tokens_array,
        progress_array,
        episode_ends: np.ndarray,
        episode_mask: np.ndarray,
        is_train: bool = True,
        fixed_stride: int = 4,
    ):
        super().__init__()
        self.tokens_array = tokens_array
        self.progress_array = progress_array
        self.episode_ends = episode_ends
        self.is_train = is_train
        self.fixed_stride = fixed_stride

        # Build list of valid (global_frame_idx, ep_start_idx)
        self.frame_indices = []
        for ep_idx, keep in enumerate(episode_mask):
            if not keep:
                continue
            ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            ep_end = int(episode_ends[ep_idx])
            for t in range(ep_start, ep_end):
                self.frame_indices.append((t, ep_start))

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int):
        t, ep_start = self.frame_indices[idx]

        # Stride jitter during training: s in {3, 4, 5}; fixed during val
        s = random.randint(3, 5) if self.is_train else self.fixed_stride

        # Clamping at episode start (t < 3s clamped to ep_start)
        t0 = t
        t1 = max(ep_start, t - s)
        t2 = max(ep_start, t - 2 * s)
        t3 = max(ep_start, t - 3 * s)

        # (4, N_p, D) in float32
        tokens_4 = self.tokens_array[[t0, t1, t2, t3]].astype(np.float32)

        # Target in [0, 1]
        target = float(self.progress_array[t])

        return torch.from_numpy(tokens_4), torch.tensor([target], dtype=torch.float32)


def evaluate(model, dataloader, device):
    """Evaluate val_loss, val_mae, and val_fpr_09 (false positive rate)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for tokens, targets in dataloader:
            tokens = tokens.to(device)
            targets = targets.to(device)

            preds = model(tokens)
            loss = model.compute_loss(preds, targets)

            total_loss += loss.item() * tokens.size(0)
            total_mae += torch.abs(preds - targets).sum().item()

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    n_samples = len(dataloader.dataset)
    val_loss = total_loss / n_samples
    val_mae = total_mae / n_samples

    all_preds = torch.cat(all_preds, dim=0).squeeze(-1)
    all_targets = torch.cat(all_targets, dim=0).squeeze(-1)

    # False Positive Rate: Proportion of frames with Ground-Truth < 0.7 where pred >= 0.9
    early_mask = (all_targets < 0.7)
    if early_mask.any():
        fpr_09 = (all_preds[early_mask] >= 0.9).float().mean().item()
    else:
        fpr_09 = 0.0

    return val_loss, val_mae, fpr_09


def plot_validation_episodes(
    model: nn.Module,
    tokens_arr,
    progress_arr,
    episode_ends: np.ndarray,
    val_mask: np.ndarray,
    save_path: str,
    device: torch.device,
    num_episodes: int = 4,
):
    """Plot ground truth curve vs model predictions over time for 3-4 validation episodes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    val_ep_indices = np.nonzero(val_mask)[0]
    if len(val_ep_indices) == 0:
        return

    selected_eps = val_ep_indices[:min(num_episodes, len(val_ep_indices))]
    n_plots = len(selected_eps)

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]

    stride = 4
    with torch.no_grad():
        for i, ep_idx in enumerate(selected_eps):
            ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            ep_end = int(episode_ends[ep_idx])
            ep_len = ep_end - ep_start

            frames_tokens = []
            gt_curve = []
            for t in range(ep_start, ep_end):
                t0 = t
                t1 = max(ep_start, t - stride)
                t2 = max(ep_start, t - 2 * stride)
                t3 = max(ep_start, t - 3 * stride)
                frames_tokens.append(tokens_arr[[t0, t1, t2, t3]].astype(np.float32))
                gt_curve.append(float(progress_arr[t]))

            inp_tensor = torch.from_numpy(np.stack(frames_tokens, axis=0)).to(device)
            pred_chunks = []
            for b in range(0, inp_tensor.size(0), 128):
                pred_chunks.append(model(inp_tensor[b:b + 128]).cpu())
            preds = torch.cat(pred_chunks, dim=0).squeeze(-1).numpy()
            gts = np.array(gt_curve)

            ax = axes[i]
            time_steps = np.arange(ep_len)
            ax.plot(time_steps, gts, label="Ground Truth (PCHIP)", color="#1f77b4", linewidth=2.0)
            ax.plot(time_steps, preds, label="Prediction (ProgressHeadST)", color="#d62728", linestyle="--", linewidth=2.0)
            ax.axhline(0.9, color="gray", linestyle=":", label="Threshold 0.90")
            ax.axhline(0.7, color="lightgray", linestyle=":", label="Threshold 0.70")

            mae = float(np.mean(np.abs(preds - gts)))
            ax.set_title(f"Val Episode {ep_idx} (Len: {ep_len} frames | MAE: {mae:.3f})", fontsize=11)
            ax.set_ylabel("Progress")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper left")
            if i == n_plots - 1:
                ax.set_xlabel("Timestep (frames)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Train] Validation plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Spatio-Temporal ProgressHeadST on cached features.")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to Zarr dataset containing cached features.")
    parser.add_argument("--token_key", type=str, default="wrist_dinov3_tokens", help="Key for cached token array.")
    parser.add_argument("--progress_key", type=str, default="progress", help="Key for PCHIP progress target.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of episodes held out for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay.")
    parser.add_argument("--noise_ratio", type=float, default=0.02, help="Relative Gaussian feature noise std ratio.")
    parser.add_argument("--patch_dropout", type=float, default=0.3, help="Patch token masking ratio.")
    parser.add_argument("--save_dir", type=str, default="checkpoints/progress_head", help="Directory to save checkpoints.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Train] Loading Zarr from: {args.zarr_path}")
    root = zarr.open(args.zarr_path, mode='r')
    group = root['data'] if 'data' in root else root

    if args.token_key not in group:
        raise KeyError(f"Tokens '{args.token_key}' not found in {args.zarr_path}. Run cache_wrist_dinov3.py first.")
    if args.progress_key not in group:
        raise KeyError(f"Progress labels '{args.progress_key}' not found in {args.zarr_path}.")

    tokens_arr = group[args.token_key]
    progress_arr = group[args.progress_key]
    episode_ends = root['meta/episode_ends'][:] if 'meta/episode_ends' in root else group['meta/episode_ends'][:]

    n_episodes = len(episode_ends)
    val_mask = get_val_mask(n_episodes, args.val_ratio, seed=args.seed)
    train_mask = ~val_mask

    train_ds = CachedProgressDataset(tokens_arr, progress_arr, episode_ends, train_mask, is_train=True)
    val_ds = CachedProgressDataset(tokens_arr, progress_arr, episode_ends, val_mask, is_train=False)

    print(f"[Train] Split: {n_episodes} episodes -> {np.sum(train_mask)} train ({len(train_ds)} frames), "
          f"{np.sum(val_mask)} val ({len(val_ds)} frames)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    N_p, D = tokens_arr.shape[1], tokens_arr.shape[2]
    print(f"[Train] Creating ProgressHeadST with N_p={N_p}, D={D}...")
    model = ProgressHeadST(
        feature_dim=D,
        patch_dropout=args.patch_dropout,
        noise_ratio=args.noise_ratio,
        loss_type="smooth_l1",
        beta=0.05,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameter count: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_val_loss = float('inf')
    best_fpr_09 = float('inf')
    best_path = os.path.join(args.save_dir, "best_progress_head.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for tokens, targets in pbar:
            tokens = tokens.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(tokens)
            loss = model.compute_loss(preds, targets)
            loss.backward()
            optimizer.step()

            mae = torch.abs(preds - targets).mean().item()
            train_loss += loss.item() * tokens.size(0)
            train_mae += mae * tokens.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mae": f"{mae:.4f}"})

        scheduler.step()
        train_loss /= len(train_ds)
        train_mae /= len(train_ds)

        # Validation
        val_loss, val_mae, fpr_09 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}: "
              f"Train Loss={train_loss:.4f}, MAE={train_mae:.4f} | "
              f"Val Loss={val_loss:.4f}, MAE={val_mae:.4f}, FPR(>=0.9 on GT<0.7)={fpr_09*100:.2f}%")

        # Save best model based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_fpr_09 = fpr_09
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_fpr_09': fpr_09,
                'feature_dim': D,
                'num_patches': N_p,
            }, best_path)
            print(f"  --> Saved new best checkpoint to {best_path} (Val Loss: {val_loss:.4f}, FPR_09: {fpr_09*100:.2f}%)")

    # Save latest
    latest_path = os.path.join(args.save_dir, "latest_progress_head.pt")
    torch.save(model.state_dict(), latest_path)

    # Load best checkpoint to generate validation plot
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])

    val_plot_path = os.path.join(args.save_dir, "val_predictions.png")
    plot_validation_episodes(
        model=model,
        tokens_arr=tokens_arr,
        progress_arr=progress_arr,
        episode_ends=episode_ends,
        val_mask=val_mask,
        save_path=val_plot_path,
        device=device,
        num_episodes=4,
    )

    print(f"\n[Train] Finished! Best Val Loss: {best_val_loss:.4f} with FPR_09={best_fpr_09*100:.2f}%. Saved to {best_path}")


if __name__ == "__main__":
    main()
