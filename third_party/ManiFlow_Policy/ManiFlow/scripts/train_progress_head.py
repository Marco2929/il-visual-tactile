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
import yaml

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
    parser.add_argument("--config", type=str, default=None, help="Path to YAML training configuration file.")
    parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr dataset containing cached features.")
    parser.add_argument("--token_key", type=str, default=None, help="Key for cached token array.")
    parser.add_argument("--progress_key", type=str, default=None, help="Key for PCHIP progress target.")
    parser.add_argument("--val_ratio", type=float, default=None, help="Fraction of episodes held out for validation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for train/val split.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--noise_ratio", type=float, default=None, help="Relative Gaussian feature noise std ratio.")
    parser.add_argument("--patch_dropout", type=float, default=None, help="Patch token masking ratio.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints.")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader num_workers.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None, help="Save periodic checkpoint every N epochs (default: 10).")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=None, help="Enable Weights & Biases logging (default: True).")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name.")
    args = parser.parse_args()

    # Default config path if none passed
    default_cfg_path = os.path.join(repo_root, "maniflow", "config", "progress_head.yaml")
    cfg_path = args.config if args.config else (default_cfg_path if os.path.exists(default_cfg_path) else None)

    cfg = {}
    if cfg_path and os.path.exists(cfg_path):
        print(f"[Train] Loading config from: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})
    log_cfg = cfg.get("logging", {})

    zarr_path = args.zarr_path or data_cfg.get("zarr_path")
    if not zarr_path:
        raise ValueError("zarr_path must be specified either via --zarr_path or in YAML under data.zarr_path")

    token_key = args.token_key or data_cfg.get("token_key", "wrist_dinov3_tokens")
    progress_key = args.progress_key or data_cfg.get("progress_key", "progress")
    val_ratio = args.val_ratio if args.val_ratio is not None else data_cfg.get("val_ratio", 0.1)
    seed = args.seed if args.seed is not None else data_cfg.get("seed", 42)
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 4)

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 128)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 1e-3)
    weight_decay = args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 0.05)
    noise_ratio = args.noise_ratio if args.noise_ratio is not None else train_cfg.get("noise_ratio", 0.02)
    patch_dropout = args.patch_dropout if args.patch_dropout is not None else train_cfg.get("patch_dropout", 0.3)
    device_str = args.device or train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.save_dir or out_cfg.get("save_dir", "/ros2_ws/data/outputs/progress")
    save_every = args.save_every if args.save_every is not None else train_cfg.get("save_every", 10)

    # W&B setup (active by default)
    use_wandb = args.wandb if args.wandb is not None else log_cfg.get("use_wandb", True)
    wandb_project = args.wandb_project or log_cfg.get("project", "progress-head")
    wandb_name = log_cfg.get("name", None)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    "zarr_path": zarr_path,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "noise_ratio": noise_ratio,
                    "patch_dropout": patch_dropout,
                    "val_ratio": val_ratio,
                    "seed": seed,
                }
            )
            print(f"[Train] Initialized W&B run: {wandb_run.name} (Project: {wandb_project})")
        except Exception as e:
            print(f"Warning: Failed to initialize W&B ({e}). Continuing without W&B.")
            wandb_run = None

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Train] Loading Zarr from: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    group = root['data'] if 'data' in root else root

    if token_key not in group:
        raise KeyError(f"Tokens '{token_key}' not found in {zarr_path}. Run cache_wrist_dinov3.py first.")
    if progress_key not in group:
        raise KeyError(f"Progress labels '{progress_key}' not found in {zarr_path}.")

    tokens_arr = group[token_key]
    progress_arr = group[progress_key]
    episode_ends = root['meta/episode_ends'][:] if 'meta/episode_ends' in root else group['meta/episode_ends'][:]

    in_memory = data_cfg.get("in_memory", True)
    if in_memory:
        import time
        t_load = time.time()
        print(f"[Train] Preloading {tokens_arr.shape} ({tokens_arr.nbytes / 1e9:.1f} GB) tokens into RAM for instant DataLoader access...")
        tokens_arr = tokens_arr[:]
        progress_arr = progress_arr[:]
        print(f"[Train] Preloading complete in {time.time() - t_load:.1f}s! In-memory RAM cache active.")

    n_episodes = len(episode_ends)
    val_mask = get_val_mask(n_episodes, val_ratio, seed=seed)
    train_mask = ~val_mask

    train_ds = CachedProgressDataset(tokens_arr, progress_arr, episode_ends, train_mask, is_train=True)
    val_ds = CachedProgressDataset(tokens_arr, progress_arr, episode_ends, val_mask, is_train=False)

    print(f"[Train] Split: {n_episodes} episodes -> {np.sum(train_mask)} train ({len(train_ds)} frames), "
          f"{np.sum(val_mask)} val ({len(val_ds)} frames)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    N_p, D = tokens_arr.shape[1], tokens_arr.shape[2]
    print(f"[Train] Creating ProgressHeadST with N_p={N_p}, D={D}...")
    model = ProgressHeadST(
        feature_dim=D,
        patch_dropout=patch_dropout,
        noise_ratio=noise_ratio,
        num_queries=model_cfg.get("num_queries", 2),
        dropout=model_cfg.get("dropout", 0.2),
        loss_type=model_cfg.get("loss_type", "smooth_l1"),
        beta=model_cfg.get("beta", 0.05),
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameter count: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    best_val_loss = float('inf')
    best_fpr_09 = float('inf')
    best_path = os.path.join(save_dir, "best_progress_head.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
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

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/mae": train_mae,
                "val/loss": val_loss,
                "val/mae": val_mae,
                "val/fpr_09": fpr_09,
            })

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

        # Periodic checkpoint every save_every epochs
        if save_every and epoch % save_every == 0:
            epoch_ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_fpr_09': fpr_09,
                'feature_dim': D,
                'num_patches': N_p,
            }, epoch_ckpt_path)
            print(f"  --> Saved periodic checkpoint to {epoch_ckpt_path}")

            # Also generate and log periodic validation plot
            epoch_plot_path = os.path.join(save_dir, f"val_predictions_epoch_{epoch:03d}.png")
            plot_validation_episodes(
                model=model,
                tokens_arr=tokens_arr,
                progress_arr=progress_arr,
                episode_ends=episode_ends,
                val_mask=val_mask,
                save_path=epoch_plot_path,
                device=device,
                num_episodes=4,
            )
            if wandb_run is not None:
                wandb_run.log({f"val/predictions_epoch_{epoch:03d}": wandb.Image(epoch_plot_path)})

    # Save latest
    latest_path = os.path.join(save_dir, "latest_progress_head.pt")
    torch.save(model.state_dict(), latest_path)

    # Load best checkpoint to generate validation plot
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])

    val_plot_path = os.path.join(save_dir, "val_predictions.png")
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

    if wandb_run is not None:
        if os.path.exists(val_plot_path):
            import wandb
            wandb_run.log({"val/predictions_plot": wandb.Image(val_plot_path)})
        wandb_run.finish()

    print(f"\n[Train] Finished! Best Val Loss: {best_val_loss:.4f} with FPR_09={best_fpr_09*100:.2f}%. Saved to {best_path}")


if __name__ == "__main__":
    main()
