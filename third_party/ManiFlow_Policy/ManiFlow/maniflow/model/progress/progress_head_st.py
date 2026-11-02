"""Spatio-Temporal Progress Head for DINOv3 Wrist Features.

Processes frozen detached patch tokens F_t in R^{B x K x N_p x D} from the wrist camera.
K = 4 frames: current frame t (index 0) and past frames t-s, t-2s, t-3s with stride s.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    """(M,) positions -> (M, dim) sin-cos embedding."""
    omega = torch.arange(dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000.0 ** (omega / (dim / 2.0)))
    out = pos.float()[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _sincos_2d(dim: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """Factorised 2D sin-cos PE over an (h, w) raster -> (h*w, dim)."""
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4, got {dim}")
    grid_h, grid_w = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij')
    emb_h = _sincos_1d(dim // 2, grid_h.reshape(-1))
    emb_w = _sincos_1d(dim // 2, grid_w.reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1).to(dtype)


class ProgressHeadST(nn.Module):
    """Spatio-Temporal Progress Head based on DINOv3 Wrist Patch Tokens.

    Architecture:
      Input: F in R^{B x 4 x N_p x D} (detached patch tokens)
      1. Bottleneck + 2D-PE:
         Z = Linear(D -> 64) -> LayerNorm(64) -> GELU + grid_pe (B, 4, N_p, 64)
      2. Latent differences & Fusion:
         Z_curr = Z[:, 0, :, :]
         deltas = [Z_curr - Z[:, k, :, :] for k in 1..3]
         U = concat([Z_curr] + deltas, dim=-1)  # (B, N_p, 256)
         M = Linear(256 -> 64) -> LayerNorm(64) -> GELU -> Dropout(0.2)
      3. Augmentation & Token-Dropout:
         During training: Relative Gaussian noise on F:
           F = F + randn_like(F) * noise_ratio * F.std(dim=-1, keepdim=True)
         30% patch masking via key_padding_mask in Cross-Attention.
      4. Attention Pooling:
         query_tokens = Parameter(1, 2, 64)
         attn_out, _ = MultiheadAttention(64, num_heads=4)(query=q, key=M, value=M)
         H = LayerNorm(q + attn_out)  # (B, 2, 64)
      5. Readout-MLP:
         Flatten -> Linear(128 -> 64) -> LN -> GELU -> Dropout(0.2) -> Linear(64 -> 1) -> Sigmoid
    """

    def __init__(
        self,
        feature_dim: int = 384,
        patch_grid: Optional[Tuple[int, int]] = (16, 16),
        num_queries: int = 2,
        dropout: float = 0.2,
        patch_dropout: float = 0.3,
        noise_ratio: float = 0.02,
        loss_type: str = "smooth_l1",
        beta: float = 0.05,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_grid = patch_grid
        self.num_queries = num_queries
        self.dropout = dropout
        self.patch_dropout = patch_dropout
        self.noise_ratio = noise_ratio
        self.loss_type = loss_type
        self.beta = beta

        # 1. Bottleneck projection (D -> 64)
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 2. Per-patch fusion: Z_curr (64) + 3 deltas (3*64 = 192) = 256 -> 64
        self.fusion = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Attention-pooling over query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, 64) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(64)

        # 5. Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(num_queries * 64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Cache for parameter-free grid PE: (gh, gw, dtype, device) -> tensor
        self._grid_pe_cache = {}

    def _get_grid_pe(self, num_patches: int, device, dtype) -> torch.Tensor:
        """Get or compute parameter-free 2D sin-cos grid PE of dimension 64."""
        if self.patch_grid is not None and self.patch_grid[0] * self.patch_grid[1] == num_patches:
            gh, gw = self.patch_grid
        else:
            side = int(math.isqrt(num_patches))
            if side * side == num_patches:
                gh, gw = side, side
            else:
                gh, gw = 12, 16  # standard 192 patches fallback
                if gh * gw != num_patches:
                    raise ValueError(f"Cannot infer 2D grid for num_patches={num_patches}. Provide patch_grid.")

        cache_key = (gh, gw, dtype, str(device))
        pe = self._grid_pe_cache.get(cache_key)
        if pe is None:
            pe = _sincos_2d(64, gh, gw, device, dtype)
            self._grid_pe_cache[cache_key] = pe
        return pe

    def forward(self, F_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            F_tokens: (B, 4, N_p, D) raw patch tokens. Detached from policy.
                      Index 0: frame t
                      Index 1: frame t - s
                      Index 2: frame t - 2s
                      Index 3: frame t - 3s
        Returns:
            progress: (B, 1) scalar progress estimate in [0, 1].
        """
        F_tokens = F_tokens.detach()

        B, K, N_p, D = F_tokens.shape
        if K != 4:
            raise ValueError(f"Expected K=4 frames, got K={K}")

        # Step 3 (part 1): Relative Gaussian noise during training
        if self.training and self.noise_ratio > 0:
            std = F_tokens.std(dim=-1, keepdim=True).clamp_min(1e-6)
            F_tokens = F_tokens + torch.randn_like(F_tokens) * (self.noise_ratio * std)

        # Step 1: Bottleneck projection + 2D Sin-Cos PE
        # (B, 4, N_p, D) -> (B, 4, N_p, 64)
        Z = self.bottleneck(F_tokens)
        grid_pe = self._get_grid_pe(N_p, Z.device, Z.dtype)  # (N_p, 64)
        Z = Z + grid_pe.unsqueeze(0).unsqueeze(0)  # broadcast over B and K

        # Step 2: Latent differences & Fusion
        Z_curr = Z[:, 0, :, :]  # (B, N_p, 64)
        deltas = [Z_curr - Z[:, k, :, :] for k in range(1, 4)]  # 3 x (B, N_p, 64)

        # Concatenate along channel axis: (B, N_p, 64 + 3*64 = 256)
        U = torch.cat([Z_curr] + deltas, dim=-1)

        # Per-patch fusion: (B, N_p, 64)
        M = self.fusion(U)

        # Step 3 (part 2): Patch Token-Dropout (never drop before deltas!)
        key_padding_mask = None
        if self.training and self.patch_dropout > 0:
            key_padding_mask = torch.rand(B, N_p, device=M.device) < self.patch_dropout
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)
            key_padding_mask = key_padding_mask & (~all_masked)

        # Step 4: Attention-Pooling via Query Tokens
        q = self.query_tokens.expand(B, -1, -1)  # (B, 2, 64)
        attn_out, _ = self.cross_attn(
            query=q,
            key=M,
            value=M,
            key_padding_mask=key_padding_mask,
        )
        H = self.norm_attn(q + attn_out)  # (B, 2, 64)

        # Step 5: Readout-MLP
        H_flat = H.reshape(B, self.num_queries * 64)  # (B, 128)
        progress = self.readout(H_flat)  # (B, 1)

        return progress

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss against ground truth progress target.

        Args:
            pred: (B, 1) predicted progress in [0, 1]
            target: (B, 1) or (B,) ground truth PCHIP progress in [0, 1]
        Returns:
            scalar loss tensor
        """
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        target = target.to(dtype=pred.dtype, device=pred.device)

        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, beta=self.beta)
        elif self.loss_type == "mse":
            return F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
