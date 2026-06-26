import torch
import torch.nn as nn


class ProgressHead(nn.Module):
    """
    Progress prediction head (Regression baseline).
    Supports vision-only or vision + force input.
    """
    def __init__(self, visual_feat_dim=256, force_feat_dim=256, hidden_dim=128, use_force=False):
        super().__init__()
        self.use_force = use_force

        input_dim = visual_feat_dim
        if use_force:
            input_dim += force_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_tokens, force_tokens=None):
        """
        Args:
            visual_tokens: (B, N_vis, D)
            force_tokens:  (B, N_force, D) or None
        Returns:
            progress: (B, 1) ∈ [0, 1]
        """
        pooled = visual_tokens.mean(dim=1)  # (B, D)

        if self.use_force:
            assert force_tokens is not None, "force_tokens required when use_force=True"
            force_pooled = force_tokens.mean(dim=1)  # (B, D)
            pooled = torch.cat([pooled, force_pooled], dim=-1)

        return self.mlp(pooled)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        MSE loss for progress regression.
        Args:
            pred:   (B, 1) predicted progress
            target: (B, 1) ground-truth progress ∈ [0, 1]
        """
        return nn.functional.mse_loss(pred, target)