import torch
import torch.nn as nn


class VisionActionHead(nn.Module):
    """
    Vision-ONLY auxiliary action predictor.

    Forces the point-cloud encoder to encode all task-relevant spatial
    information (object location, geometry) by requiring it to predict the
    action WITHOUT proprioception. This directly counteracts the proprio
    shortcut that causes vision to "die" during training.

    Unlike the main flow model (which has state_cond), this head sees ONLY
    vis_cond -> cannot cheat via joint angles.
    """
    def __init__(self, visual_feat_dim, action_dim, horizon, hidden_dim=256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(visual_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(self, visual_tokens):
        """
        Args:
            visual_tokens: (B, N_vis, D)
        Returns:
            action_pred: (B, horizon, action_dim)
        """
        pooled = visual_tokens.mean(dim=1)          # (B, D)
        out = self.net(pooled)                       # (B, horizon*action_dim)
        return out.view(-1, self.horizon, self.action_dim)
