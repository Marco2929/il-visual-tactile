"""
vision_memory.py

Eigenständiges RGB-Vision-Memory-Modul.
- Encodiert RGB-Historie mit eingefrorenem DINOv2
- Ringbuffer mit Feature-Caching (Rollout: nur neuer Frame wird encodet)
- Gibt Memory-Tokens [B, T*P, D] zurück -> als Key/Value fuer eigene Policy

Die Einspeisung in die Policy (Cross-Attention etc.) macht der Nutzer selbst.
"""
from collections import deque
from typing import Optional

import torch
import torch.nn as nn


class DINOv2FrameEncoder(nn.Module):
    """Encodiert EINEN RGB-Frame -> Patch-Tokens. Backbone eingefroren."""

    def __init__(self, out_dim: int, model_name: str = "dinov2_vitb14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embed_dim = self.backbone.embed_dim
        # trainierbare Projektion in den gewuenschten Feature-Raum
        self.proj = nn.Linear(self.embed_dim, out_dim)

    @torch.no_grad()
    def _backbone_tokens(self, img: torch.Tensor) -> torch.Tensor:
        # img: [B, 3, H, W]  (H, W Vielfache von 14 fuer ViT-14)
        feats = self.backbone.forward_features(img)
        return feats["x_norm_patchtokens"]          # [B, P, embed_dim]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        tok = self._backbone_tokens(img)            # [B, P, embed_dim] (no grad)
        return self.proj(tok)                       # [B, P, out_dim]   (trainierbar)


class VisionMemory(nn.Module):
    """
    Verwaltet die RGB-Historie und liefert Memory-Tokens.

    Zwei Betriebsarten:
      - forward_sequence(...) : Training, encodiert komplette Sequenz auf einmal
      - reset() + step(...)   : Rollout, Ringbuffer + Feature-Caching
    """

    def __init__(
        self,
        out_dim: int = 512,
        horizon: int = 8,
        model_name: str = "dinov2_vitb14",
        use_temporal_embedding: bool = True,
    ):
        super().__init__()
        self.encoder = DINOv2FrameEncoder(out_dim=out_dim, model_name=model_name)
        self.out_dim = out_dim
        self.horizon = horizon

        # lernbares Zeit-Embedding: Index 0 = aeltester, horizon-1 = neuester
        self.use_temporal_embedding = use_temporal_embedding
        if use_temporal_embedding:
            self.temporal_embed = nn.Parameter(torch.zeros(horizon, out_dim))
            nn.init.normal_(self.temporal_embed, std=0.02)

        # Ringbuffer fuer den Rollout (haelt bereits encodete Tokens)
        self._buffer: deque = deque(maxlen=horizon)

    # ---------- Hilfsfunktion: Zeit-Embedding anhaengen ----------
    def _add_temporal(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, P, D]
        if not self.use_temporal_embedding:
            return tokens
        T = tokens.shape[1]
        # nimm die letzten T Zeit-Slots (neueste Frames -> hoehere Indizes)
        te = self.temporal_embed[-T:]               # [T, D]
        return tokens + te.view(1, T, 1, self.out_dim)

    # ---------- Training: ganze Sequenz auf einmal ----------
    def forward_sequence(self, rgb_hist: torch.Tensor) -> torch.Tensor:
        """
        rgb_hist: [B, T, 3, H, W]  (T <= horizon, chronologisch: alt -> neu)
        return:   [B, T*P, D]  Memory-Tokens (Key/Value)
        """
        B, T = rgb_hist.shape[:2]
        flat = rgb_hist.flatten(0, 1)               # [B*T, 3, H, W]
        tok = self.encoder(flat)                    # [B*T, P, D]
        P = tok.shape[1]
        tok = tok.view(B, T, P, self.out_dim)       # [B, T, P, D]
        tok = self._add_temporal(tok)               # + Zeit
        return tok.reshape(B, T * P, self.out_dim)  # [B, T*P, D]

    # ---------- Rollout: Ringbuffer mit Caching ----------
    @torch.no_grad()
    def reset(self):
        """Vor jedem Episode-Start aufrufen."""
        self._buffer.clear()

    @torch.no_grad()
    def step(self, rgb_frame: torch.Tensor) -> torch.Tensor:
        """
        Fuegt EINEN neuen Frame hinzu (wird einmal encodet) und gibt den
        aktuellen Memory-Zustand zurueck.

        rgb_frame: [B, 3, H, W]
        return:    [B, t*P, D]   (t = aktuell gefuellte Buffer-Laenge)
        """
        tok = self.encoder(rgb_frame)               # [B, P, D]  einmalig encodet
        self._buffer.append(tok)                    # aeltester faellt bei maxlen raus

        stacked = torch.stack(list(self._buffer), dim=1)   # [B, t, P, D]
        stacked = self._add_temporal(stacked)              # + Zeit
        B, t, P, D = stacked.shape
        return stacked.reshape(B, t * P, D)                # [B, t*P, D]

    # ---------- Info fuer die Policy-Seite ----------
    def output_dim(self) -> int:
        return self.out_dim


if __name__ == "__main__":
    B, T, H, W = 2, 8, 224, 224
    mem = VisionMemory(out_dim=512, horizon=T)

    # --- Training: ganze Sequenz ---
    rgb_hist = torch.randn(B, T, 3, H, W)
    kv = mem.forward_sequence(rgb_hist)
    print("train memory tokens:", kv.shape)          # [B, T*P, 512]

    # --- Rollout: Frame fuer Frame ---
    mem.reset()
    for i in range(3):
        frame = torch.randn(B, 3, H, W)
        kv_live = mem.step(frame)
        print(f"rollout step {i}:", kv_live.shape)    # waechst bis T*P