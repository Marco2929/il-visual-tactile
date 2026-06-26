import torch.nn as nn


def force_mlp(n_obs_steps: int, force_cond_len: int, encoder_output_dim: int):
            # MLP Variante (Flatten)
            # Erwartet Input: (Batch, Time, Channels)
            return nn.Sequential(
                nn.Flatten(), # Macht aus (Batch, n_obs_steps, 65) -> (Batch, n_obs_steps * 65)
                nn.Linear(n_obs_steps * force_cond_len, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, encoder_output_dim)
            )

def force_cnn(force_cond_len: int, encoder_output_dim: int):
        # Wir nutzen Padding='same' (padding=1 bei kernel=3), 
            # damit die Zeit-Länge (z.B. 16) exakt erhalten bleibt.
        return nn.Sequential(
            nn.Conv1d(in_channels=force_cond_len, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=encoder_output_dim, kernel_size=3, padding=1),
        )