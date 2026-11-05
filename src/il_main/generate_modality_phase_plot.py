#!/usr/bin/env python3
"""Plot modality importance across task phases for the key models."""

import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

OUT_PLOT = "data/rollout_logs/investigations/modality_ablation/plots/modality_phase_distribution.png"
ARTIFACT_PLOT = "/home/mm/.gemini/antigravity/brain/93b2b22b-696f-4543-8358-70a30cd58288/plots/modality_phase_distribution.png"

# Phases
phases = ["Approach (Anfahrt)", "Search (Entscheidung)", "Transition (Greifen)", "Hold (Heben)"]
x = np.arange(len(phases))
width = 0.22

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

# 1. Model 0016: Abs Wrist Only
# Shuffle ratios (x full error)
wrist_0016 = [2.69, 2.56, 5.77, 1.38]
state_0016 = [1.53, 1.47, 6.79, 1.25]
ref_0016 = [1.0, 1.0, 1.0, 1.0]

ax1.bar(x - width/2, wrist_0016, width, label="Shuffle Wrist RGB (Vision)", color="#1f77b4", alpha=0.9)
ax1.bar(x + width/2, state_0016, width, label="Shuffle Proprio (State)", color="#ff7f0e", alpha=0.9)
ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="Reference (Full Obs = 1.0x)")
ax1.set_title("Run 0016 (Abs: Wrist Only)\nTop-Performer Absolut", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(phases, rotation=15, ha="right", fontsize=9)
ax1.set_ylabel("Error Multiplier (x Full Baseline MSE)", fontsize=11)
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="upper left", fontsize=9)
ax1.set_ylim(0, 7.5)

# 2. Model 0020: Rel Wrist Only
wrist_0020 = [2.40, 1.05, 0.91, 2.21]  # search XY error without wrist is 3.55x
state_0020 = [1.29, 1.03, 1.38, 1.40]

ax2.bar(x - width/2, wrist_0020, width, label="Shuffle Wrist RGB (Vision)", color="#2ca02c", alpha=0.9)
ax2.bar(x + width/2, state_0020, width, label="Shuffle Proprio (State)", color="#d62728", alpha=0.9)
ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
ax2.set_title("Run 0020 (Rel: Wrist Only)\nTop-Performer Relativ", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(phases, rotation=15, ha="right", fontsize=9)
ax2.grid(axis="y", linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", fontsize=9)

# 3. Model 0015: Abs PC + Wrist
w_0015 = 0.20
wrist_0015 = [2.05, 0.85, 0.52, 2.08]
state_0015 = [1.27, 1.01, 1.20, 2.12]
pc_0015    = [1.01, 1.00, 0.85, 1.02]  # but search XY error without PC explodes to 3.18x!

ax3.bar(x - w_0015, wrist_0015, w_0015, label="Shuffle Wrist RGB", color="#1f77b4", alpha=0.9)
ax3.bar(x, state_0015, w_0015, label="Shuffle Proprio", color="#ff7f0e", alpha=0.9)
ax3.bar(x + w_0015, pc_0015, w_0015, label="Shuffle Point Cloud", color="#9467bd", alpha=0.9)
ax3.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
ax3.set_title("Run 0015 (Abs: Wrist + Point Cloud)\nMultimodal Baseline", fontsize=12, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(phases, rotation=15, ha="right", fontsize=9)
ax3.grid(axis="y", linestyle=":", alpha=0.6)
ax3.legend(loc="upper left", fontsize=9)

# Annotations for key takeaways
ax1.annotate("Wrist & State\nbeide kritisch (6x)", xy=(2, 6.0), xytext=(1.5, 6.6),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=8, fontweight="bold")
ax2.annotate("Hold ohne Vision:\n17.7x Error", xy=(3, 2.21), xytext=(2.3, 3.8),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=8, fontweight="bold")
ax3.annotate("Decision XY ohne PC:\n3.18x lateral error", xy=(1, 1.0), xytext=(0.6, 2.8),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=8, fontweight="bold")

fig.suptitle("Modality Importance by Task Phase (Shuffle Ablation across 4 Task Phases)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
plt.savefig(OUT_PLOT, dpi=200, bbox_inches="tight")
os.makedirs(os.path.dirname(ARTIFACT_PLOT), exist_ok=True)
plt.savefig(ARTIFACT_PLOT, dpi=200, bbox_inches="tight")
print(f"Saved phase distribution plot to {OUT_PLOT} and {ARTIFACT_PLOT}")

