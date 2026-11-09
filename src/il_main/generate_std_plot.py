#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

json_path = "/home/mm/dev/il-visual-tactile/data/prediction_std_comparison.json"
with open(json_path) as f:
    data = json.load(f)

models = [
    '0016_abs_wrist_only', '0015_abs_pc_wrist', '0017_abs_wrist_front',
    '0020_rel_wrist_only', '0019_rel_pc_wrist', '0021_rel_wrist_front',
    'old_abs_0112', 'old_rel_0112'
]

model_labels = [
    '0016: Abs Wrist', '0015: Abs PC+Wrist', '0017: Abs Wrist+Front',
    '0020: Rel Wrist', '0019: Rel PC+Wrist', '0021: Rel Wrist+Front',
    'Old Abs (0112)', 'Old Rel (0112)'
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1. Overall Sample Std (All Dims)
ax1 = axes[0]
sample_stds = [data[m]['all_dims']['sample_std'] for m in models]
colors1 = ['#2b5c8f', '#d95f02', '#7570b3', '#386cb0', '#e6ab02', '#e7298a', '#999999', '#666666']
bars1 = ax1.bar(range(len(models)), sample_stds, color=colors1, width=0.6, edgecolor='black', linewidth=0.8)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9.5)
ax1.set_title('Prediction Spread: Sample STD (All Dimensions)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Sample STD (Normalized Action Space)', fontsize=11)
ax1.set_ylim(0, 0.045)
ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
for b in bars1:
    h = b.get_height()
    ax1.annotate(f'{h:.4f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# 2. Per-Axis STD breakdown (XY, Z, Rot, Hand)
ax2 = axes[1]
groups = ['xy', 'z', 'rot', 'hand']
group_labels = ['XY (Lateral)', 'Z (Height)', 'Rotation', 'Hand (Pinch)']
x_p = np.arange(len(groups))
w = 0.18

# Compare 4 representative models: 0016 (Abs Wrist), 0020 (Rel Wrist), 0015 (Abs PC+Wrist), 0019 (Rel PC+Wrist)
sel_models = ['0016_abs_wrist_only', '0020_rel_wrist_only', '0015_abs_pc_wrist', '0019_rel_pc_wrist']
sel_labels = ['0016 (Abs Wrist)', '0020 (Rel Wrist)', '0015 (Abs PC+Wrist)', '0019 (Rel PC+Wrist)']
sel_colors = ['#2b5c8f', '#386cb0', '#d95f02', '#e6ab02']

for idx, (m, lbl, col) in enumerate(zip(sel_models, sel_labels, sel_colors)):
    stds = [data[m]['axis_groups'][g]['std'] for g in groups]
    ax2.bar(x_p + (idx - 1.5)*w, stds, width=w, label=lbl, color=col, edgecolor='black', linewidth=0.7)

ax2.set_xticks(x_p)
ax2.set_xticklabels(group_labels, fontsize=10)
ax2.set_title('Sample STD by Action Axis Group', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sample STD (Normalized)', fontsize=11)
ax2.set_ylim(0, 0.042)
ax2.legend(frameon=True, fontsize=9.5)
ax2.grid(True, axis='y', linestyle=':', alpha=0.7)

# 3. Decision Window (Search Phase) XY STD & Multimodality Gain
ax3 = axes[2]
search_stds = [data[m]['phases_xy']['search']['std'] for m in models]
search_gains = [data[m]['phases_xy']['search']['gain'] for m in models]

bars3 = ax3.bar(range(len(models)), search_stds, color=colors1, width=0.6, edgecolor='black', linewidth=0.8)
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9.5)
ax3.set_title('Decision Window (Search Phase): XY STD', fontsize=12, fontweight='bold')
ax3.set_ylabel('Search Phase XY STD', fontsize=11)
ax3.set_ylim(0, 0.05)
ax3.grid(True, axis='y', linestyle=':', alpha=0.7)

for b, g in zip(bars3, search_gains):
    h = b.get_height()
    ax3.annotate(f'{h:.3f}\n({g:.2f}x)', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
out_plot = "/home/mm/dev/il-visual-tactile/data/rollouts/analysis/investigations/modality_ablation/plots/prediction_std_multimodality_analysis.png"
brain_plot = "/home/mm/.gemini/antigravity/brain/93b2b22b-696f-4543-8358-70a30cd58288/plots/prediction_std_multimodality_analysis.png"
os.makedirs(os.path.dirname(out_plot), exist_ok=True)
os.makedirs(os.path.dirname(brain_plot), exist_ok=True)
plt.savefig(out_plot, dpi=200)
plt.savefig(brain_plot, dpi=200)
plt.close()
print("Saved plots to:", out_plot, brain_plot)
