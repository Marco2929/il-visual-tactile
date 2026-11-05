#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_dir = "/home/mm/.gemini/antigravity/brain/93b2b22b-696f-4543-8358-70a30cd58288/plots"
os.makedirs(fig_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. PLOT 1: Learning Curves & Best MSE Matrix
# -------------------------------------------------------------
runs_data = {
    # Cleaned Recorded Action Absolute (proprio_dropout=0.8, 15D)
    '0015 (Abs PC+Wrist)': [
        (0, 0.129994), (10, 0.041263), (20, 0.043372), (30, 0.039922),
        (40, 0.040344), (50, 0.039610), (60, 0.039003), (70, 0.038337)
    ],
    '0016 (Abs Wrist only)': [
        (0, 0.127607), (10, 0.038602), (20, 0.034214), (30, 0.031528),
        (40, 0.027767), (50, 0.027129), (60, 0.026752), (70, 0.027055)
    ],
    '0017 (Abs Wrist+Front)': [
        (0, 0.130413), (10, 0.046331), (20, 0.047327), (30, 0.044313),
        (40, 0.040865), (50, 0.040045), (60, 0.039107), (70, 0.038920)
    ],
    # Cleaned Recorded Action Relative (proprio_dropout=0.0, 17D)
    '0019 (Rel PC+Wrist)': [
        (0, 0.126578), (10, 0.044613), (20, 0.045110), (30, 0.039892),
        (40, 0.038494), (50, 0.038166), (60, 0.037778), (70, 0.037454)
    ],
    '0020 (Rel Wrist only)': [
        (0, 0.122504), (10, 0.041972), (20, 0.039528), (30, 0.035851),
        (40, 0.033331), (50, 0.031996), (60, 0.031287), (70, 0.031641)
    ],
    '0021 (Rel Wrist+Front)': [
        (0, 0.131995), (10, 0.046116), (20, 0.063042), (30, 0.057131),
        (40, 0.054643), (50, 0.055313), (60, 0.055389), (70, 0.056586)
    ],
    # Old Baselines (uncleaned datasets)
    'Old Rel 0112 (Uncleaned)': [
        (0, 0.151875), (10, 0.030378), (20, 0.026271), (30, 0.023511),
        (40, 0.022836), (50, 0.021724), (60, 0.021500), (70, 0.021276)
    ],
    'Old Abs 0014 (Next-State)': [
        (0, 0.127516), (10, 0.028288), (20, 0.025412), (30, 0.023485),
        (40, 0.023737), (50, 0.023716), (60, 0.023961), (70, 0.023846)
    ]
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Subplot 1: Absolute Models
ax1 = axes[0]
for name in ['0016 (Abs Wrist only)', '0015 (Abs PC+Wrist)', '0017 (Abs Wrist+Front)', 'Old Abs 0014 (Next-State)']:
    data = runs_data[name]
    x, y = zip(*data)
    style = '--' if 'Old' in name else '-o'
    color = '#1f77b4' if 'Wrist only' in name else ('#ff7f0e' if 'PC+Wrist' in name else ('#2ca02c' if 'Front' in name else '#7f7f7f'))
    ax1.plot(x, y, style, label=name, color=color, linewidth=2.2, markersize=5)
ax1.set_title('Absolute Models: Validation Action MSE', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Val Action MSE', fontsize=11)
ax1.set_ylim(0.015, 0.14)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(frameon=True, fontsize=10)

# Subplot 2: Relative Models
ax2 = axes[1]
for name in ['0020 (Rel Wrist only)', '0019 (Rel PC+Wrist)', '0021 (Rel Wrist+Front)', 'Old Rel 0112 (Uncleaned)']:
    data = runs_data[name]
    x, y = zip(*data)
    style = '--' if 'Old' in name else '-s'
    color = '#1f77b4' if 'Wrist only' in name else ('#ff7f0e' if 'PC+Wrist' in name else ('#d62728' if 'Front' in name else '#7f7f7f'))
    ax2.plot(x, y, style, label=name, color=color, linewidth=2.2, markersize=5)
ax2.set_title('Relative Models: Validation Action MSE', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Val Action MSE', fontsize=11)
ax2.set_ylim(0.015, 0.14)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(frameon=True, fontsize=10)

# Subplot 3: Bar comparison of Best MSE
ax3 = axes[2]
labels = [
    'Abs: Wrist', 'Abs: PC+Wrist', 'Abs: Wrist+Front',
    'Rel: Wrist', 'Rel: PC+Wrist', 'Rel: Wrist+Front',
    'Old Abs (0014)', 'Old Rel (0112)'
]
bests = [
    0.026752, 0.038337, 0.038920,
    0.031287, 0.037454, 0.054643,
    0.023485, 0.021276
]
colors = [
    '#2b5c8f', '#d95f02', '#7570b3',
    '#386cb0', '#e6ab02', '#e7298a',
    '#999999', '#666666'
]
bars = ax3.bar(range(len(labels)), bests, color=colors, width=0.65, edgecolor='black', linewidth=0.8)
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9.5)
ax3.set_title('Best Validation Action MSE Comparison', fontsize=13, fontweight='bold')
ax3.set_ylabel('Best Val Action MSE (lower is better)', fontsize=11)
ax3.set_ylim(0, 0.065)
ax3.grid(True, axis='y', linestyle=':', alpha=0.7)

for b in bars:
    h = b.get_height()
    ax3.annotate(f'{h:.4f}',
                 xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
p1 = os.path.join(fig_dir, "ablation_matrix_learning_curves.png")
plt.savefig(p1, dpi=200)
plt.close()
print("Saved:", p1)


# -------------------------------------------------------------
# 2. PLOT 2: Rollout Evaluation (Old Absolute vs Old Relative)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 2.1: Success Rates by Condition
ax_succ = axes[0, 0]
conditions = [
    'Nominal\n(Standard)',
    'Lowered\n(-Z)',
    'Increased\n(+Z)',
    'Untere Hoehe\n(Bottom)',
    'Zwischenhoehe\n(Middle)',
    'Links/Mitte/Rechts\n(XY Offset)',
    'Overall\nSuccess'
]
# Rates:
# abs: Nominal 1/3 (33%), Lowered 0/3 (0%), Increased 0/3 (0%), Untere 3/3 (100%), Zwischen 0/3 (0%), XY 0/3 (0%), Overall 4/19 (21.1%)
# rel: Nominal 1/3 (33%), Lowered 0/3 (0%), Increased 0/3 (0%), Untere 2/3 (66.7%), Zwischen 2/3 (66.7%), XY 1/5 (20%), Overall 7/23 (30.4%)
abs_rates = [33.3, 0.0, 0.0, 100.0, 0.0, 0.0, 21.1]
rel_rates = [33.3, 0.0, 0.0, 66.7, 66.7, 20.0, 30.4]

x_idx = np.arange(len(conditions))
w = 0.35
b1 = ax_succ.bar(x_idx - w/2, abs_rates, width=w, label='Old Absolute (wrist_e40)', color='#1f77b4', edgecolor='black')
b2 = ax_succ.bar(x_idx + w/2, rel_rates, width=w, label='Old Relative (relative_e40)', color='#ff7f0e', edgecolor='black')

ax_succ.set_title('Real-Robot Test Success Rate by Condition', fontsize=12, fontweight='bold')
ax_succ.set_ylabel('Success Rate (%)', fontsize=11)
ax_succ.set_xticks(x_idx)
ax_succ.set_xticklabels(conditions, fontsize=9)
ax_succ.set_ylim(0, 115)
ax_succ.grid(True, axis='y', linestyle=':', alpha=0.6)
ax_succ.legend(frameon=True, fontsize=10)

for b in list(b1) + list(b2):
    h = b.get_height()
    if h > 0:
        ax_succ.annotate(f'{h:.1f}%',
                         xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Subplot 2.2: Grasp Timing Offset (Time from lowest Z to Closure Peak)
ax_time = axes[0, 1]
# Data from systematic_report.py
# Absolute offsets in seconds:
abs_offsets = [4.10, 3.80, 4.35, 3.65, 3.50, 2.95, 2.70, 3.80, 4.10, 3.60, 4.05, 3.55, 3.45, 3.35, 3.25, 3.65, 3.45, 3.60, 3.50]
# Relative offsets in seconds:
rel_offsets = [1.70, 7.37, 1.85, 17.90, 5.66, 4.60, 13.42, 5.70, 10.06, 5.21, 15.03, 7.05, 5.65, 4.10, 7.15, 42.07, 8.25, 6.25, 9.15, 12.14, 7.59, 4.80, 9.35]

box = ax_time.boxplot([abs_offsets, rel_offsets], tick_labels=['Old Absolute (wrist_e40)', 'Old Relative (relative_e40)'],
                      patch_artist=True, showmeans=True,
                      meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'})
colors_box = ['#1f77b4', '#ff7f0e']
for patch, color in zip(box['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_time.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Optimal Closure (at lowest Z)')
ax_time.set_title('Grasp Timing Lag: Closure Peak after Lowest Z (s)', fontsize=12, fontweight='bold')
ax_time.set_ylabel('Delay (seconds)', fontsize=11)
ax_time.set_ylim(-1, 25)
ax_time.grid(True, linestyle=':', alpha=0.6)
ax_time.legend(frameon=True, fontsize=10)
ax_time.annotate(f'Median: +{np.median(abs_offsets):.2f}s', xy=(1, np.median(abs_offsets)), xytext=(1.15, np.median(abs_offsets)+1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontweight='bold', fontsize=9.5)
ax_time.annotate(f'Median: +{np.median(rel_offsets):.2f}s\n(Extreme Lag)', xy=(2, np.median(rel_offsets)), xytext=(1.55, np.median(rel_offsets)+4),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontweight='bold', fontsize=9.5)

# Subplot 2.3: Pinch Closure Deficit (Filter Loss vs Mechanical Loss)
ax_pinch = axes[1, 0]
categories = ['Filter Loss\n(Butterworth/Clamp)', 'Mechanical Loss\n(Controller Stall)', 'Total Deficit\n(Angle)', 'Aperture Gap\n(Width in mm)']
abs_pinch = [0.8, 4.6, 5.4, 3.4]
rel_pinch = [0.7, 7.4, 8.1, 4.9]

x_p = np.arange(len(categories))
bp1 = ax_pinch.bar(x_p - w/2, abs_pinch, width=w, label='Old Absolute (wrist_e40)', color='#1f77b4', edgecolor='black')
bp2 = ax_pinch.bar(x_p + w/2, rel_pinch, width=w, label='Old Relative (relative_e40)', color='#ff7f0e', edgecolor='black')
ax_pinch.set_title('Gripper Closure Deficit Breakdown', fontsize=12, fontweight='bold')
ax_pinch.set_ylabel('Degrees / Millimeters', fontsize=11)
ax_pinch.set_xticks(x_p)
ax_pinch.set_xticklabels(categories, fontsize=9.5)
ax_pinch.set_ylim(0, 10.5)
ax_pinch.grid(True, axis='y', linestyle=':', alpha=0.6)
ax_pinch.legend(frameon=True, fontsize=10)

for b in list(bp1) + list(bp2):
    h = b.get_height()
    ax_pinch.annotate(f'{h:.1f}',
                      xy=(b.get_x() + b.get_width() / 2, h),
                      xytext=(0, 2), textcoords="offset points",
                      ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Subplot 2.4: Height Adaptation (Commanded Depth Z_min across Pallet Heights)
ax_z = axes[1, 1]
# Data from systematic_report.py:
# pallet too_low, nominal, too_high
pallet_layers = ['Pallet Lowered\n(-50 mm)', 'Pallet Nominal\n(Reference)', 'Pallet Increased\n(+50 mm)']
# Medians of commanded z_min (in mm):
# abs: too_low -25.8, nominal +28.8, too_high +104.4 -> delta too_high - too_low = 130.2 mm
abs_z = [-25.8, +28.8, +104.4]
# rel: too_low -33.8, nominal +56.0, too_high +271.5 -> delta too_high - too_low = 305.3 mm
rel_z = [-33.8, +56.0, +271.5]

ax_z.plot(pallet_layers, abs_z, '-o', label='Old Absolute (Range: 130 mm)', color='#1f77b4', linewidth=2.5, markersize=8)
ax_z.plot(pallet_layers, rel_z, '-s', label='Old Relative (Range: 305 mm, Overreacts)', color='#ff7f0e', linewidth=2.5, markersize=8)
ax_z.plot(pallet_layers, [-50, 0, +50], '--k', label='Ideal 1:1 Tracking (Range: 100 mm)', alpha=0.6, linewidth=1.8)

ax_z.set_title('Height Tracking: Commanded Z_min vs Pallet Height', fontsize=12, fontweight='bold')
ax_z.set_ylabel('Commanded Tool0 Z_min (mm)', fontsize=11)
ax_z.grid(True, linestyle=':', alpha=0.6)
ax_z.legend(frameon=True, fontsize=10)

for x, y in zip(pallet_layers, abs_z):
    ax_z.annotate(f'{y:+.1f} mm', xy=(x, y), xytext=(-25, 8), textcoords='offset points', fontweight='bold', color='#1f77b4', fontsize=9)
for x, y in zip(pallet_layers, rel_z):
    ax_z.annotate(f'{y:+.1f} mm', xy=(x, y), xytext=(5, -15), textcoords='offset points', fontweight='bold', color='#d95f02', fontsize=9)

plt.tight_layout()
p2 = os.path.join(fig_dir, "rollout_eval_comparison.png")
plt.savefig(p2, dpi=200)
plt.close()
print("Saved:", p2)

