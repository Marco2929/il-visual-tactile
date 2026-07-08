"""
Dataset Visualization for ManiFlow Zarr Datasets.
Checks variance, diversity, and potential issues in training data.
"""

import zarr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
import os
import argparse
from pathlib import Path


# EE-Pose layout: first 7 dims of action/state are [x, y, z, qx, qy, qz, qw]
# Hand joints are dims 7:13 (6 finger joints) — ignored for spatial analysis
EE_POSE_DIMS = 7  # x, y, z, qx, qy, qz, qw
EE_POS_DIMS = 3   # x, y, z only
HAND_START = 7
HAND_END = 13


def load_zarr_dataset(zarr_path):
    """Load zarr dataset and extract episodes."""
    root = zarr.open(zarr_path, mode='r')
    
    print(f"\n{'='*60}")
    print(f"Dataset: {zarr_path}")
    print(f"{'='*60}")
    
    # Detect structure: flat (root-level) or nested (data/ group)
    if 'data' in root.keys():
        data = root['data']
    else:
        data = root  # flat structure
    meta = root['meta']
    
    print(f"\nData keys: {[k for k in data.keys() if k != 'meta']}")
    print(f"Meta keys: {list(meta.keys())}")
    
    # Load arrays
    states = np.array(data['state'])
    actions = np.array(data['action'])
    point_clouds = np.array(data['point_cloud'])
    
    # Episode boundaries
    episode_ends = np.array(meta['episode_ends'])
    
    print(f"\nShapes:")
    print(f"  States:       {states.shape} | range [{states.min():.3f}, {states.max():.3f}]")
    print(f"  Actions:      {actions.shape} | range [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  Point Clouds: {point_clouds.shape} | range [{point_clouds.min():.3f}, {point_clouds.max():.3f}]")
    print(f"  Episodes:     {len(episode_ends)}")
    
    # Extract episode slices
    episodes = []
    prev_end = 0
    for end in episode_ends:
        episodes.append({
            'state': states[prev_end:end],
            'action': actions[prev_end:end],
            'point_cloud': point_clouds[prev_end:end],
        })
        prev_end = end
    
    ep_lengths = [len(ep['state']) for ep in episodes]
    print(f"  Episode lengths: min={min(ep_lengths)}, max={max(ep_lengths)}, "
          f"mean={np.mean(ep_lengths):.1f}, std={np.std(ep_lengths):.1f}")
    
    return episodes, states, actions, point_clouds


def plot_action_distributions(episodes, actions, output_dir):
    """Plot action distributions across all episodes."""
    n_dims = actions.shape[1]
    
    # Dimension labels
    dim_labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] + \
                 [f'hand_{i}' for i in range(max(0, n_dims - 7))]
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for d in range(min(n_dims, 15)):
        ax = axes[d]
        
        # Histogram of all actions for this dim
        color = 'steelblue' if d < EE_POSE_DIMS else 'coral'
        ax.hist(actions[:, d], bins=50, alpha=0.7, density=True, color=color)
        
        # Per-episode start/end markers
        starts = [ep['action'][0, d] for ep in episodes]
        ends = [ep['action'][-1, d] for ep in episodes]
        
        ax.axvline(np.mean(starts), color='green', linestyle='--', label='mean start')
        ax.axvline(np.mean(ends), color='red', linestyle='--', label='mean end')
        
        label = dim_labels[d] if d < len(dim_labels) else f'dim{d}'
        category = 'EE' if d < EE_POSE_DIMS else 'Hand'
        ax.set_title(f'{category}: {label}\nμ={actions[:, d].mean():.3f}, σ={actions[:, d].std():.3f}')
        ax.set_xlabel('Value')
        if d == 0:
            ax.legend(fontsize=8)
    
    # Hide unused axes
    for d in range(n_dims, len(axes)):
        axes[d].set_visible(False)
    
    fig.suptitle('Action Distributions (blue=EE pose, orange=hand joints)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_action_distributions.png'), dpi=150)
    plt.close()
    print("  ✓ Action distributions")


def plot_action_trajectories(episodes, output_dir):
    """Plot action trajectories over time for all episodes (EE-focused)."""
    n_dims = episodes[0]['action'].shape[1]
    
    # Dimension labels
    dim_labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] + \
                 [f'hand_{i}' for i in range(max(0, n_dims - 7))]
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for d in range(min(n_dims, 15)):
        ax = axes[d]
        
        for i, ep in enumerate(episodes):
            # Normalize time to [0, 1]
            t = np.linspace(0, 1, len(ep['action']))
            alpha = 0.15 if len(episodes) > 20 else 0.4
            color = 'steelblue' if d < EE_POSE_DIMS else 'coral'
            ax.plot(t, ep['action'][:, d], alpha=alpha, linewidth=0.5, color=color)
        
        # Mean trajectory
        max_len = max(len(ep['action']) for ep in episodes)
        all_trajs = []
        for ep in episodes:
            interp = np.interp(
                np.linspace(0, 1, max_len),
                np.linspace(0, 1, len(ep['action'])),
                ep['action'][:, d]
            )
            all_trajs.append(interp)
        all_trajs = np.array(all_trajs)
        mean_traj = all_trajs.mean(axis=0)
        std_traj = all_trajs.std(axis=0)
        
        t_norm = np.linspace(0, 1, max_len)
        ax.plot(t_norm, mean_traj, 'k-', linewidth=2, label='mean')
        ax.fill_between(t_norm, mean_traj - std_traj, mean_traj + std_traj,
                       alpha=0.2, color='red', label='±1σ')
        
        label = dim_labels[d] if d < len(dim_labels) else f'dim{d}'
        category = 'EE' if d < EE_POSE_DIMS else 'Hand'
        ax.set_title(f'{category}: {label} | σ={std_traj.mean():.4f}')
        ax.set_xlabel('Normalized time')
        if d == 0:
            ax.legend(fontsize=8)
    
    for d in range(n_dims, len(axes)):
        axes[d].set_visible(False)
    
    fig.suptitle('Action Trajectories Over Time (blue=EE, orange=hand)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_action_trajectories.png'), dpi=150)
    plt.close()
    print("  ✓ Action trajectories")


def plot_state_variance(episodes, states, output_dir):
    """Analyze state variance — are starting states diverse?"""
    n_dims = states.shape[1]
    
    # Starting states
    start_states = np.array([ep['state'][0] for ep in episodes])
    end_states = np.array([ep['state'][-1] for ep in episodes])
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # Bar plot: std per dimension for start vs all
    dims = np.arange(n_dims)
    width = 0.35
    
    axes[0].bar(dims - width/2, start_states.std(axis=0), width, 
                label='Start states σ', color='green', alpha=0.7)
    axes[0].bar(dims + width/2, states.std(axis=0), width,
                label='All states σ', color='blue', alpha=0.7)
    axes[0].set_xlabel('State dimension')
    axes[0].set_ylabel('Standard deviation')
    axes[0].set_title('State Variance: Starting States vs All States')
    axes[0].legend()
    axes[0].set_xticks(dims)
    
    # Pairwise distances between starting states
    if len(start_states) > 1:
        dists = pdist(start_states)
        axes[1].hist(dists, bins=30, alpha=0.7, color='purple')
        axes[1].set_title(f'Pairwise Distances Between Starting States\n'
                         f'mean={dists.mean():.4f}, std={dists.std():.4f}, '
                         f'min={dists.min():.4f}, max={dists.max():.4f}')
        axes[1].set_xlabel('L2 distance')
        axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_state_variance.png'), dpi=150)
    plt.close()
    print("  ✓ State variance")
    
    # Print warning if starting states are too similar
    start_std = start_states.std(axis=0)
    low_var_dims = np.where(start_std < 0.01)[0]
    if len(low_var_dims) > 0:
        print(f"  ⚠️  LOW VARIANCE in start state dims: {low_var_dims}")
        print(f"     σ values: {start_std[low_var_dims]}")


def plot_point_cloud_samples(episodes, output_dir, n_samples=6):
    """Visualize point clouds from different episodes."""
    indices = np.linspace(0, len(episodes)-1, n_samples, dtype=int)
    
    fig = plt.figure(figsize=(20, 12))
    
    for i, idx in enumerate(indices):
        pc = episodes[idx]['point_cloud'][0]  # First timestep
        xyz = pc[:, :3]
        
        # Top-down
        ax1 = fig.add_subplot(3, n_samples, i + 1)
        ax1.scatter(xyz[:, 0], xyz[:, 1], s=0.5, alpha=0.5, c=xyz[:, 2], cmap='viridis')
        ax1.set_title(f'Ep {idx} Top (XY)')
        ax1.set_aspect('equal')
        
        # Front
        ax2 = fig.add_subplot(3, n_samples, n_samples + i + 1)
        ax2.scatter(xyz[:, 0], xyz[:, 2], s=0.5, alpha=0.5, c=xyz[:, 1], cmap='viridis')
        ax2.set_title(f'Front (XZ)')
        ax2.set_aspect('equal')
        
        # 3D
        ax3 = fig.add_subplot(3, n_samples, 2*n_samples + i + 1, projection='3d')
        ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.3, alpha=0.3)
        ax3.set_title(f'3D')
    
    fig.suptitle('Point Cloud Samples (First Timestep per Episode)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_pointcloud_samples.png'), dpi=150)
    plt.close()
    print("  ✓ Point cloud samples")


def plot_point_cloud_variance(episodes, output_dir):
    """Check how much the point clouds vary between episodes."""
    # Collect first-timestep point clouds
    first_pcs = []
    for ep in episodes:
        pc = ep['point_cloud'][0, :, :3]  # (N, 3) XYZ only
        first_pcs.append(pc)
    
    first_pcs = np.array(first_pcs)  # (num_eps, N, 3)
    
    # Statistics
    mean_pc = first_pcs.mean(axis=0)  # (N, 3)
    std_pc = first_pcs.std(axis=0)    # (N, 3)
    
    # Per-point spatial std
    point_std = np.linalg.norm(std_pc, axis=-1)  # (N,)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram of per-point std
    axes[0].hist(point_std, bins=50, alpha=0.7, color='coral')
    axes[0].set_title(f'Per-Point Spatial Std\nμ={point_std.mean():.4f}, '
                     f'median={np.median(point_std):.4f}')
    axes[0].set_xlabel('Std (m)')
    axes[0].axvline(point_std.mean(), color='red', linestyle='--')
    
    # Bounding box variance
    bboxes_min = np.array([ep['point_cloud'][0, :, :3].min(axis=0) for ep in episodes])
    bboxes_max = np.array([ep['point_cloud'][0, :, :3].max(axis=0) for ep in episodes])
    
    labels = ['X', 'Y', 'Z']
    x = np.arange(3)
    axes[1].bar(x - 0.2, bboxes_min.std(axis=0), 0.4, label='Min corner σ', color='blue', alpha=0.7)
    axes[1].bar(x + 0.2, bboxes_max.std(axis=0), 0.4, label='Max corner σ', color='orange', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title('Bounding Box Corner Variance')
    axes[1].legend()
    axes[1].set_ylabel('Std (m)')
    
    # Centroid variance
    centroids = np.array([ep['point_cloud'][0, :, :3].mean(axis=0) for ep in episodes])
    axes[2].scatter(centroids[:, 0], centroids[:, 1], c=centroids[:, 2], 
                   cmap='viridis', s=30, alpha=0.7)
    axes[2].set_title(f'PC Centroids (XY)\nσ_x={centroids[:,0].std():.4f}, '
                     f'σ_y={centroids[:,1].std():.4f}, σ_z={centroids[:,2].std():.4f}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_aspect('equal')
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='Z')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_pointcloud_variance.png'), dpi=150)
    plt.close()
    print("  ✓ Point cloud variance")
    
    # Warnings
    if point_std.mean() < 0.005:
        print(f"  ⚠️  VERY LOW PC VARIANCE: mean per-point σ = {point_std.mean():.5f}m")
        print(f"     → Point clouds are nearly identical across episodes!")
        print(f"     → Model has NO reason to look at point clouds.")


def plot_trajectory_diversity(episodes, output_dir):
    """PCA on flattened action trajectories to see episode clustering."""
    # Interpolate all trajectories to same length
    target_len = 64
    n_dims = episodes[0]['action'].shape[1]
    
    flat_trajs = []
    for ep in episodes:
        interp_traj = np.zeros((target_len, n_dims))
        for d in range(n_dims):
            interp_traj[:, d] = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(ep['action'])),
                ep['action'][:, d]
            )
        flat_trajs.append(interp_traj.flatten())
    
    flat_trajs = np.array(flat_trajs)
    
    # PCA
    pca = PCA(n_components=min(3, len(episodes)-1))
    projected = pca.fit_transform(flat_trajs)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 2D PCA
    axes[0].scatter(projected[:, 0], projected[:, 1], c=np.arange(len(episodes)),
                   cmap='viridis', s=30, alpha=0.7)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('Trajectory PCA (colored by episode index)')
    plt.colorbar(axes[0].collections[0], ax=axes[0], label='Episode')
    
    # Explained variance
    axes[1].bar(range(len(pca.explained_variance_ratio_)), 
               pca.explained_variance_ratio_, color='steelblue')
    axes[1].set_title(f'PCA Explained Variance\nTotal: {pca.explained_variance_ratio_.sum()*100:.1f}%')
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Variance ratio')
    
    # Pairwise trajectory distances
    traj_dists = pdist(flat_trajs)
    axes[2].hist(traj_dists, bins=40, alpha=0.7, color='teal')
    axes[2].set_title(f'Pairwise Trajectory Distances\n'
                     f'μ={traj_dists.mean():.3f}, σ={traj_dists.std():.3f}')
    axes[2].set_xlabel('L2 distance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_trajectory_diversity.png'), dpi=150)
    plt.close()
    print("  ✓ Trajectory diversity (PCA)")
    
    # Warning
    if pca.explained_variance_ratio_[0] > 0.8:
        print(f"  ⚠️  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance!")
        print(f"     → Trajectories are very similar (low diversity)")


def plot_action_state_correlation(episodes, output_dir):
    """Check if actions can be predicted from state alone (the shortcut!)."""
    # For each timestep, correlate state → next action
    all_states = []
    all_next_actions = []
    
    for ep in episodes:
        if len(ep['state']) < 2:
            continue
        all_states.append(ep['state'][:-1])
        all_next_actions.append(ep['action'][1:])
    
    states = np.concatenate(all_states, axis=0)
    next_actions = np.concatenate(all_next_actions, axis=0)
    
    n_state = states.shape[1]
    n_action = next_actions.shape[1]
    
    # Correlation matrix
    corr_matrix = np.zeros((n_state, n_action))
    for s in range(n_state):
        for a in range(n_action):
            corr_matrix[s, a] = np.corrcoef(states[:, s], next_actions[:, a])[0, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im = axes[0].imshow(np.abs(corr_matrix), cmap='hot', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xlabel('Action dimension')
    axes[0].set_ylabel('State dimension')
    axes[0].set_title('|Correlation| State(t) → Action(t+1)\n'
                     '(High = state predicts action)')
    plt.colorbar(im, ax=axes[0])
    
    # Max correlation per action dim
    max_corr = np.abs(corr_matrix).max(axis=0)
    axes[1].bar(range(n_action), max_corr, color='crimson', alpha=0.7)
    axes[1].axhline(0.9, color='red', linestyle='--', label='0.9 threshold')
    axes[1].axhline(0.7, color='orange', linestyle='--', label='0.7 threshold')
    axes[1].set_xlabel('Action dimension')
    axes[1].set_ylabel('Max |correlation| with any state dim')
    axes[1].set_title('Action Predictability from State Alone')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_state_action_correlation.png'), dpi=150)
    plt.close()
    print("  ✓ State-Action correlation")
    
    highly_correlated = np.where(max_corr > 0.9)[0]
    if len(highly_correlated) > 0:
        print(f"  ⚠️  ACTION DIMS HIGHLY PREDICTABLE FROM STATE: {highly_correlated}")
        print(f"     Correlations: {max_corr[highly_correlated]}")
        print(f"     → Model can ignore vision for these dims!")


def plot_temporal_action_autocorrelation(episodes, output_dir):
    """Check how much action(t+1) ≈ state(t) — the autoregressive shortcut."""
    # action(t) vs state(t) — since action = joint positions and state = joint positions,
    # they might be nearly identical!
    all_states = []
    all_actions = []
    
    for ep in episodes:
        all_states.append(ep['state'])
        all_actions.append(ep['action'])
    
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    n_dims = min(states.shape[1], actions.shape[1])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Per-dim: state(t) vs action(t) scatter
    residuals = actions[:, :n_dims] - states[:, :n_dims]
    residual_std = residuals.std(axis=0)
    
    axes[0].bar(range(n_dims), residual_std, color='navy', alpha=0.7)
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Std of (action - state)')
    axes[0].set_title('Residual: Action(t) - State(t)\n'
                     'If small → action ≈ state → trivial prediction from proprio')
    axes[0].axhline(0.01, color='red', linestyle='--', label='0.01 threshold')
    axes[0].legend()
    
    # Overall: how much does action change per step?
    action_deltas = []
    for ep in episodes:
        if len(ep['action']) > 1:
            deltas = np.diff(ep['action'], axis=0)
            action_deltas.append(deltas)
    
    all_deltas = np.concatenate(action_deltas, axis=0)
    delta_magnitudes = np.linalg.norm(all_deltas, axis=1)
    
    axes[1].hist(delta_magnitudes, bins=50, alpha=0.7, color='darkgreen')
    axes[1].set_title(f'Action Step Size (‖Δa‖)\n'
                     f'μ={delta_magnitudes.mean():.5f}, '
                     f'median={np.median(delta_magnitudes):.5f}')
    axes[1].set_xlabel('Step size')
    axes[1].axvline(delta_magnitudes.mean(), color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_temporal_autocorrelation.png'), dpi=150)
    plt.close()
    print("  ✓ Temporal autocorrelation")
    
    near_zero_dims = np.where(residual_std < 0.01)[0]
    if len(near_zero_dims) > 0:
        ee_trivial = [d for d in near_zero_dims if d < EE_POSE_DIMS]
        hand_trivial = [d for d in near_zero_dims if d >= HAND_START]
        print(f"  ⚠️  DIMS WHERE action ≈ state: {near_zero_dims}")
        if ee_trivial:
            print(f"     → EE dims {ee_trivial}: Trivially predictable from proprio!")
        if hand_trivial:
            print(f"     → Hand dims {hand_trivial}: Expected (hand ≈ state copy)")


def plot_ee_start_end_positions(episodes, output_dir):
    """Plot start and end EE positions (XYZ) for all episodes."""
    starts_xyz = np.array([ep['action'][0, :EE_POS_DIMS] for ep in episodes])
    ends_xyz = np.array([ep['action'][-1, :EE_POS_DIMS] for ep in episodes])

    fig = plt.figure(figsize=(20, 12))
    
    gs = GridSpec(2, 3, figure=fig)

    # --- 3D scatter: starts vs ends ---
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.scatter(starts_xyz[:, 0], starts_xyz[:, 1], starts_xyz[:, 2],
                c='green', s=40, alpha=0.7, label='Start')
    ax3d.scatter(ends_xyz[:, 0], ends_xyz[:, 1], ends_xyz[:, 2],
                c='red', s=40, alpha=0.7, label='End')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title('EE Start/End Positions (3D)')
    ax3d.legend()

    # --- Top-down XY ---
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.scatter(starts_xyz[:, 0], starts_xyz[:, 1], c='green', s=40, alpha=0.7, label='Start')
    ax_xy.scatter(ends_xyz[:, 0], ends_xyz[:, 1], c='red', s=40, alpha=0.7, label='End')
    for i in range(len(episodes)):
        ax_xy.annotate('', xy=ends_xyz[i, :2], xytext=starts_xyz[i, :2],
                      arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))
    ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')
    ax_xy.set_title('Top-Down (XY)')
    ax_xy.set_aspect('equal')
    ax_xy.legend()

    # --- Front XZ ---
    ax_xz = fig.add_subplot(gs[0, 2])
    ax_xz.scatter(starts_xyz[:, 0], starts_xyz[:, 2], c='green', s=40, alpha=0.7, label='Start')
    ax_xz.scatter(ends_xyz[:, 0], ends_xyz[:, 2], c='red', s=40, alpha=0.7, label='End')
    for i in range(len(episodes)):
        ax_xz.annotate('', xy=(ends_xyz[i, 0], ends_xyz[i, 2]),
                      xytext=(starts_xyz[i, 0], starts_xyz[i, 2]),
                      arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))
    ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z')
    ax_xz.set_title('Front (XZ)')
    ax_xz.set_aspect('equal')
    ax_xz.legend()

    # --- Statistics ---
    ax_stats = fig.add_subplot(gs[1, 0])
    labels = ['X', 'Y', 'Z']
    x = np.arange(3)
    width = 0.35
    ax_stats.bar(x - width/2, starts_xyz.std(axis=0), width, label='Start σ', color='green', alpha=0.7)
    ax_stats.bar(x + width/2, ends_xyz.std(axis=0), width, label='End σ', color='red', alpha=0.7)
    ax_stats.set_xticks(x); ax_stats.set_xticklabels(labels)
    ax_stats.set_ylabel('Std (m)')
    ax_stats.set_title('Position Variance (Start vs End)')
    ax_stats.legend()

    # --- Displacement magnitude histogram ---
    ax_disp = fig.add_subplot(gs[1, 1])
    displacements = np.linalg.norm(ends_xyz - starts_xyz, axis=1)
    ax_disp.hist(displacements, bins=20, alpha=0.7, color='purple')
    ax_disp.axvline(displacements.mean(), color='red', linestyle='--',
                   label=f'mean={displacements.mean():.4f}m')
    ax_disp.set_xlabel('Displacement (m)')
    ax_disp.set_ylabel('Count')
    ax_disp.set_title('Start→End Displacement per Episode')
    ax_disp.legend()

    # --- Per-episode trajectories in XY ---
    ax_traj = fig.add_subplot(gs[1, 2])
    for i, ep in enumerate(episodes):
        traj = ep['action'][:, :EE_POS_DIMS]
        alpha = 0.2 if len(episodes) > 20 else 0.5
        ax_traj.plot(traj[:, 0], traj[:, 1], alpha=alpha, linewidth=0.7)
    ax_traj.scatter(starts_xyz[:, 0], starts_xyz[:, 1], c='green', s=20, zorder=5, label='Start')
    ax_traj.scatter(ends_xyz[:, 0], ends_xyz[:, 1], c='red', s=20, zorder=5, label='End')
    ax_traj.set_xlabel('X'); ax_traj.set_ylabel('Y')
    ax_traj.set_title('EE Trajectories (XY)')
    ax_traj.set_aspect('equal')
    ax_traj.legend()

    fig.suptitle('End-Effector Start/End Analysis (Hand joints ignored)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_ee_start_end_positions.png'), dpi=150)
    plt.close()
    print("  ✓ EE start/end positions")

    # Print summary
    print(f"    Start XYZ: mean={starts_xyz.mean(axis=0)}, σ={starts_xyz.std(axis=0)}")
    print(f"    End   XYZ: mean={ends_xyz.mean(axis=0)}, σ={ends_xyz.std(axis=0)}")
    print(f"    Displacement: mean={displacements.mean():.4f}m, std={displacements.std():.4f}m")


def plot_ee_orientation_analysis(episodes, output_dir):
    """Analyze quaternion orientation at start/end of episodes."""
    starts_quat = np.array([ep['action'][0, 3:EE_POSE_DIMS] for ep in episodes])  # qx,qy,qz,qw
    ends_quat = np.array([ep['action'][-1, 3:EE_POSE_DIMS] for ep in episodes])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Quaternion components at start
    quat_labels = ['qx', 'qy', 'qz', 'qw']
    for q in range(4):
        axes[0, 0].hist(starts_quat[:, q], bins=20, alpha=0.5, label=quat_labels[q])
    axes[0, 0].set_title('Start Orientation (Quaternion Components)')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Value')

    # Quaternion components at end
    for q in range(4):
        axes[0, 1].hist(ends_quat[:, q], bins=20, alpha=0.5, label=quat_labels[q])
    axes[0, 1].set_title('End Orientation (Quaternion Components)')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Value')

    # Quaternion norm check
    start_norms = np.linalg.norm(starts_quat, axis=1)
    end_norms = np.linalg.norm(ends_quat, axis=1)
    axes[1, 0].hist(start_norms, bins=20, alpha=0.7, color='green', label='Start')
    axes[1, 0].hist(end_norms, bins=20, alpha=0.7, color='red', label='End')
    axes[1, 0].set_title(f'Quaternion Norms (should be ~1.0)\n'
                         f'Start: {start_norms.mean():.4f}±{start_norms.std():.4f}, '
                         f'End: {end_norms.mean():.4f}±{end_norms.std():.4f}')
    axes[1, 0].legend()
    axes[1, 0].axvline(1.0, color='black', linestyle='--')

    # Angular distance start→end (via dot product)
    dot_products = np.abs(np.sum(starts_quat * ends_quat, axis=1))
    dot_products = np.clip(dot_products, 0, 1)
    angular_dists_deg = 2 * np.arccos(dot_products) * 180 / np.pi
    axes[1, 1].hist(angular_dists_deg, bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title(f'Start→End Angular Distance\n'
                         f'mean={angular_dists_deg.mean():.1f}°, std={angular_dists_deg.std():.1f}°')
    axes[1, 1].set_xlabel('Degrees')
    axes[1, 1].axvline(angular_dists_deg.mean(), color='red', linestyle='--')

    fig.suptitle('End-Effector Orientation Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_ee_orientation.png'), dpi=150)
    plt.close()
    print("  ✓ EE orientation analysis")

    # Warnings
    bad_norms = np.sum((start_norms < 0.99) | (start_norms > 1.01))
    if bad_norms > 0:
        print(f"  ⚠️  {bad_norms} episodes have non-unit quaternions at start!")


def plot_ee_vs_hand_summary(episodes, actions, output_dir):
    """Compare EE-pose action dims vs hand joint dims."""
    ee_actions = actions[:, :EE_POSE_DIMS]
    hand_actions = actions[:, HAND_START:HAND_END]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Variance comparison
    ee_std = ee_actions.std(axis=0)
    hand_std = hand_actions.std(axis=0)
    axes[0].bar(range(EE_POSE_DIMS), ee_std, color='steelblue', alpha=0.7)
    axes[0].set_title('EE-Pose Std per Dim')
    axes[0].set_xlabel('Dim (x,y,z,qx,qy,qz,qw)')
    axes[0].set_ylabel('Std')
    axes[0].set_xticks(range(EE_POSE_DIMS))
    axes[0].set_xticklabels(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    axes[1].bar(range(hand_actions.shape[1]), hand_std, color='coral', alpha=0.7)
    axes[1].set_title('Hand Joint Std per Dim')
    axes[1].set_xlabel('Finger joint index')
    axes[1].set_ylabel('Std')

    # Step size: EE vs Hand
    ee_deltas = np.linalg.norm(np.diff(ee_actions[:, :3], axis=0), axis=1)
    hand_deltas = np.linalg.norm(np.diff(hand_actions, axis=0), axis=1)
    axes[2].hist(ee_deltas, bins=50, alpha=0.6, label=f'EE pos Δ (μ={ee_deltas.mean():.5f})', color='steelblue')
    axes[2].hist(hand_deltas, bins=50, alpha=0.6, label=f'Hand Δ (μ={hand_deltas.mean():.5f})', color='coral')
    axes[2].set_title('Per-Step Action Change: EE vs Hand')
    axes[2].set_xlabel('L2 step size')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_ee_vs_hand.png'), dpi=150)
    plt.close()
    print("  ✓ EE vs Hand summary")


def main():
    parser = argparse.ArgumentParser(description='Visualize ManiFlow Zarr Dataset')
    parser.add_argument('zarr_path', type=str, help='Path to zarr dataset')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for plots')
    args = parser.parse_args()
    
    if args.output_dir is None:
        dataset_name = Path(args.zarr_path).stem
        args.output_dir = f'data/dataset_viz/{dataset_name}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load
    episodes, states, actions, point_clouds = load_zarr_dataset(args.zarr_path)
    
    print(f"\nGenerating visualizations → {args.output_dir}/")
    
    # Generate all plots
    plot_action_distributions(episodes, actions, args.output_dir)
    plot_action_trajectories(episodes, args.output_dir)
    plot_state_variance(episodes, states, args.output_dir)
    plot_point_cloud_samples(episodes, args.output_dir)
    plot_point_cloud_variance(episodes, args.output_dir)
    plot_trajectory_diversity(episodes, args.output_dir)
    plot_action_state_correlation(episodes, args.output_dir)
    plot_temporal_action_autocorrelation(episodes, args.output_dir)
    plot_ee_start_end_positions(episodes, args.output_dir)
    plot_ee_orientation_analysis(episodes, args.output_dir)
    plot_ee_vs_hand_summary(episodes, actions, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Done! All plots saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()