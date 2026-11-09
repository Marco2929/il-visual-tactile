#!/usr/bin/env python3
"""
Verification script for robust foil surface height extraction (z_foil).
Tests on 50 episodes across layers (unten, mitte, oben) to ensure:
  1. High consistency within each layer (low std).
  2. Clear height separation between layers.
  3. ZERO outliers from the box rim (z ~ -0.02 .. +0.03 m).
"""
import zarr
import numpy as np

def estimate_foil_height(pc_pts, park_xy=None, search_radius=0.25):
    """
    Robustly estimate the foil surface height z_foil in base_link frame.

    Algorithm:
      1. Spatial workspace filter:
         - Discard robot arm/gripper reflections (z > 0.12 m).
         - Discard table drop-offs (z < -0.55 m).
      2. Optional park/grasp cylinder ROI (if park_xy is given).
      3. Histogram mode analysis (5 mm bins):
         - Identifies the dominant horizontal plane (highest point density).
         - If a deep foil stack exists (significant peak below -0.05 m),
           suppresses the box rim peak (-0.04 to +0.03 m) to prevent rim latching.
      4. Trimmed median of inliers within +/- 15 mm of the mode peak.

    Returns:
      z_foil (float): Estimated surface height in metres.
      n_inliers (int): Number of point cloud points supporting this plane.
      status (str): 'ok', 'low_inliers', or 'too_few_points'.
    """
    pts = pc_pts[:, :3]
    mask = (pts[:, 2] < 0.12) & (pts[:, 2] > -0.55)
    valid_pts = pts[mask]
    if len(valid_pts) < 10:
        return None, 0, 'too_few_points'

    if park_xy is not None:
        d = np.linalg.norm(valid_pts[:, :2] - park_xy, axis=1)
        cyl_pts = valid_pts[d < search_radius]
        if len(cyl_pts) >= 15:
            valid_pts = cyl_pts

    bin_size = 0.005  # 5 mm
    bins = np.arange(-0.55, 0.12 + bin_size, bin_size)
    counts, edges = np.histogram(valid_pts[:, 2], bins=bins)

    # Box rim suppression when foil is inside the box:
    # If there is a dominant cluster below -0.05 m, search only below -0.05 m.
    lower_mask = edges[:-1] < -0.05
    if np.any(lower_mask) and np.max(counts[lower_mask]) > 25:
        search_counts = counts.copy()
        search_counts[~lower_mask] = 0
        peak_idx = np.argmax(search_counts)
    else:
        peak_idx = np.argmax(counts)

    mode_z = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
    inliers = valid_pts[np.abs(valid_pts[:, 2] - mode_z) <= 0.015, 2]
    if len(inliers) < 5:
        return float(mode_z), len(inliers), 'low_inliers'

    z_est = float(np.median(inliers))
    return z_est, len(inliers), 'ok'


def main():
    zarr_path = '/ros2_ws/data/zarr_datasets/foil_lifting_box.zarr'
    root = zarr.open(zarr_path, mode='r')
    ep_ends = root['meta']['episode_ends'][:]
    sources = root['meta']['episode_sources'][:]
    state = root['state']
    pc = root['point_cloud']

    # Select 50 episodes across the 3 main layers:
    # 15 unten, 20 mitte, 15 oben
    unten_all = [i for i, s in enumerate(sources) if 'unten' in s]
    mitte_all = [i for i, s in enumerate(sources) if 'mitte' in s]
    oben_all = [i for i, s in enumerate(sources) if 'oben' in s]

    selected_indices = (
        unten_all[:15] +
        mitte_all[:20] +
        oben_all[:15]
    )
    assert len(selected_indices) == 50, f"Expected 50 episodes, got {len(selected_indices)}"

    results = []
    print(f"{'Ep':>3s} | {'Source / Layer':<32s} | {'z_est (mm)':>10s} | {'Inliers':>7s} | {'EE Grasp (mm)':>14s} | {'dz (EE - Foil)':>14s}")
    print("-" * 92)

    for ep in selected_indices:
        start = ep_ends[ep-1] if ep > 0 else 0
        end = ep_ends[ep]
        src = sources[ep]
        
        # EE grasp pose
        st = state[start:end, :3]
        g_idx = np.argmin(st[:, 2])
        ee_grasp_z = float(st[g_idx, 2])
        park_xy = st[0, :2]

        # Extract height from t=0 observation
        p0 = pc[start]
        z_est, n_inliers, status = estimate_foil_height(p0, park_xy=park_xy)

        dz = (ee_grasp_z - z_est) * 1000.0 if z_est is not None else float('nan')
        z_est_mm = z_est * 1000.0 if z_est is not None else float('nan')
        ee_grasp_mm = ee_grasp_z * 1000.0

        layer = 'unten' if 'unten' in src else ('mitte' if 'mitte' in src else 'oben')
        results.append({
            'ep': ep,
            'src': src,
            'layer': layer,
            'z_est_mm': z_est_mm,
            'n_inliers': n_inliers,
            'ee_grasp_mm': ee_grasp_mm,
            'dz_mm': dz
        })

        print(f"{ep:3d} | {src:<32s} | {z_est_mm:+10.1f} | {n_inliers:7d} | {ee_grasp_mm:+14.1f} | {dz:+14.1f}")

    # Summary statistics
    print("\n" + "=" * 92)
    print("LAYER SUMMARY STATISTICS (50 EPISODES)")
    print("=" * 92)
    print(f"{'Layer':<10s} | {'N':>3s} | {'Mean z_est (mm)':>16s} | {'Std (mm)':>10s} | {'Min (mm)':>10s} | {'Max (mm)':>10s} | {'Box Rim Outliers':>17s}")
    print("-" * 92)

    for layer_name in ['unten', 'mitte', 'oben']:
        group = [r for r in results if r['layer'] == layer_name]
        z_vals = np.array([r['z_est_mm'] for r in group])
        # Box rim sits at z > -50 mm for unten/mitte
        rim_outliers = sum(1 for z in z_vals if z > -50.0) if layer_name in ['unten', 'mitte'] else 0
        print(f"{layer_name:<10s} | {len(group):3d} | {z_vals.mean():+16.1f} | {z_vals.std():10.1f} | {z_vals.min():+10.1f} | {z_vals.max():+10.1f} | {rim_outliers:17d}")

if __name__ == '__main__':
    main()
