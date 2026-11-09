#!/usr/bin/env python3
"""Comprehensive Dataset Analysis and Anomaly Detection.

Analyzes:
1. Episode lengths & statistical distributions across box heights.
2. Grasp & lift metrics (did the hand close? did it lift?).
3. Progress signal monotonicity and terminal values.
4. Z-height minima and force spikes to detect potential collisions with the box/ground.
5. Action comparison: next_state vs recorded action (tracking lag, errors).
6. Wrist camera visual consistency across chronological episodes.
"""

import os
import sys
import json
import glob
import numpy as np
import zarr

def run_analysis():
    zarr_box_path = "/ros2_ws/data/zarr_datasets/foil_lifting_box.zarr"
    zarr_rec_path = "/ros2_ws/data/zarr_datasets/foil_lifting_box_recorded_action.zarr"
    ext_dir = "/ros2_ws/data/episodes/external"

    print("=" * 70)
    print("1. LOADING ZARR DATASETS")
    print("=" * 70)

    z_box = zarr.open(zarr_box_path, "r")
    z_rec = zarr.open(zarr_rec_path, "r")

    ends = z_box["meta/episode_ends"][:]
    sources = [str(s) for s in z_box["meta/episode_sources"][:]]
    state = z_box["state"][:]
    action_ns = z_box["action"][:]
    action_rec = z_rec["action"][:]
    progress = z_box["progress"][:]
    n_episodes = len(ends)

    starts = np.concatenate([[0], ends[:-1]])
    lengths = ends - starts

    print(f"Total Episodes: {n_episodes}")
    print(f"Total Frames:   {ends[-1]}")
    print(f"Frame lengths:  min={lengths.min()}, max={lengths.max()}, median={np.median(lengths):.1f}, mean={lengths.mean():.1f}")

    # Identify short or long outliers
    short_eps = [(i, sources[i], lengths[i]) for i in range(n_episodes) if lengths[i] < 60]
    long_eps = [(i, sources[i], lengths[i]) for i in range(n_episodes) if lengths[i] > 350]
    print(f"Short episodes (< 60 frames): {len(short_eps)}")
    for i, s, l in short_eps:
        print(f"  [Ep {i:03d}] {s}: {l} frames")
    print(f"Long episodes (> 350 frames): {len(long_eps)}")
    for i, s, l in long_eps[:5]:
        print(f"  [Ep {i:03d}] {s}: {l} frames")

    print("\n" + "=" * 70)
    print("2. GRASP, LIFT & PROGRESS ANOMALY DETECTION")
    print("=" * 70)

    # State layout:
    # 0..6: ee_pose (x, y, z, qx, qy, qz, qw)
    # 7..10: thumb (rj_dg_1_1..4)
    # 11..14: index (rj_dg_2_1..4)
    # Finger closure: sum of index angles (dims 11..14)
    INDEX_DIMS = slice(11, 15)
    THUMB_DIMS = slice(7, 11)
    EE_Z_IDX = 2

    no_grasp = []
    no_lift = []
    low_progress = []
    early_reopen = []
    ep_metrics = []

    for i in range(n_episodes):
        a, b = int(starts[i]), int(ends[i])
        ep_state = state[a:b]
        ep_prog = progress[a:b]
        src = sources[i]

        # Grasp closure
        index_closure = ep_state[:, INDEX_DIMS].sum(axis=1)
        peak_idx = int(np.argmax(index_closure))
        start_closure = index_closure[0]
        peak_closure = index_closure[peak_idx]
        end_closure = index_closure[-1]
        closure_delta = peak_closure - start_closure

        # Lift
        z_vals = ep_state[:, EE_Z_IDX]
        z_min = float(np.min(z_vals))
        z_grasp = float(z_vals[peak_idx])
        z_end = float(z_vals[-1])
        lift_travel = z_end - z_min

        # Progress
        final_prog = float(ep_prog[-1])
        max_prog = float(np.max(ep_prog))

        # Check anomalies
        if closure_delta < 15.0:  # barely closed fingers
            no_grasp.append((i, src, closure_delta))

        if lift_travel < 0.03:  # lifted less than 3 cm
            no_lift.append((i, src, lift_travel))

        if final_prog < 0.80:
            low_progress.append((i, src, final_prog))

        if (peak_closure - end_closure) > 25.0:
            early_reopen.append((i, src, peak_closure - end_closure))

        ep_metrics.append({
            "idx": i,
            "source": src,
            "length": b - a,
            "z_min": z_min,
            "z_grasp": z_grasp,
            "z_end": z_end,
            "lift_travel": lift_travel,
            "peak_closure": peak_closure,
            "closure_delta": closure_delta,
            "reopen": peak_closure - end_closure,
            "final_prog": final_prog,
        })

    print(f"Episodes with weak/no grasp (closure delta < 15°): {len(no_grasp)}")
    for i, s, d in no_grasp:
        print(f"  [Ep {i:03d}] {s}: delta={d:.1f}°")

    print(f"Episodes with weak/no lift (travel < 3cm): {len(no_lift)}")
    for i, s, d in no_lift:
        print(f"  [Ep {i:03d}] {s}: lift={d*100:.1f} cm")

    print(f"Episodes with final progress < 0.80: {len(low_progress)}")
    for i, s, p in low_progress:
        print(f"  [Ep {i:03d}] {s}: final_prog={p:.2f}")

    print(f"Episodes where hand reopened significantly (> 25° before end): {len(early_reopen)}")
    for i, s, r in early_reopen[:5]:
        print(f"  [Ep {i:03d}] {s}: reopened by {r:.1f}°")

    print("\n" + "=" * 70)
    print("3. Z-HEIGHT & COLLISION RISK ANALYSIS (GROUND PENETRATION)")
    print("=" * 70)

    # Group by height level from collection name (folien_oben_*, folien_mitte_*, folien_unten_*)
    by_height = {"oben": [], "mitte": [], "unten": [], "other": []}
    for m in ep_metrics:
        src = m["source"]
        if "folien_oben_" in src:
            by_height["oben"].append(m)
        elif "folien_mitte_" in src:
            by_height["mitte"].append(m)
        elif "folien_unten_" in src:
            by_height["unten"].append(m)
        else:
            by_height["other"].append(m)

    z_outliers = []
    for h_name, group in by_height.items():
        if not group:
            continue
        z_mins = [m["z_min"] for m in group]
        mean_z = np.mean(z_mins)
        std_z = np.std(z_mins)
        min_z = np.min(z_mins)
        max_z = np.max(z_mins)
        print(f"Height '{h_name}' ({len(group)} eps): z_min mean={mean_z:.4f}m, std={std_z:.4f}m, range=[{min_z:.4f}, {max_z:.4f}]m")

        # Threshold for collision: > 3.0 std below mean
        for m in group:
            if m["z_min"] < mean_z - 2.5 * std_z:
                z_outliers.append((m["idx"], m["source"], m["z_min"], mean_z, (mean_z - m["z_min"])/std_z))

    print(f"\nPotential bottom collision / extreme z_min outliers (> 2.5 sigma below height mean): {len(z_outliers)}")
    for idx, src, z_val, mean_val, sigma in z_outliers:
        print(f"  [Ep {idx:03d}] {src}: z_min={z_val:.4f}m (mean={mean_val:.4f}m, -{sigma:.1f}σ)")

    print("\n" + "=" * 70)
    print("4. RAW FORCE & CONTACT SPIKES (COLLISION DETECTION)")
    print("=" * 70)

    # Check raw force files if available
    force_spikes = []
    for m in ep_metrics:
        src = m["source"]
        force_path = os.path.join(ext_dir, src, "force.npy")
        if os.path.exists(force_path):
            try:
                f_arr = np.load(force_path)
                # force is (T, 5) or (T, 20) or similar (fingertip normal forces)
                max_f = float(np.max(np.abs(f_arr)))
                p99_f = float(np.percentile(np.abs(f_arr), 99))
                if max_f > 80.0:  # unusually high force
                    force_spikes.append((m["idx"], src, max_f, p99_f))
            except Exception:
                pass

    print(f"Episodes with extreme peak force (> 80.0 N): {len(force_spikes)}")
    for idx, src, max_f, p99_f in sorted(force_spikes, key=lambda x: -x[2])[:10]:
        print(f"  [Ep {idx:03d}] {src}: max_force={max_f:.1f} N, p99={p99_f:.1f} N")

    print("\n" + "=" * 70)
    print("5. ACTION COMPARISON: NEXT_STATE vs RECORDED ACTION")
    print("=" * 70)

    # Difference between recorded action and next state action
    # Position dims: 0, 1, 2 (x, y, z)
    # Orientation dims: 3, 4, 5, 6 (quat)
    # Fingers: 7..14
    pos_diff = np.linalg.norm(action_rec[:, :3] - action_ns[:, :3], axis=1)
    finger_diff = np.mean(np.abs(action_rec[:, 7:15] - action_ns[:, 7:15]), axis=1)

    print(f"Cartesian Pos Action Delta (Recorded vs NextState):")
    print(f"  Mean:   {np.mean(pos_diff)*1000:.1f} mm")
    print(f"  Median: {np.median(pos_diff)*1000:.1f} mm")
    print(f"  P95:    {np.percentile(pos_diff, 95)*1000:.1f} mm")
    print(f"  Max:    {np.max(pos_diff)*1000:.1f} mm")

    print(f"\nFinger Action Delta (Recorded vs NextState):")
    print(f"  Mean:   {np.mean(finger_diff):.2f}°")
    print(f"  Median: {np.median(finger_diff):.2f}°")
    print(f"  Max:    {np.max(finger_diff):.2f}°")

    # Find episodes with extreme action divergence (> 150mm difference)
    large_divergence = []
    for i in range(n_episodes):
        a, b = int(starts[i]), int(ends[i])
        ep_pos_diff = pos_diff[a:b]
        if np.max(ep_pos_diff) > 0.15:  # > 15 cm divergence
            large_divergence.append((i, sources[i], float(np.max(ep_pos_diff)), float(np.mean(ep_pos_diff))))

    print(f"\nEpisodes with large action divergence (> 15 cm peak): {len(large_divergence)}")
    for idx, src, max_d, mean_d in large_divergence[:10]:
        print(f"  [Ep {idx:03d}] {src}: peak delta={max_d*1000:.1f} mm, mean={mean_d*1000:.1f} mm")

    print("\n" + "=" * 70)
    print("6. WRIST CAMERA DISPLACEMENT / SHIFT DETECTION")
    print("=" * 70)

    # To detect physical camera shifting on its mount:
    # 1. In wrist images, the camera sees the robot gripper/mounting plate at a fixed position.
    # 2. We inspect mean wrist image across chronological collections/episodes.
    # 3. Check for abrupt jumps in image edge profile or color histogram.
    wrist_images = z_box["wrist_rgb"]  # (N, 192, 256, 3)

    # Sample middle frame of each episode (when hand is in approach/grasp)
    sample_frames = []
    for i in range(n_episodes):
        mid_idx = int((starts[i] + ends[i]) // 2)
        sample_frames.append(wrist_images[mid_idx])
    sample_frames = np.stack(sample_frames)  # (358, 192, 256, 3)

    # Compute mean intensity & vertical centroid
    mean_intensities = sample_frames.mean(axis=(1, 2, 3))  # (358,)
    # Compute difference between consecutive episodes
    diffs = np.mean(np.abs(sample_frames[1:].astype(float) - sample_frames[:-1].astype(float)), axis=(1, 2, 3))
    p99_diff = np.percentile(diffs, 95)

    abrupt_visual_shifts = []
    for i in range(len(diffs)):
        if diffs[i] > 35.0:  # significant visual jump between consecutive episodes
            abrupt_visual_shifts.append((i+1, sources[i+1], diffs[i]))

    print(f"Episodes with abrupt visual frame changes (> 35 intensity delta): {len(abrupt_visual_shifts)}")
    for idx, src, d in abrupt_visual_shifts[:10]:
        print(f"  [Ep {idx:03d}] {src}: delta={d:.2f}")

    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    run_analysis()
