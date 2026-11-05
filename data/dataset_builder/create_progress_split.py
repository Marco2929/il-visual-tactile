#!/usr/bin/env python3
"""Generate a strict, stratified 70/30 train/val split manifest across Demos and Rollouts.

Guarantees:
- Zero frame- or episode-level train/test contamination
- Equal representation of success, failure, and reactivity across splits
- Deterministic, persistent JSON manifest for auditable, reproducible experiments
"""

import argparse
import json
import pathlib
import sys
import numpy as np
import zarr

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEMO_ZARR = ROOT_DIR / "data" / "zarr_datasets" / "foil_lifting_box_progress.zarr"
ROLLOUT_ZARR = ROOT_DIR / "data" / "zarr_datasets" / "foil_lifting_rollouts_progress.zarr"
OUT_SPLIT = ROOT_DIR / "data" / "zarr_datasets" / "progress_split_70_30.json"


def generate_split(
    demo_zarr_path: pathlib.Path,
    rollout_zarr_path: pathlib.Path,
    out_json_path: pathlib.Path,
    val_ratio: float = 0.30,
    seed: int = 42,
):
    rng = np.random.default_rng(seed=seed)
    buckets = {}

    # 1. Parse Teleop Demos
    if demo_zarr_path.exists():
        z_demo = zarr.open(str(demo_zarr_path), mode="r")
        demo_ends = z_demo["meta/episode_ends"][:]
        demo_sources = z_demo["meta/episode_sources"][:] if "meta/episode_sources" in z_demo else [f"demo_{i:04d}" for i in range(len(demo_ends))]

        print(f"[Split] Found {len(demo_ends)} demo episodes in {demo_zarr_path.name}")
        for ep_idx in range(len(demo_ends)):
            src = str(demo_sources[ep_idx])
            # Determine collection group from source path
            group_name = "demo_other"
            if "/" in src:
                coll = src.split("/")[0]
                group_name = f"demo_{coll}"
            elif "_" in src:
                group_name = f"demo_{src.split('_')[0]}"

            buckets.setdefault(group_name, []).append({
                "store": "demos",
                "ep_idx": int(ep_idx),
                "name": src,
                "is_success": True,
                "group": group_name,
            })
    else:
        print(f"Warning: Demo Zarr not found at {demo_zarr_path}")

    # 2. Parse Rollouts
    if rollout_zarr_path.exists():
        z_rollout = zarr.open(str(rollout_zarr_path), mode="r")
        rollout_ends = z_rollout["meta/episode_ends"][:]
        rollout_sources = z_rollout["meta/episode_sources"][:]
        rollout_success = z_rollout["meta/is_success"][:]

        print(f"[Split] Found {len(rollout_ends)} rollout episodes in {rollout_zarr_path.name}")
        for ep_idx in range(len(rollout_ends)):
            src = str(rollout_sources[ep_idx])
            succ = bool(rollout_success[ep_idx])

            # Classify rollout
            if "reactivity" in src.lower():
                bucket_name = "rollout_reactivity"
            elif succ:
                bucket_name = "rollout_pass"
            else:
                bucket_name = "rollout_fail"

            buckets.setdefault(bucket_name, []).append({
                "store": "rollouts",
                "ep_idx": int(ep_idx),
                "name": src,
                "is_success": succ,
                "group": bucket_name,
            })
    else:
        print(f"Warning: Rollout Zarr not found at {rollout_zarr_path}")

    # 3. Stratified Partitioning
    train_list = []
    val_list = []
    group_stats = {}

    for bucket_name, items in sorted(buckets.items()):
        items_arr = np.array(items, dtype=object)
        n = len(items_arr)
        perm = rng.permutation(n)
        shuffled = items_arr[perm]

        n_val = int(round(n * val_ratio))
        # Ensure at least 1 in val if n >= 3
        if n >= 3 and n_val == 0:
            n_val = 1
        n_train = n - n_val

        val_items = shuffled[:n_val].tolist()
        train_items = shuffled[n_val:].tolist()

        train_list.extend(train_items)
        val_list.extend(val_items)

        group_stats[bucket_name] = {
            "total": n,
            "train": n_train,
            "val": n_val,
            "val_pct": round((n_val / n) * 100, 1),
        }

    total_all = len(train_list) + len(val_list)
    manifest = {
        "metadata": {
            "val_ratio": val_ratio,
            "seed": seed,
            "total_episodes": total_all,
            "train_episodes": len(train_list),
            "val_episodes": len(val_list),
            "effective_val_pct": round((len(val_list) / total_all) * 100, 2) if total_all else 0.0,
            "stores": {
                "demos": str(demo_zarr_path),
                "rollouts": str(rollout_zarr_path),
            },
        },
        "group_statistics": group_stats,
        "train": train_list,
        "val": val_list,
    }

    out_json_path = pathlib.Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[Split] Successfully saved stratified manifest to {out_json_path}")
    print(f"  Total: {total_all} episodes | Train: {len(train_list)} ({(len(train_list)/total_all)*100:.1f}%) | Val: {len(val_list)} ({(len(val_list)/total_all)*100:.1f}%)")
    print("\nGroup Breakdown:")
    for grp, st in group_stats.items():
        print(f"  {grp:<30}: {st['total']:3d} total -> {st['train']:3d} train, {st['val']:2d} val ({st['val_pct']}%)")


def main():
    parser = argparse.ArgumentParser(description="Create stratified 70/30 train/val split manifest.")
    parser.add_argument("--demos", type=str, default=str(DEMO_ZARR))
    parser.add_argument("--rollouts", type=str, default=str(ROLLOUT_ZARR))
    parser.add_argument("--out", type=str, default=str(OUT_SPLIT))
    parser.add_argument("--val_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_split(
        demo_zarr_path=pathlib.Path(args.demos),
        rollout_zarr_path=pathlib.Path(args.rollouts),
        out_json_path=pathlib.Path(args.out),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

