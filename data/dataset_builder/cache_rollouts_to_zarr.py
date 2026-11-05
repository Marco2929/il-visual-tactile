#!/usr/bin/env python3
"""Cache DINOv3 tokens and progress labels for all rollout logs into a dedicated Zarr store.

Designed for long-term scalability:
- Idempotent and incremental: only processes new or updated rollouts
- Exact preprocessing parity with ManiFlow DINOv3ObsEncoder
- Stores float16 tokens and float32 progress curves
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import zarr
from tqdm import tqdm

# Ensure dataset_builder, rollout_logs, and maniflow are in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "data" / "rollout_logs"
MANIFLOW_DIR = ROOT_DIR / "third_party" / "ManiFlow_Policy" / "ManiFlow"

for p in (ROOT_DIR / "data" / "dataset_builder", LOGS_DIR, MANIFLOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import viz.wrist as wrist
from maniflow.model.vision_2d.dinov3_obs_encoder import DINOv3ObsEncoder

DEFAULT_ROLLOUT_ROOTS = [
    ROOT_DIR / "data" / "rollout_logs" / "foil_lifting_wrist_e40",
    ROOT_DIR / "data" / "rollout_logs" / "foil_lifting_relative_e40",
]
DEFAULT_OUT_ZARR = ROOT_DIR / "data" / "zarr_datasets" / "foil_lifting_rollouts_progress.zarr"


def discover_rollouts(roots):
    """Find all rollout directories that have both progress.npz and wrist_bag."""
    rollouts = []
    for root in roots:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        for prog_file in sorted(root.glob("**/progress.npz")):
            ep_dir = prog_file.parent
            bag_dir = ep_dir / "wrist_bag"
            if not bag_dir.exists() or not list(bag_dir.glob("*.mcap")):
                continue
            outcome_file = ep_dir / "outcome.json"
            verdict = "unknown"
            is_success = False
            tags = []
            if outcome_file.exists():
                try:
                    d = json.loads(outcome_file.read_text())
                    verdict = d.get("verdict", "unknown")
                    is_success = d.get("success", verdict in ("pass", "success"))
                    tags = d.get("tags", [])
                except Exception:
                    pass

            rel_key = str(ep_dir.relative_to(ROOT_DIR / "data" / "rollout_logs"))
            rollouts.append({
                "path": ep_dir,
                "key": rel_key,
                "verdict": verdict,
                "is_success": is_success,
                "tags": tags,
            })
    return rollouts


def build_rollouts_zarr(
    rollouts,
    out_zarr_path: pathlib.Path,
    model_name: str = "vit_small_patch16_dinov3.lvd1689m",
    batch_size: int = 64,
    device: str = "cuda",
):
    out_zarr_path = pathlib.Path(out_zarr_path)
    out_zarr_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Cache] Target Zarr: {out_zarr_path}")
    root = zarr.open(str(out_zarr_path), mode="a")
    group = root["data"] if "data" in root else root

    # Check already cached episodes
    cached_keys = set()
    if "meta/episode_sources" in root:
        cached_keys = set(root["meta/episode_sources"][:])
        print(f"[Cache] Found {len(cached_keys)} episodes already cached in Zarr.")

    to_process = [r for r in rollouts if r["key"] not in cached_keys]
    print(f"[Cache] {len(to_process)} new episodes to encode out of {len(rollouts)} total.")

    if not to_process:
        print("[Cache] All rollouts are already cached! Nothing to do.")
        return

    # Initialize DINOv3ObsEncoder
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Cache] Initializing DINOv3ObsEncoder ({model_name}) on {device_obj}...")
    shape_meta = {"obs": {"wrist_rgb": {"shape": [3, 192, 256], "type": "rgb", "horizon": 1}}}
    encoder = DINOv3ObsEncoder(
        shape_meta=shape_meta,
        model_name=model_name,
        pretrained=True,
        frozen=True,
        share_rgb_model=False,
    ).to(device_obj).eval()

    expected_patches = 192  # 12 x 16
    feature_dim = 384       # ViT-S/16

    # Collect existing arrays or create new
    existing_tokens = group["wrist_dinov3_tokens"] if "wrist_dinov3_tokens" in group else None
    existing_prog = group["progress"] if "progress" in group else None
    existing_ends = root["meta/episode_ends"] if "meta/episode_ends" in root else None
    existing_sources = list(root["meta/episode_sources"][:]) if "meta/episode_sources" in root else []
    existing_success = list(root["meta/is_success"][:]) if "meta/is_success" in root else []

    current_frame_offset = existing_tokens.shape[0] if existing_tokens is not None else 0
    current_ends = list(existing_ends[:]) if existing_ends is not None else []

    all_new_tokens = []
    all_new_progress = []
    new_sources = []
    new_success = []
    new_ends = []

    print(f"[Cache] Starting token extraction for {len(to_process)} rollouts...")
    for idx, r in enumerate(to_process):
        ep_dir = r["path"]
        key = r["key"]

        # 1. Load progress curve
        p_npz = np.load(ep_dir / "progress.npz")
        prog = p_npz["progress"].astype(np.float32)

        # 2. Load wrist camera frames
        meta_file = ep_dir / "meta.json"
        started_unix = 0.0
        if meta_file.exists():
            try:
                started_unix = json.loads(meta_file.read_text()).get("started_unix", 0.0)
            except Exception:
                pass

        _, frames, note = wrist.load_frames(ep_dir, started_unix=started_unix, max_frames=0, size=(256, 192))
        if not frames:
            print(f"Warning: No frames decoded for {key} ({note}). Skipping.")
            continue

        n_frames = len(frames)
        # Align frame counts if slight mismatch
        if len(prog) != n_frames:
            prog = np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, len(prog)), prog).astype(np.float32)

        # 3. Extract DINOv3 tokens in batches
        frame_tensor = torch.from_numpy(np.stack(frames, axis=0))
        ep_tokens = []

        with torch.no_grad():
            for b_start in range(0, n_frames, batch_size):
                b_end = min(b_start + batch_size, n_frames)
                batch = frame_tensor[b_start:b_end].to(device_obj)
                tokens = encoder.extract_patch_tokens("wrist_rgb", batch)
                ep_tokens.append(tokens.half().cpu().numpy())

        ep_tokens_arr = np.concatenate(ep_tokens, axis=0)  # (n_frames, 192, 384) in float16

        all_new_tokens.append(ep_tokens_arr)
        all_new_progress.append(prog)
        current_frame_offset += n_frames
        current_ends.append(current_frame_offset)
        existing_sources.append(key)
        existing_success.append(bool(r["is_success"]))

        print(f"  [{idx+1}/{len(to_process)}] Cached {key}: {n_frames} frames, verdict={r['verdict']}")

    if not all_new_tokens:
        print("[Cache] No new frames were processed.")
        return

    # Concatenate all new tokens
    new_tokens_block = np.concatenate(all_new_tokens, axis=0)
    new_prog_block = np.concatenate(all_new_progress, axis=0)

    # Append or create Zarr datasets
    if existing_tokens is None:
        group.create_dataset(
            "wrist_dinov3_tokens",
            data=new_tokens_block,
            chunks=(256, expected_patches, feature_dim),
            dtype="float16",
            overwrite=True,
        )
        group.create_dataset(
            "progress",
            data=new_prog_block,
            chunks=(2048,),
            dtype="float32",
            overwrite=True,
        )
    else:
        existing_tokens.append(new_tokens_block, axis=0)
        existing_prog.append(new_prog_block, axis=0)

    # Save metadata
    root.create_dataset("meta/episode_ends", data=np.array(current_ends, dtype=np.int64), overwrite=True)
    root.create_dataset("meta/episode_sources", data=np.array(existing_sources, dtype=str), overwrite=True)
    root.create_dataset("meta/is_success", data=np.array(existing_success, dtype=bool), overwrite=True)

    print(f"\n[Cache] Successfully updated Zarr dataset: {out_zarr_path}")
    print(f"  Total cached episodes: {len(current_ends)}")
    print(f"  Total frames: {current_frame_offset:,}")
    print(f"  wrist_dinov3_tokens shape: {group['wrist_dinov3_tokens'].shape}")
    print(f"  progress shape: {group['progress'].shape}")


def main():
    parser = argparse.ArgumentParser(description="Cache DINOv3 tokens for all rollouts into Zarr.")
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROLLOUT_ROOTS)
    parser.add_argument("--output_zarr", type=str, default=str(DEFAULT_OUT_ZARR))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rollouts = discover_rollouts(args.roots)
    print(f"Discovered {len(rollouts)} valid rollout episodes.")
    build_rollouts_zarr(
        rollouts,
        out_zarr_path=args.output_zarr,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

