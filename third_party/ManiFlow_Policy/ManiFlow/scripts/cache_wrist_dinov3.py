#!/usr/bin/env python3
"""Offline feature extraction script for Wrist DINOv3 features.

Caches raw frozen patch tokens F_t in R^{N_p x D} as float16 directly in Zarr.
Uses DINOv3ObsEncoder.extract_patch_tokens() directly to guarantee 100% preprocessing parity
(interpolation, normalization, image order, token stripping) with live inference.
"""

import argparse
import os
import sys

import numpy as np
import torch
import zarr
from tqdm import tqdm

# Ensure maniflow is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from maniflow.model.vision_2d.dinov3_obs_encoder import DINOv3ObsEncoder


def main():
    parser = argparse.ArgumentParser(description="Cache raw DINOv3 wrist patch tokens into Zarr.")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to source Zarr dataset.")
    parser.add_argument("--output_zarr", type=str, default=None,
                        help="Optional companion Zarr path. If omitted, writes into source zarr directly.")
    parser.add_argument("--key_in", type=str, default="wrist_rgb", help="Input image key in zarr.")
    parser.add_argument("--key_out", type=str, default="wrist_dinov3_tokens", help="Output token key.")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--height", type=int, default=192, help="Target image height.")
    parser.add_argument("--width", type=int, default=256, help="Target image width.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--copy_to", type=str, default=None,
                        help="Optional twin Zarr path to copy the cached tokens into immediately.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[Cache] Opening source Zarr: {args.zarr_path}")
    src_root = zarr.open(args.zarr_path, mode='r' if args.output_zarr else 'a')

    src_group = src_root['data'] if 'data' in src_root else src_root
    if args.key_in not in src_group:
        raise KeyError(f"Key '{args.key_in}' not found in {args.zarr_path}. Available: {list(src_group.keys())}")

    wrist_arr = src_group[args.key_in]
    num_frames = wrist_arr.shape[0]
    print(f"[Cache] Found {num_frames} frames in '{args.key_in}' with shape {wrist_arr.shape}")

    # Setup target zarr
    if args.output_zarr:
        out_root = zarr.open(args.output_zarr, mode='a')
        out_group = out_root['data'] if 'data' in out_root else out_root
    else:
        out_group = src_group

    # Build DINOv3ObsEncoder directly for 100% code parity
    shape_meta = {'obs': {args.key_in: {'shape': [3, args.height, args.width], 'type': 'rgb', 'horizon': 1}}}
    print(f"[Cache] Initializing DINOv3ObsEncoder ({args.model_name}) on {device}...")
    encoder = DINOv3ObsEncoder(
        shape_meta=shape_meta,
        model_name=args.model_name,
        pretrained=True,
        frozen=True,
        share_rgb_model=False,
    ).to(device).eval()

    gh = args.height // 16
    gw = args.width // 16
    expected_patches = gh * gw
    feature_dim = encoder.key_feature_dim_map[args.key_in]
    print(f"[Cache] Grid: {gh}x{gw} = {expected_patches} patches, feature_dim = {feature_dim}")

    chunk_frames = min(args.batch_size * 2, 256)
    out_shape = (num_frames, expected_patches, feature_dim)
    chunks = (chunk_frames, expected_patches, feature_dim)

    if args.key_out in out_group:
        print(f"[Cache] Existing '{args.key_out}' found with shape {out_group[args.key_out].shape}. Overwriting.")
        token_ds = out_group[args.key_out]
    else:
        token_ds = out_group.create_dataset(
            args.key_out,
            shape=out_shape,
            chunks=chunks,
            dtype='float16',
            overwrite=True,
        )

    # Process in batches
    bs = args.batch_size
    print(f"[Cache] Processing {num_frames} frames with batch size {bs}...")
    with torch.no_grad():
        for start in tqdm(range(0, num_frames, bs)):
            end = min(start + bs, num_frames)
            batch_raw = wrist_arr[start:end]
            batch_tensor = torch.from_numpy(batch_raw)

            # extract_patch_tokens runs exact same crop/resize/norm/forward_features
            patches = encoder.extract_patch_tokens(args.key_in, batch_tensor)

            token_ds[start:end] = patches.half().cpu().numpy()

    print(f"[Cache] Done! Successfully wrote {num_frames} token vectors to '{args.key_out}' ({token_ds.shape})")

    if args.copy_to:
        print(f"[Cache] Syncing tokens to twin dataset: {args.copy_to}")
        c_root = zarr.open(args.copy_to, mode='a')
        c_group = c_root['data'] if 'data' in c_root else c_root
        if args.key_out in c_group:
            c_group[args.key_out][:] = token_ds[:]
        else:
            c_group.create_dataset(args.key_out, data=token_ds[:], chunks=chunks, dtype='float16', overwrite=True)
        print(f"[Cache] Successfully synced '{args.key_out}' to {args.copy_to}")


if __name__ == "__main__":
    main()
