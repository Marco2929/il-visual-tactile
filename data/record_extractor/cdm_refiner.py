#!/usr/bin/env python3
"""
Camera Depth Model (CDM) depth refinement.

Wraps the ByteDance CDM (camera-depth-model) to clean/refine raw depth maps
before point cloud generation. The CDM takes RGB + noisy depth and predicts
a refined metric depth map.

Usage:
    refiner = CDMRefiner(model_repo='depth-anything/camera-depth-model-zed2i-neural')
    clean_depth = refiner.refine(rgb_bgr, raw_depth_meters)
"""

import os
import sys

import cv2
import numpy as np
import torch

# CDM lives in different places per container:
#   ur_artus:     third_party/manip-as-in-sim-suite/cdm
#   zed_container: a local 'manip/cdm' next to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
_CDM_CANDIDATES = [
    os.environ.get("CDM_DIR", ""),
    "/ros2_ws/third_party/manip-as-in-sim-suite/cdm",
    os.path.join(_HERE, "manip", "cdm"),
]
for _cand in _CDM_CANDIDATES:
    if _cand and os.path.isfile(os.path.join(_cand, "infer.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
else:
    raise ImportError(
        f"Could not find cdm/infer.py in any of: {_CDM_CANDIDATES}")

from infer import load_model, model_configs  # noqa: E402


def _resolve_checkpoint(spec: str) -> str:
    """Resolve a model spec to a local checkpoint file.

    spec is either a local .ckpt/.pth path or a HuggingFace repo id.
    """
    if os.path.isfile(spec):
        return spec

    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub required to resolve repo ids. "
            "pip install huggingface_hub or pass a local file path."
        )

    files = list_repo_files(spec)
    ckpts = [f for f in files if f.endswith((".ckpt", ".pth", ".safetensors"))]
    if not ckpts:
        raise SystemExit(f"No checkpoint in repo '{spec}'. Files: {files}")
    if len(ckpts) > 1:
        print(f"[cdm] multiple checkpoints in {spec}, using: {ckpts[0]}")

    print(f"[cdm] resolving {spec} -> {ckpts[0]}")
    return hf_hub_download(repo_id=spec, filename=ckpts[0])


class CDMRefiner:
    """Refines raw depth maps using a Camera Depth Model."""

    def __init__(
        self,
        model_spec: str = "depth-anything/camera-depth-model-zed2i-neural",
        encoder: str = "vitl",
        input_size: int = 518,
        max_depth: float = 25.0,
    ):
        """
        Args:
            model_spec: HF repo id OR local checkpoint path
            encoder: ViT encoder size (vits/vitb/vitl/vitg). CDMs use vitl.
            input_size: inference resolution
            max_depth: depth values above this (meters) are treated as invalid
        """
        checkpoint = _resolve_checkpoint(model_spec)
        self.model = load_model(encoder, checkpoint)
        self.input_size = input_size
        self.max_depth = max_depth
        print(f"[cdm] CDMRefiner ready (encoder={encoder}, input_size={input_size})")

    def refine(self, color_bgr: np.ndarray, depth_meters: np.ndarray) -> np.ndarray:
        """
        Refine a raw depth map.

        Args:
            color_bgr: (H, W, 3) uint8 BGR image (as from cv2 / ZED)
            depth_meters: (H, W) float32 depth in METERS

        Returns:
            (H, W) float32 refined depth in METERS
        """
        # CDM expects RGB
        rgb = np.asarray(color_bgr[:, :, ::-1])

        # Build inverse (similarity) depth for the model input
        depth_clean = depth_meters.astype(np.float32).copy()
        depth_clean[~np.isfinite(depth_clean)] = 0.0
        depth_clean[depth_clean > self.max_depth] = 0.0

        simi_depth = np.zeros_like(depth_clean)
        valid = depth_clean > 0
        simi_depth[valid] = 1.0 / depth_clean[valid]

        # Inference → inverse depth → metric depth
        pred_inv = self.model.infer_image(rgb, simi_depth, input_size=self.input_size)
        pred_metric = 1.0 / pred_inv

        # Resize to original resolution if the model changed it
        if pred_metric.shape[:2] != depth_meters.shape[:2]:
            pred_metric = cv2.resize(
                pred_metric,
                (depth_meters.shape[1], depth_meters.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return pred_metric.astype(np.float32)
