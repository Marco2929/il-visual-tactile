# dataset_builder/processing/depth.py
"""Depth preprocessing for policy training.

Kept separate from rgb.py on purpose - depth plays by different rules:

  * Resizing may ONLY use nearest-neighbour. Any averaging kernel invents
    distances that exist nowhere in the scene (a pixel halfway between a
    foreground edge and the background behind it), and averages the 0 "no
    measurement" sentinel into a plausible-looking real depth.
  * No photometric augmentation. Brightness/contrast/saturation/blur are
    meaningless on a metric quantity: scaling a distance by 1.2 does not model
    any real sensor variation, it just corrupts the geometry.
  * Normalization uses a FIXED range (the workspace depth bounds), never
    per-frame min/max. Per-frame scaling makes the same physical distance map
    to different values depending on what else is in view, so the encoder can
    never learn a consistent notion of "the hand is 20 cm above the table".

Output contract (mirrors rgb.py so the writers need no special case):

  * Layout (T, H, W, 2), channels LAST. The dataset/encoder does the HWC->CHW
    permute.
  * Channel 0: depth, normalized to [0, 1] over FRONT_DEPTH_RANGE_M.
  * Channel 1: validity, 1.0 where a real measurement exists, 0.0 otherwise.
  * dtype float32. Unlike RGB this is NOT stored as uint8: quantizing a
    normalized metric range to 256 steps over ~0.9 m gives ~3.5 mm resolution,
    which is the same order as the signal that distinguishes "pinched" from
    "slipped".

Why a validity channel instead of hole-filling:
  Stereo depth has systematic holes on homogeneous bright surfaces - exactly
  what a white foil is. Filling them with any constant makes them
  indistinguishable from a real measurement at that distance, so the network
  cannot tell "the table is here" from "there is no reading here". The explicit
  mask keeps that difference learnable, and it is the channel that carries the
  slip signal when the foil itself drops out of the depth map.
"""

import numpy as np
from termcolor import cprint

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# Sentinel written by record_extractor/extract_episodes.py for NaN/Inf/negative
# ZED readings. Must stay in sync with save_depth_png_mm() there.
INVALID_DEPTH = 0

MM_PER_M = 1000.0


def _resize_depth(frames: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a (T, H, W) stack with nearest-neighbour only.

    The interpolation mode is hardcoded rather than configurable: there is no
    valid alternative for depth, and exposing it as a knob invites someone to
    set "area" for consistency with the RGB config and quietly destroy the
    depth edges.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for depth preprocessing")

    out = np.empty((frames.shape[0], out_h, out_w), dtype=frames.dtype)
    for i, frame in enumerate(frames):
        out[i] = cv2.resize(frame, (out_w, out_h),
                            interpolation=cv2.INTER_NEAREST)
    return out


def _center_crop_to_aspect(frames: np.ndarray, target_h: int,
                           target_w: int) -> np.ndarray:
    """Center-crop a (T, H, W) stack to the target aspect ratio.

    Mirrors rgb.py's version so RGB and depth stay pixel-aligned. For the front
    camera this is a no-op: 640x360 and the 192x320 target are both 16:9. It
    exists so the two paths cannot diverge if the target ever changes.
    """
    _, h, w = frames.shape
    target_aspect = target_w / target_h
    cur_aspect = w / h

    if abs(cur_aspect - target_aspect) < 1e-6:
        return frames

    if cur_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        return frames[:, :, x0:x0 + new_w]

    new_h = int(round(w / target_aspect))
    y0 = (h - new_h) // 2
    return frames[:, y0:y0 + new_h, :]


def preprocess_depth(frames: np.ndarray, config: dict) -> np.ndarray:
    """Turn raw 16-bit millimetre depth maps into a 2-channel policy input.

    Args:
        frames: (T, H, W) uint16, millimetres, 0 = invalid. This is exactly
            what cv2.imread(path, IMREAD_UNCHANGED) returns for the PNGs
            written by the extractor.
        config: `depth` config block:
            enabled:      bool - if False, return frames untouched
            resolution:   [H, W] target size (default [192, 320], matching
                          front_rgb; bound to the ViT patch size 16)
            depth_range:  [min_m, max_m] clip/normalization bounds. Defaults to
                          camera_info.FRONT_DEPTH_RANGE_M.
            center_crop:  bool - crop to target aspect first (no-op at 16:9)
            encoder_downsample_ratio: patch size / stride for the
                          divisibility check (default 16, DINOv3 ViT-S/16)

    Returns:
        (T, out_h, out_w, 2) float32 - [normalized_depth, validity].
    """
    if frames is None:
        return None

    if not config.get("enabled", True):
        return frames

    if frames.ndim != 3:
        raise ValueError(
            f"Expected depth frames of shape (T, H, W), got {frames.shape}"
        )

    src_h, src_w = frames.shape[1], frames.shape[2]

    res = config.get("resolution", [192, 320])
    out_h, out_w = int(res[0]), int(res[1])

    # The RGB towers run DINOv3 ViT-S/16, so the geometry that matters is
    # divisibility by the patch size: a fractional patch grid changes the
    # token count. Depth is currently NOT an observation key of the policy
    # (it is extracted and written to zarr only), but it is kept aligned with
    # front_rgb so it can be attached without a rebuild.
    stride = int(config.get("encoder_downsample_ratio", 16))
    if out_h % stride or out_w % stride:
        cprint(
            f"  ⚠️  depth resolution {out_h}x{out_w} is not divisible by "
            f"{stride}, so the patch grid would be fractional and the token "
            f"count would not match the configured shape.",
            "red",
        )

    rng = config.get("depth_range")
    if rng is None:
        from camera_info import FRONT_DEPTH_RANGE_M
        rng = FRONT_DEPTH_RANGE_M
    d_min, d_max = float(rng[0]), float(rng[1])
    if not d_max > d_min:
        raise ValueError(
            f"depth_range must be increasing, got ({d_min}, {d_max})"
        )

    # --- 1. mm (uint16) -> m (float32) ------------------------------------
    # Cast first: the arithmetic below would otherwise wrap around in uint16.
    depth_m = frames.astype(np.float32) / MM_PER_M

    # Validity is decided BEFORE clipping, on the original sentinel, and then
    # narrowed by the range test. Doing it after clipping would be impossible:
    # np.clip maps the 0 sentinel to d_min, making it indistinguishable from a
    # real reading at the near plane.
    valid = frames != INVALID_DEPTH

    # --- 2. Clip to the workspace depth bounds ----------------------------
    # Out-of-range readings are marked INVALID rather than clamped. Clamping
    # would place the 2.0 m background wall at exactly d_max, i.e. assert
    # "there is a surface at the far plane" across the whole background - a
    # confident, wrong measurement. Marking it invalid says "no usable reading
    # here", which is the truth, and keeps the depth map consistent with the
    # point cloud, which drops those points entirely.
    in_range = (depth_m >= d_min) & (depth_m <= d_max)
    valid &= in_range

    # --- 3. Normalize over the FIXED range --------------------------------
    normalized = (depth_m - d_min) / (d_max - d_min)
    # Invalid pixels get a defined value instead of whatever the raw mm
    # happened to be. 0.0 is safe only because channel 1 flags it as invalid;
    # on its own it would read as "surface at the near plane".
    normalized = np.where(valid, np.clip(normalized, 0.0, 1.0), 0.0)
    normalized = normalized.astype(np.float32)

    validity = valid.astype(np.float32)

    # --- 4. Crop + resize (nearest only) ----------------------------------
    # The validity mask rides through the SAME geometric ops as the depth, so
    # the two stay in register. Nearest-neighbour keeps it strictly binary;
    # any smoothing would produce fractional "half valid" pixels.
    if config.get("center_crop", True):
        normalized = _center_crop_to_aspect(normalized, out_h, out_w)
        validity = _center_crop_to_aspect(validity, out_h, out_w)

    if (normalized.shape[1], normalized.shape[2]) != (out_h, out_w):
        normalized = _resize_depth(normalized, out_h, out_w)
        validity = _resize_depth(validity, out_h, out_w)

    out = np.stack([normalized, validity], axis=-1)

    invalid_pct = 100.0 * (1.0 - float(validity.mean()))
    mb = out.nbytes / 1e6
    cprint(
        f"  Depth: ({src_h}x{src_w}) -> ({out_h}x{out_w}) float32 HWC 2ch "
        f"[{d_min:.2f}-{d_max:.2f} m], {out.shape[0]} frames, "
        f"{invalid_pct:.1f}% invalid, {mb:.1f} MB",
        "cyan",
    )

    return np.ascontiguousarray(out, dtype=np.float32)