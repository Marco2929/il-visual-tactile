# dataset_builder/processing/rgb.py
"""RGB preprocessing for policy training.

Design contract (must stay in sync with the ManiFlow image encoder):

  * Output layout is (T, H, W, C) — channels LAST. The dataset/encoder does the
    HWC -> CHW permute, matching how `realdex_dataset.py` already treats
    `wrist_rgb`.
  * Output dtype is uint8 in [0, 255]. We deliberately do NOT scale to [0,1] or
    apply ImageNet mean/std here: the encoder does `/255` (+ optional
    normalization) itself. Keeping uint8 makes the zarr 4x smaller than float32
    and lets you swap backbones without rebuilding the dataset.
  * Colors are RGB (the loaders already flip OpenCV's BGR).

IMPORTANT — resolution constraint:
  `TimmObsEncoder` computes
      feature_map_shape = [x // downsample_ratio for x in image_shape]
  with FLOOR division, while a ResNet's real spatial output is CEIL. If H or W
  is not divisible by `downsample_ratio` (32 by default) the `spatial_embedding`
  / `transformer` aggregators are allocated the wrong number of tokens and the
  forward pass breaks. So keep H and W divisible by 32.

  Per-key resolutions (share_rgb_model: false, so each tower gets its own
  batch and they need not agree):

    wrist_rgb  480x640 -> 192x256. Both 4:3, the native sensor aspect, so
               center_crop is a no-op and the full FOV survives undistorted.
               6x8 = 48 tokens at stride 32.

    front_rgb  360x640 -> 192x320. The source is an isotropically HALVED HD720
               frame (1280x720 -> 640x360), written that way by
               record_extractor/extract_episodes.py. Both source and target are
               16:9, so center_crop is again a no-op. 6x10 = 60 tokens.

  The front path used to be squashed anisotropically to 640x480 to fake the
  wrist camera's 4:3. That is gone: a non-uniform scale breaks the single
  isotropic 0.5 factor that relates the stored images to the camera intrinsics,
  so RGB, depth and the point cloud would no longer be mutually projectable.
  See camera_info.camera_intrinsics vs. camera_intrinsics_half.
"""

import numpy as np
from termcolor import cprint

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


_INTERP = {
    "area": "INTER_AREA",
    "linear": "INTER_LINEAR",
    "cubic": "INTER_CUBIC",
    "nearest": "INTER_NEAREST",
}


def _resize_frames(frames: np.ndarray, out_h: int, out_w: int,
                   interpolation: str = "area") -> np.ndarray:
    """Resize a (T, H, W, C) uint8 stack to (T, out_h, out_w, C)."""
    if cv2 is None:
        raise ImportError("opencv-python is required for RGB preprocessing")

    interp = getattr(cv2, _INTERP.get(interpolation, "INTER_AREA"))
    out = np.empty((frames.shape[0], out_h, out_w, frames.shape[3]),
                   dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = cv2.resize(frame, (out_w, out_h), interpolation=interp)
    return out


def _center_crop_to_aspect(frames: np.ndarray, target_h: int,
                           target_w: int) -> np.ndarray:
    """Center-crop so the aspect ratio matches target before resizing.

    Avoids the geometric distortion you get from squashing 4:3 (640x480)
    straight into a square. Distorted geometry is especially harmful for a
    wrist camera, where the policy has to reason about metric spatial offsets.
    """
    _, h, w, _ = frames.shape
    target_aspect = target_w / target_h
    cur_aspect = w / h

    if abs(cur_aspect - target_aspect) < 1e-6:
        return frames

    if cur_aspect > target_aspect:
        # Too wide -> trim width
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        return frames[:, :, x0:x0 + new_w, :]

    # Too tall -> trim height
    new_h = int(round(w / target_aspect))
    y0 = (h - new_h) // 2
    return frames[:, y0:y0 + new_h, :, :]


def preprocess_rgb(frames: np.ndarray, config: dict) -> np.ndarray:
    """Prepare raw RGB frames for training storage.

    Args:
        frames: (T, H, W, 3) uint8 RGB. Wrist: native 480x640. Front: the
            already-halved 360x640 HD720 frame from the extractor.
        config: `rgb` config block:
            enabled:       bool  - if False, return frames untouched
            resolution:    [H, W] target size (default [192, 256], must be
                           divisible by encoder_downsample_ratio)
            center_crop:   bool  - crop to target aspect first. No-op when the
                           aspect already matches (480x640 -> 192x256 at 4:3,
                           360x640 -> 192x320 at 16:9).
            interpolation: area|linear|cubic|nearest (default "area")
            encoder_downsample_ratio: backbone stride for the divisibility
                           check (default 32)

    Returns:
        (T, out_h, out_w, 3) uint8, channels-last, RGB, unnormalized.
    """
    if frames is None:
        return None

    if not config.get("enabled", True):
        return np.ascontiguousarray(frames, dtype=np.uint8)

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB frames of shape (T, H, W, 3), got {frames.shape}"
        )

    # The loaders return a negative-stride view from the BGR->RGB flip;
    # make it contiguous so cv2 can operate on it.
    frames = np.ascontiguousarray(frames, dtype=np.uint8)

    res = config.get("resolution", [192, 256])
    out_h, out_w = int(res[0]), int(res[1])
    src_h, src_w = frames.shape[1], frames.shape[2]

    # Warn early instead of failing deep inside the encoder.
    stride = int(config.get("encoder_downsample_ratio", 32))
    if out_h % stride or out_w % stride:
        cprint(
            f"  ⚠️  resolution {out_h}x{out_w} is not divisible by {stride}. "
            f"TimmObsEncoder's floor-division feature_map_shape will disagree "
            f"with the backbone's ceil output and break aggregation.",
            "red",
        )

    if config.get("center_crop", True):
        frames = _center_crop_to_aspect(frames, out_h, out_w)

    if (frames.shape[1], frames.shape[2]) != (out_h, out_w):
        frames = _resize_frames(
            frames, out_h, out_w,
            interpolation=config.get("interpolation", "area"),
        )

    mb = frames.nbytes / 1e6
    cprint(
        f"  RGB: ({src_h}x{src_w}) -> ({out_h}x{out_w}) "
        f"uint8 HWC, {frames.shape[0]} frames, {mb:.1f} MB",
        "cyan",
    )

    return np.ascontiguousarray(frames, dtype=np.uint8)