# dataset_builder/processing/trim.py
"""Cut the approach off an episode, keeping the grasp onwards.

WHY
---
Every recording drives in from far away and ends up at roughly the same
spot before the pinch, so the approach is both (a) trivially reproducible by
an IK move and (b) a large share of the frames. Training on it lets the
policy spend capacity on a motion that does not need learning, and - worse -
lets it predict most of the trajectory from proprioception alone, which is
exactly the shortcut that keeps the wrist camera from mattering.

WHERE THE CUT GOES
------------------
The anchor is the first frame at which the hand is within `margin_m` of the
LOWEST point it reaches in that episode (grasp_detection.descent_anchor).

Three properties make this the right anchor, all of them measured on
foil_lifting_wrist_tesollo.zarr (252 episodes):

  * It is a fixed METRIC offset. Over 20 inspected episodes the cut landed
    at 9.6-9.9 cm above the low point every time. The earlier candidate -
    a fraction of the episode's z span - varied per episode and collapsed
    onto the grasp itself on six of them, which would have removed the
    entire search phase.
  * It needs no absolute height. The reference is the episode's own z
    minimum, so a different pallet height just shifts the whole curve.
    Contrast compute_progress' hardcoded `z_grasp=0.10`.
  * It scans FORWARD. Searching backwards from the pinch would let the
    re-approach after a failed grasp pull the anchor later and cut away the
    recovery.

RETRIES ARE KEPT WHOLE
----------------------
Everything from the anchor to the END of the episode is kept. A failed pinch
followed by a re-grasp is ONE recovery behaviour - notice the foil slipped,
set up again - not two samples, so attempts are never split and the
"successful" one is never singled out. `n_attempts` is recorded for
reporting only.

MARGIN AND RANDOMISATION
------------------------
Build with a GENEROUS margin (20 cm) and let the training dataloader sample
the actual start between ~10 and 20 cm. The margin that ends up in the zarr
is an upper bound, not the final crop: cutting tight at 10 cm here would be
irreversible without reprocessing all 252 episodes. `anchor_idx` and
`first_grasp` are written to the episode metadata so the loader can do that
without re-running the detector.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .grasp_detection import detect_episode_grasps

# Keep at least this many frames, else the episode is dropped. An action
# chunk has to fit inside the segment; shorter than this and the sequence
# sampler would pad more than it reads.
MIN_SEGMENT_FRAMES = 24


@dataclass
class TrimResult:
    """Where to cut one episode, and why."""
    start: int = 0
    end: int = 0            # exclusive
    skip: bool = False
    reason: str = ""
    info: Dict = field(default_factory=dict)

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def n_frames(self) -> int:
        return max(self.end - self.start, 0)


def compute_trim(state: np.ndarray, config: Optional[Dict] = None) -> TrimResult:
    """Decide the kept frame range for ONE episode.

    Args:
        state: (T, 15) UNNORMALISED proprio - ee_pose (7) + pinch hand (8),
            i.e. the output of build_state_vector with hand.pinch: true.
            Thresholds are in metres and joint-angle units, so a normalised
            state would silently produce nonsense.
        config: the task YAML's `trim:` block. Keys:
            enabled        (bool, default False)
            margin_m       (float, default 0.20) metres above the low point
            min_frames     (int, default 24)
            require_grasp  (bool, default True) drop episodes with no
                           detectable pinch instead of keeping them untrimmed

    Returns:
        TrimResult. With `enabled: false` this is the full range, so callers
        need no second code path.
    """
    cfg = config or {}
    n = len(state)

    if not cfg.get("enabled", False):
        return TrimResult(start=0, end=n, reason="disabled")

    margin = float(cfg.get("margin_m", 0.20))
    min_frames = int(cfg.get("min_frames", MIN_SEGMENT_FRAMES))
    require_grasp = bool(cfg.get("require_grasp", True))

    info = detect_episode_grasps(state, margin_m=margin)

    # No detectable pinch means the signal assumptions do not hold for this
    # episode. Dropping it is the honest option: keeping it UNTRIMMED would
    # quietly mix full approaches into a dataset that is defined by not
    # having them.
    if require_grasp and not info["has_grasp"]:
        return TrimResult(start=0, end=n, skip=True,
                          reason="no_grasp_detected", info=info)

    start = info["anchor_idx"]
    if start < 0:
        return TrimResult(start=0, end=n, skip=True,
                          reason="no_anchor", info=info)

    # The anchor must not land after the fingers close - that would cut away
    # the grasp this dataset exists for. Observed margin was 6-60 frames, so
    # this is a guard against a pathological episode, not a routine case.
    first = info["first_grasp"]
    if first >= 0 and start > first:
        return TrimResult(start=0, end=n, skip=True,
                          reason="anchor_after_grasp", info=info)

    if n - start < min_frames:
        return TrimResult(start=start, end=n, skip=True,
                          reason=f"segment_too_short({n - start})", info=info)

    return TrimResult(start=int(start), end=n, reason="ok", info=info)


def trim_metadata(result: TrimResult, orig_frames: int) -> Dict:
    """Per-episode provenance, indices RELATIVE to the trimmed segment.

    Relative because that is the only frame of reference that still exists
    once the segment is written: the dataloader sees the trimmed episode and
    has no way to undo the offset.
    """
    info = result.info or {}
    first = info.get("first_grasp", -1)
    return {
        "orig_frames": int(orig_frames),
        "trim_start": int(result.start),
        "n_frames": int(result.n_frames),
        "first_grasp": int(first - result.start) if first >= 0 else -1,
        "n_attempts": int(info.get("n_attempts", 0)),
    }