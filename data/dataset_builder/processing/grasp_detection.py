# dataset_builder/processing/grasp_detection.py
"""Detect grasp attempts in a proprioceptive trajectory.

Standalone reimplementation - deliberately does NOT import from
third_party/ManiFlow_Policy, so the builder has no sys.path dependency on
the policy repo.

WHAT IS DETECTED
----------------
Every stretch where the pinch is CLOSED is one grasp *attempt*. Episodes
with a failed first pinch therefore yield several attempts, which is the
whole point: the retry is the recovery behaviour we want to keep, not noise
to be filtered out.

SIGNAL
------
Finger 2 (index) does the grasping; finger 1 is the opposing thumb and
barely moves. On foil_lifting_wrist_tesollo.zarr the per-dim spread of the
unnormalised state is

    [11] rj_dg_2_1  -22.1 ..  3.5   std  2.96
    [12] rj_dg_2_2    2.5 .. 70.2   std 22.83
    [13] rj_dg_2_3    2.4 .. 65.2   std 18.09
    [14] rj_dg_2_4    1.7 .. 70.4   std 21.44

so dim 11 carries almost no grasp information while 12-14 swing by ~60
units. GRASP_DIMS excludes dim 11 for that reason: averaging it in only
adds a near-constant offset and dilutes the signal.

Because the dims differ from the ManiFlow detector (which averages 11-14),
the threshold is NOT transferable. It is estimated per episode from the
open/closed levels instead of hardcoded - see `closed_mask`.

ABSOLUTE HEIGHT IS NOT USED
---------------------------
Only the SHAPE of the z trajectory (descending / arrived) is used, never an
absolute z value. Pallet height varies between collections and is going to
vary more, so a constant like compute_progress' `z_grasp=0.10` would be a
built-in failure mode.
"""

import numpy as np

# state = 7 (ee_pose: xyz + quat xyzw) + 8 (pinch hand: rj_dg_1_1..1_4,
# rj_dg_2_1..2_4). Matches proprioception.build_state_vector with
# hand.pinch: true.
STATE_DIM = 15
EE_Z_DIM = 2
# rj_dg_2_2 .. rj_dg_2_4 - the joints that actually close (see module doc).
GRASP_DIMS = (12, 13, 14)
# Fraction of the open->closed span at which the hand counts as closed.
# Derived per episode, so it survives a recalibration of the hand.
CLOSED_FRAC = 0.5
# Closed runs shorter than this are the fingers brushing the foil on the way
# in, not a grasp. Measured on foil_lifting_wrist_tesollo.zarr: several
# episodes open with a 3-5 frame blip well before the real pinch
# (ep 174: run[85,89) then the actual close at 168; ep 24: run[119,124)
# then 139). At 3 frames those blips became `first_grasp` and pulled the
# trim anchor ~80 frames too early, i.e. straight back into the approach
# this experiment is meant to remove. 10 frames = 0.5 s at 20 fps, which is
# shorter than every real pinch observed (shortest sustained run: 12) and
# longer than every blip (longest: 5).
MIN_CLOSED_FRAMES = 10
# The approach ends once z has come within this fraction of the episode's z
# span above its minimum, i.e. the arm has arrived at working height.
#
# Kept only as a diagnostic (`arrived_idx`). It is NOT the trim anchor: a
# FRACTION of the span is a different metric height in every episode, so on
# steep descents it fires only once the fingers are already closing
# (measured: arrived == first_grasp on eps 100, 200, 49, 90, 99, 229) and
# would cut away the whole search phase. Use `descent_anchor` instead.
DESCENT_FRAC = 0.15
# Trim anchor: metres above the LOWEST point the hand reaches. Fixed metric
# offset from a per-episode reference, so it is identical in every episode
# and still immune to a varying pallet height.
APPROACH_MARGIN_M = 0.10


def descent_anchor(state, margin_m=APPROACH_MARGIN_M, smooth_w=5):
    """First frame at which the hand is within `margin_m` of its lowest point.

    This is the trim anchor. Everything from here to the END of the episode
    is kept - including any failed pinch and re-grasp, which is one recovery
    behaviour rather than two samples.

    The reference is the episode's own z minimum, so no absolute height is
    assumed anywhere and a different pallet height just shifts the whole
    curve. Scanning FORWARD and taking the first crossing means a later
    descent (the re-approach after a failed pinch) cannot move the anchor
    backwards.

    Returns -1 for an empty episode.
    """
    z = smooth(np.asarray(state)[:, EE_Z_DIM], smooth_w)
    if len(z) == 0:
        return -1
    below = np.flatnonzero(z <= float(z.min()) + margin_m)
    return int(below[0]) if len(below) else 0


def grasp_signal(state):
    """(T, 15) -> (T,) scalar, high while the pinch is closed."""
    return np.asarray(state)[:, list(GRASP_DIMS)].mean(axis=1)


def smooth(x, w):
    """Centred moving average with EDGE padding; w <= 1 is a no-op.

    Edge padding is essential rather than cosmetic: np.convolve(mode='same')
    pads with ZEROS, which drags the first and last frames of every episode
    towards zero and can push a closed pinch back below the threshold.
    """
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"),
                       np.ones(w) / w, mode="valid")[:len(x)]


def runs(mask):
    """Boolean mask -> list of (start, stop) half-open runs of True."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2], edges[1::2]))


def closed_mask(sig, closed_frac=CLOSED_FRAC):
    """Per-episode open/closed decision plus the levels it was based on.

    The threshold is taken from THIS episode's own percentiles rather than a
    global constant. Reason: how far the pinch closes depends on how much
    material it grips - a curved sheet closes less far - so a global level
    tuned on the median episode rejects the tails. p5/p95 instead of
    min/max so a single dropout frame cannot define either level.

    Returns (mask, open_level, closed_level, threshold).
    """
    lo = float(np.percentile(sig, 5))
    hi = float(np.percentile(sig, 95))
    thr = lo + closed_frac * (hi - lo)
    return sig > thr, lo, hi, thr


def detect_episode_grasps(state, smooth_w=5,
                          min_closed_frames=MIN_CLOSED_FRAMES,
                          descent_frac=DESCENT_FRAC,
                          closed_frac=CLOSED_FRAC,
                          margin_m=APPROACH_MARGIN_M):
    """Grasp attempts and phase anchors for ONE episode.

    Args:
        state: (T, 15) UNNORMALISED proprio, exactly as stored in the zarr.
            Thresholds are in joint-angle units, so a normalised state would
            silently produce garbage.
        smooth_w: moving-average window for the grasp and z signals.
        min_closed_frames: drop closed runs shorter than this.
        descent_frac: see module doc / DESCENT_FRAC.
        closed_frac: see `closed_mask`.

    Note that attempts are NOT split into separate training segments and the
    'successful' one is not singled out. A failed pinch followed by a retry
    is ONE recovery behaviour - noticing the foil slipped and re-grasping -
    and the trim keeps it as a single contiguous segment. `n_attempts` is
    therefore reported for diagnostics only; it never influences the cut.

    Returns dict with
        attempts      : list of (start, stop) for every closed run
        n_attempts    : len(attempts), diagnostics only
        first_grasp   : frame the fingers START closing for the FIRST
                        attempt, or -1. This is the trim anchor: cutting
                        here drops the approach but keeps every retry.
        last_grasp    : same for the final attempt.
        anchor_idx    : THE TRIM ANCHOR - first frame within
                        `margin_m` of the episode's lowest z.
        arrived_idx   : frame the descent ends, diagnostics only (see
                        DESCENT_FRAC for why it is not the anchor).
        z_at_first    : ee z at first_grasp (NaN if none).
        open_level / closed_level / threshold : the levels used.
        has_grasp     : whether any attempt was found.
    """
    state = np.asarray(state)
    n = len(state)
    out = {
        "attempts": [], "n_attempts": 0,
        "first_grasp": -1, "last_grasp": -1,
        "anchor_idx": -1, "arrived_idx": -1,
        "z_at_first": float("nan"),
        "open_level": float("nan"), "closed_level": float("nan"),
        "threshold": float("nan"), "has_grasp": False,
    }
    if n == 0:
        return out
    if state.shape[1] != STATE_DIM:
        raise ValueError(
            f"grasp detection expects the {STATE_DIM}-dim pinch state "
            f"(ee_pose 7 + hand 8), got {state.shape[1]}. GRASP_DIMS "
            f"{GRASP_DIMS} would point at unrelated columns."
        )

    out["anchor_idx"] = descent_anchor(state, margin_m, smooth_w)

    sig = smooth(grasp_signal(state), smooth_w)
    closed, lo, hi, thr = closed_mask(sig, closed_frac)
    out.update(open_level=lo, closed_level=hi, threshold=thr)

    attempts = [(int(a), int(b)) for a, b in runs(closed)
                if b - a >= min_closed_frames]
    out["attempts"] = attempts
    out["n_attempts"] = len(attempts)

    z = smooth(state[:, EE_Z_DIM], smooth_w)
    z_lo, z_hi = float(z.min()), float(z.max())
    z_span = max(z_hi - z_lo, 1e-6)
    # Height-RELATIVE so this survives a varying pallet height.
    above = (z - z_lo) / z_span > descent_frac

    if not attempts:
        low = np.flatnonzero(~above)
        out["arrived_idx"] = int(low[0]) if len(low) else -1
        return out

    out["has_grasp"] = True

    # Walk back from 'fully closed' to the frame where closing STARTED. The
    # onset level sits halfway between open and the threshold: the pinch
    # makes small preparatory movements while searching, and a lower level
    # latches onto those instead of the real closing motion.
    onset_level = lo + 0.5 * (thr - lo)

    def onset(closed_start):
        pre = np.flatnonzero(sig[:closed_start] <= onset_level)
        return int(pre[-1] + 1) if len(pre) else int(closed_start)

    out["first_grasp"] = onset(attempts[0][0])
    out["last_grasp"] = onset(attempts[-1][0])
    out["z_at_first"] = float(state[out["first_grasp"], EE_Z_DIM])

    g = out["first_grasp"]
    low = np.flatnonzero(~above[:g]) if g > 0 else []
    out["arrived_idx"] = int(low[0]) if len(low) else g
    return out


def iter_episodes(state, episode_ends):
    """Yield (index, start, stop, state_slice) per episode."""
    ends = np.asarray(episode_ends)
    starts = np.concatenate([[0], ends[:-1]])
    for i, (s, e) in enumerate(zip(starts, ends)):
        yield i, int(s), int(e), state[int(s):int(e)]