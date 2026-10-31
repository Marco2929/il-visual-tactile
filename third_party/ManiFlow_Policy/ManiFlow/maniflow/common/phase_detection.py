"""Task phase detection from the proprioceptive trajectory.

WHY NOT FIXED THIRDS
--------------------
The first version of the per-phase ablation split each episode into equal
thirds. That is a proxy: the real phase boundaries depend on WHEN the grasp
happens, which varies per episode, and an episode with a second grasp attempt
gets sorted into the wrong bucket entirely. The same problem would hit DAgger
data, where retries are the norm rather than the exception.

PHASES
------
    0  APPROACH   arm descending toward the pallet, hand open.
                  Point cloud matters here (pallet pose, and later a varying
                  pallet height).
    1  SEARCH     at working height, hand still open, locating a graspable
                  piece of foil. Wrist RGB dominates - this is the phase the
                  task is decided in.
    2  HOLD       hand closed. Proprio is near-sufficient; vision only still
                  matters to notice the foil slipping out of the pinch.

SIGNALS (all measured, see scripts/inspect_hand_dims.py)
--------------------------------------------------------
Grasp: mean of `rj_dg_2_1..2_4` (state dims 11..14) crossing ~31.7 upward.
Finger 2 does the grasping (delta +36..+48 open->closed) while finger 1 barely
moves (+2..+8), i.e. finger 1 is the opposing thumb. All four finger-2 joints
separate equally well (cluster gap/sd ~1.9), so their mean is used rather than
one joint - it is robust to a single noisy dimension.

Deliberately NOT used: absolute ee z height. `compute_progress` hardcodes
`z_grasp=0.10 +/- 0.05` against a measured z range of -0.08..0.587, and the
pallet height is going to vary. The joint signal alone separates cleanly, so
the height test adds a failure mode without adding information. Only the
SHAPE of the z trajectory (descending / stable) is used, which is invariant to
the absolute height.

FAILED GRASPS
-------------
A descent that resumes AFTER the hand closed means the grasp did not hold: the
arm went back down for another attempt. `detect_episode_phases` reports these,
so a partially successful episode can be labelled as such (progress ~0.6
rather than ~0.9) instead of being treated as a clean success.
"""

import numpy as np

# state = 7 (ee_pose: xyz + quat xyzw) + 8 (pinch hand), see task shape_meta.
EE_Z_DIM = 2
# rj_dg_2_1..2_4 - the grasping finger.
GRASP_DIMS = (11, 12, 13, 14)
# Measured on foil_lifting_wrist_tesollo.zarr (scripts/inspect_nograsp.py).
#
# NOT the 2-means midpoint of the open/closed clusters: that came out at 31.7,
# which sits INSIDE the distribution of per-episode grasp peaks (30.2..49.4,
# median 35.9) and therefore rejected 14 of 252 episodes whose pinch happens
# to close 1-2 units less far - the operator confirms all episodes do contain
# a grasp. The 2-means estimate is biased low because ~42% of all frames are
# approach frames with the hand open, which drags the 'closed' centroid down.
#
# 28 is below the lowest per-episode peak (30.2) and still far above the open
# level (~7-10), so it separates grasp-vs-no-grasp rather than
# fully-closed-vs-partly-closed. Episodes with a curved sheet grip more
# material and close less far; arm height is NOT the discriminator (z_max
# 0.420 vs 0.412 for the rest).
#
# Re-run scripts/inspect_nograsp.py if the hand or its calibration changes.
GRASP_THRESHOLD = 28.0
# Hand closes with INCREASING joint angle (delta +46.8 on rj_dg_2_2).
GRASP_CLOSES_INCREASING = True

PHASE_APPROACH, PHASE_SEARCH, PHASE_TRANSITION, PHASE_HOLD = 0, 1, 2, 3
PHASE_NAMES = ('approach', 'search', 'transition', 'hold')

# Action-chunk length (cfg.n_action_steps). The policy emits 32 actions at
# once, so WHERE to grasp is decided ~32 steps before the fingers move; by the
# time they do, the (x, y) target is already fixed and merely being executed.
# The search window is therefore anchored this far BEFORE closing.
DECISION_HORIZON = 32
# Frames around the closing onset that are excluded from 'search'. At that
# point ee z reverses sign (descent stops, lift begins) and the gripper
# channel steps, so a 2-frame timing offset produces a large MSE that has
# nothing to do with spatial search. Kept as its own phase instead of
# contaminating either neighbour.
TRANSITION_PRE, TRANSITION_POST = 2, 5


def grasp_signal(state):
    """(T, 15) -> (T,) scalar that is high while the pinch is closed."""
    sig = np.asarray(state)[:, list(GRASP_DIMS)].mean(axis=1)
    return sig if GRASP_CLOSES_INCREASING else -sig


def _smooth(x, w):
    """Centred moving average with EDGE padding; w<=1 is a no-op.

    Needed because both signals are differentiated below and raw joint /
    pose streams are noisy enough that a single jittery frame would create a
    spurious transition.

    The edge padding is essential, not cosmetic. np.convolve(mode='same')
    pads with ZEROS, which drags the last frames of every episode toward 0:
    a closed-pinch value of ~49 averaged with two zeros gives ~29, i.e.
    BELOW the 31.7 threshold. That made `held_to_end` false for all 238
    grasps and suppressed short runs at both ends.
    """
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode='edge'),
                       np.ones(w) / w, mode='valid')[:len(x)]


def _runs(mask):
    """Boolean mask -> list of (start, stop) half-open runs of True."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2], edges[1::2]))


def _open_baseline(sig):
    """Joint value of the fully open hand, as a low percentile.

    Taken from the data rather than assumed, because the rest position differs
    per hand calibration. p5 instead of min so a single dropout frame cannot
    define the baseline.
    """
    return float(np.percentile(sig, 5))


def detect_episode_phases(state, smooth=5, min_closed_frames=3,
                          descent_frac=0.15, onset_frac=0.5):
    """Phase label per frame for ONE episode.

    Args:
        state: (T, 15) unnormalised proprio (ee_pose + hand), i.e. exactly
            what the zarr stores - NOT normalised, since the thresholds are
            in joint-angle units.
        smooth: moving-average window for both signals.
        min_closed_frames: closed runs shorter than this are dropped as
            sensor noise rather than counted as a grasp attempt.
        descent_frac: the approach ends once z has come within this fraction
            of the episode's z span above its minimum, i.e. the arm has
            arrived at working height.
        onset_frac: the search ends once the grasp signal has risen this far
            from the open baseline toward the closed threshold, i.e. the
            fingers START to close.

    BOTH signals are used, because either one alone mislabels:
      * z alone ended the approach while the arm was still descending, so
        'search' contained late approach frames.
      * the closed-threshold alone ended the search only once the pinch was
        already shut, so 'search' also contained the whole closing motion -
        which is proprio-driven, since the decision WHERE to grasp has
        already been made by then. That diluted exactly the phase the task is
        decided in.
    With both, 'search' is the segment where the arm is at height AND the
    hand is still open - the open-hand search the operator described.

    Returns:
        dict with
            phases       : (T,) int array of PHASE_* labels
            grasp_idx    : frame where the fingers START closing for the
                           final grasp (anchor t_close)
            search_start : first frame of the decision window
            closed_idx   : frame where the pinch is fully closed
            arrived_idx  : frame where the descent ends (approach -> search)
            attempts     : list of (start, stop) for every closed run
            n_attempts   : len(attempts)
            failed_grasp : True if the arm descended again after a grasp
            held_to_end  : True if the hand is still closed on the last frame
    """
    state = np.asarray(state)
    n = len(state)
    out = {
        'phases': np.full(n, PHASE_APPROACH, dtype=np.int8),
        'grasp_idx': -1, 'closed_idx': -1, 'arrived_idx': -1,
        'search_start': -1,
        'attempts': [], 'n_attempts': 0,
        'failed_grasp': False, 'held_to_end': False,
    }
    if n == 0:
        return out

    sig = _smooth(grasp_signal(state), smooth)
    closed = sig > GRASP_THRESHOLD
    # Onset level sits between the open baseline and the closed threshold, so
    # it fires when the fingers begin to move rather than when they arrive.
    base = _open_baseline(sig)
    onset_level = base + onset_frac * (GRASP_THRESHOLD - base)
    # Every closed run is a grasp ATTEMPT; keeping all of them (instead of
    # just the first crossing) is what makes retries representable.
    attempts = [(a, b) for a, b in _runs(closed)
                if b - a >= min_closed_frames]
    out['attempts'] = attempts
    out['n_attempts'] = len(attempts)

    z = _smooth(state[:, EE_Z_DIM], smooth)
    z_lo, z_hi = float(z.min()), float(z.max())
    z_span = max(z_hi - z_lo, 1e-6)
    # Height-RELATIVE, so this survives a varying pallet height; an absolute
    # z_grasp constant would not.
    above = (z - z_lo) / z_span > descent_frac

    if not attempts:
        # No grasp at all: everything after the descent counts as searching.
        first_low = np.flatnonzero(~above)
        start = int(first_low[0]) if len(first_low) else n
        out['phases'][start:] = PHASE_SEARCH
        return out

    # The LAST attempt is the one that matters for the phase layout; earlier
    # ones are retries and stay inside SEARCH, which is where they belong
    # semantically (still looking for a graspable edge).
    closed_idx = int(attempts[-1][0])
    out['closed_idx'] = closed_idx
    out['held_to_end'] = bool(closed[-1])

    # Walk back from 'fully closed' to the frame where closing started; that
    # is the end of the open-hand search.
    #
    # onset_frac=0.5 (halfway from open to the closed threshold) rather than
    # something smaller: the pinch makes small preparatory movements during
    # the search, and a low level latches onto those, collapsing 'search' to
    # ~7% of all frames. Half-way is past that jitter but still well before
    # the pinch is shut.
    pre = np.flatnonzero(sig[:closed_idx] <= onset_level)
    grasp_idx = int(pre[-1] + 1) if len(pre) else closed_idx
    out['grasp_idx'] = grasp_idx

    # A renewed descent after an earlier attempt means that attempt failed.
    if len(attempts) > 1:
        out['failed_grasp'] = True
    else:
        a_end = attempts[0][1]
        if a_end < n:
            # Did z drop significantly below its post-grasp level again?
            post = z[a_end:]
            if len(post) > smooth and (post.max() - post.min()) > 0.3 * z_span:
                out['failed_grasp'] = bool(
                    np.argmin(post) > np.argmax(post))

    # --- phase layout -------------------------------------------------
    # search is the DECISION window [t_close - H, t_close - pre), not the
    # whole stretch between arrival and closing: with action chunking the
    # grasp point is chosen a chunk ahead, so measuring at the moment the
    # fingers move reports execution (proprio-driven) rather than the
    # visually-driven choice.
    #
    # Its start is additionally clamped to the end of the descent, so a long
    # chunk horizon cannot pull 'search' back into the approach.
    low = np.flatnonzero(~above[:grasp_idx]) if grasp_idx > 0 else []
    arrived = int(low[0]) if len(low) else grasp_idx
    out['arrived_idx'] = arrived

    trans_start = max(grasp_idx - TRANSITION_PRE, 0)
    trans_end = min(grasp_idx + TRANSITION_POST, n)
    search_start = max(grasp_idx - DECISION_HORIZON, arrived)
    search_start = min(search_start, trans_start)

    out['search_start'] = search_start
    out['phases'][search_start:trans_start] = PHASE_SEARCH
    out['phases'][trans_start:trans_end] = PHASE_TRANSITION
    out['phases'][trans_end:] = PHASE_HOLD
    return out


def detect_dataset_phases(state, episode_ends, **kwargs):
    """Per-frame phases for a whole replay buffer.

    Returns (phases (N,), per_episode list of the detect_episode_phases dicts).
    """
    state = np.asarray(state)
    ends = np.asarray(episode_ends)
    starts = np.concatenate([[0], ends[:-1]])
    phases = np.zeros(len(state), dtype=np.int8)
    infos = []
    for s, e in zip(starts, ends):
        info = detect_episode_phases(state[s:e], **kwargs)
        phases[s:e] = info['phases']
        infos.append(info)
    return phases, infos


def phase_of_windows(replay_buffer, sampler_indices, **kwargs):
    """Phase label for each sampled window, keyed on its FIRST frame.

    `sampler_indices[i][0]` is the global buffer index of window i (see
    SequenceSampler.create_indices), so a window is labelled by the phase it
    starts in - the observation the policy conditions on.
    """
    phases, infos = detect_dataset_phases(
        replay_buffer['state'], replay_buffer.episode_ends[:], **kwargs)
    idx = np.asarray([t[0] for t in sampler_indices])
    return phases[idx], infos