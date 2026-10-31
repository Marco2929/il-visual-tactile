#!/usr/bin/env python3
"""Quality assessment for extracted episodes.

Imported and driven by extract_episodes.py; it lives here only to keep the
extractor focused on extraction. Nothing in this module reads the SVO or the
bag again - it works on the arrays the extractor already has in memory, plus
the written PNGs - so running it costs almost nothing.

Per episode it writes:
    qa_report.txt            human-readable summary
    qa_report.json           machine-readable metrics
    qa_timeline.png          per-frame sync / force / motion timeline
    sync_histogram_*.png     the per-stream histograms

Per collection:
    qa_summary.txt / .json   one line per episode + recurring findings
    qa_summary.png           cross-episode comparison

Checks the extractor cannot easily make itself:

  * ZED frame drops. The saved arrays stay self-consistent when a frame is
    lost, so nothing complains - but the real time step doubles, which is a
    velocity outlier downstream.
  * Byte-identical consecutive wrist images. This is the ground truth for
    stale duplicates: no timestamps, no thresholds. After trimming it should
    be zero, so it doubles as a check that the trim worked.
  * Force plausibility: coverage, rail clipping, dead channels/fingertips.
  * State continuity: MAD-based jump detection, frozen (carried-forward) rows.
"""

import glob
import hashlib
import json
import os

import numpy as np
from termcolor import cprint

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False

HAND_FT_DIM = 30
N_FINGERTIPS = 5

# --- Thresholds --------------------------------------------------------
# Explicit on purpose: a QA tool whose limits are buried in code cannot be
# argued with. They are echoed into every report so a result stays readable
# later even if these change.
THRESHOLDS = {
    # ZED->ROS state sync. Sub-frame at 60 fps is the design target.
    "state_gap_mean_ms": 5.0,
    "state_gap_max_ms": 50.0,
    # ZED->wrist sync, startup ramp excluded. Half a wrist period (~33 ms at
    # 30 fps) is the most two streams can disagree while still describing the
    # same instant.
    "wrist_gap_mean_ms": 20.0,
    "wrist_gap_max_ms": 33.0,
    # Timeline integrity.
    "frame_period_tolerance": 1.5,   # x median before it counts as a drop
    "max_dropped_frames": 2,
    "max_duplicate_wrist": 0,        # after trimming, expect none
    # Depth. A 16-bit PNG in mm cannot be "wrong" dimensionally without the
    # extractor being broken, so the dtype/shape checks are hard failures.
    # The hole fraction is a WARN: it is a property of the scene (white foil
    # defeats stereo matching), not of the extraction, so it has to be judged
    # rather than rejected.
    # Applies to the ROI below, NOT the full frame. Measured over the whole
    # image this number is dominated by background that no policy ever looks
    # at (sky, far wall, out-of-range floor), so it warned on every episode
    # regardless of data quality.
    "max_depth_invalid_pct": 10.0,
    # y0, y1, x0, x1 in the stored depth resolution. Covers the workspace
    # around the foil/pallet, which is the only region where a hole actually
    # costs the policy information.
    "depth_roi": [120, 240, 180, 400],
    # Signals.
    "min_force_coverage_pct": 95.0,
    "max_force_saturation_pct": 1.0,
    "min_episode_frames": 30,
    # State continuity. The state vector mixes units - arm joints in radians,
    # hand joints in degrees - so one statistical threshold cannot serve both.
    # These are per-frame physical limits at the output rate: a step beyond
    # them implies a lost frame or a decoding error, not real motion.
    "max_arm_step_rad": 0.35,    # ~7 rad/s at 20 fps
    "max_hand_step_deg": 20.0,   # ~400 deg/s at 20 fps
}

# State layout, mirroring extract_episodes.py.
ARM_DIM = 6
HAND_JOINT_DIM = 20


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _stats(values):
    """Summarise a 1-D array, tolerating emptiness."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None,
                "max": None, "std": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def _fmt(s, unit="ms"):
    if not s or not s["n"]:
        return "n/a"
    return (f"mean {s['mean']:.2f}{unit} | median {s['median']:.2f}{unit} | "
            f"p95 {s['p95']:.2f}{unit} | max {s['max']:.2f}{unit} | "
            f"std {s['std']:.2f}{unit}")


# ═══════════════════════════════════════════════════════════════
# Sync histogram (used live by the extractor)
# ═══════════════════════════════════════════════════════════════

def sync_report(episode_dir, gaps_ms, label="Camera-to-ROS", verbose=True):
    """Terminal stats + histogram for one stream pair."""
    gaps = np.asarray(gaps_ms, dtype=np.float64).ravel()
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size == 0:
        return

    s = _stats(gaps)
    if verbose:
        cprint(f"\n--- {label} Sync Profile ---", "cyan")
        cprint(f"  Avg: {s['mean']:.2f}ms | Max: {s['max']:.2f}ms | "
               f"Jitter: {s['std']:.2f}ms",
               "green" if s["max"] < 10 else "yellow")

    if not PLOTS_AVAILABLE:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(gaps, bins=50, color="royalblue", edgecolor="black", alpha=0.7)
    plt.axvline(s["mean"], color="red", linestyle="dashed",
                label=f"Mean: {s['mean']:.2f}ms")
    plt.title(f"{label} Sync Delay")
    plt.xlabel("Gap (ms)")
    plt.ylabel("Frames")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    safe = label.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(os.path.join(episode_dir, f"sync_histogram_{safe}.png"))
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Individual checks
# ═══════════════════════════════════════════════════════════════

def _check_timeline(timestamps_ns, tol):
    """Frame-period regularity: did the ZED drop frames mid-episode?"""
    out = {"n_frames": int(len(timestamps_ns)), "median_period_ms": None,
           "dropped_frames": 0, "hiccups": [], "duration_s": None,
           "effective_fps": None, "period_stats": _stats([])}
    if len(timestamps_ns) < 3:
        return out

    ts = np.asarray(timestamps_ns, dtype=np.int64)
    dt_ms = np.diff(ts) / 1e6
    median = float(np.median(dt_ms))
    out["median_period_ms"] = median
    out["period_stats"] = _stats(dt_ms)
    out["duration_s"] = float((ts[-1] - ts[0]) / 1e9)
    if out["duration_s"] > 0:
        out["effective_fps"] = float((len(ts) - 1) / out["duration_s"])
    if median <= 0:
        return out

    idx = np.where(dt_ms > median * tol)[0]
    # Weight by gap length: a 3x period means two frames were lost, not one.
    out["dropped_frames"] = (int(np.round(np.sum(dt_ms[idx] / median - 1.0)))
                             if idx.size else 0)
    out["hiccups"] = [{"after_frame": int(i), "gap_ms": float(dt_ms[i])}
                      for i in idx[:20]]
    return out


def _check_wrist_images(episode_dir):
    """Byte-level duplicate detection on the written wrist PNGs.

    Independent of every clock and threshold: two identical files are the
    same photograph. After trimming this should report zero pairs.
    """
    out = {"n_images": 0, "n_unique_images": None, "duplicate_pairs": []}
    d = os.path.join(episode_dir, "wrist_rgb")
    if not os.path.isdir(d):
        return out
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    out["n_images"] = len(files)
    digests = []
    for p in files:
        with open(p, "rb") as f:
            digests.append(hashlib.md5(f.read()).hexdigest())
    out["n_unique_images"] = len(set(digests))
    out["duplicate_pairs"] = [{"frames": [i - 1, i]}
                              for i in range(1, len(digests))
                              if digests[i] == digests[i - 1]][:20]
    return out


def _check_force(force):
    """Coverage, rail clipping and dead channels in the fingertip F/T."""
    out = {"n_frames": int(len(force)), "coverage_pct": 0.0, "abs_max": 0.0,
           "saturated_pct": 0.0, "dead_channels": [], "dead_tips": [],
           "per_tip_abs_max": []}
    if force.size == 0:
        return out

    nonzero = int(np.count_nonzero(np.any(force != 0, axis=1)))
    out["coverage_pct"] = 100.0 * nonzero / len(force)
    out["abs_max"] = float(np.abs(force).max())

    if force.ndim == 2:
        # A channel that never moves is an unplugged or failed sensor: it
        # trains as a constant and hides inside an aggregate magnitude.
        spread = force.max(axis=0) - force.min(axis=0)
        out["dead_channels"] = [int(i) for i in np.where(spread == 0)[0]]
        if force.shape[1] == HAND_FT_DIM:
            tips = force.reshape(len(force), N_FINGERTIPS, 6)
            out["per_tip_abs_max"] = [float(v)
                                      for v in np.abs(tips).max(axis=(0, 2))]
            out["dead_tips"] = [
                int(t) for t in range(N_FINGERTIPS)
                if np.all(tips[:, t, :].max(axis=0)
                          - tips[:, t, :].min(axis=0) == 0)
            ]

    # Clipping at the rail looks like strong contact but carries no gradient,
    # so it is flagged separately from merely large forces.
    if out["abs_max"] > 0:
        at_rail = np.isclose(np.abs(force), out["abs_max"], rtol=1e-6)
        out["saturated_pct"] = float(100.0 * at_rail.sum() / force.size)
    return out


def _check_state(state, th):
    """Finiteness and continuity of the robot state.

    Continuity is checked against PHYSICAL per-frame limits, split by unit
    group. A purely statistical bound (e.g. MAD over the whole vector) does
    not work here: the arm is in radians and the hand in degrees, and the step
    distribution is bimodal (the robot is mostly still, then moves fast), so
    a robust spread estimated over the still phase flags ordinary motion.
    """
    out = {"n_frames": int(len(state)), "dim": None, "has_nan": False,
           "has_inf": False, "max_step": 0.0, "max_arm_step": None,
           "max_hand_step": None, "n_step_outliers": 0, "step_outliers": [],
           "frozen_frames": 0, "step_stats": _stats([])}
    if state.size == 0:
        return out

    out["dim"] = int(state.shape[1]) if state.ndim == 2 else 1
    out["has_nan"] = bool(np.isnan(state).any())
    out["has_inf"] = bool(np.isinf(state).any())
    if len(state) < 3:
        return out

    diffs = np.diff(state, axis=0)
    steps = np.abs(diffs)
    out["max_step"] = float(steps.max())
    out["step_stats"] = _stats(steps.max(axis=1))

    # Identical consecutive rows mean the recorder stopped delivering updates
    # and the extractor carried the last value forward.
    out["frozen_frames"] = int(np.sum(np.all(diffs == 0, axis=1)))

    if state.ndim != 2:
        return out

    # Evaluate each unit group against its own limit, then merge.
    groups = []
    if state.shape[1] >= ARM_DIM:
        groups.append(("arm", slice(0, ARM_DIM), th["max_arm_step_rad"]))
    if state.shape[1] >= ARM_DIM + HAND_JOINT_DIM:
        groups.append(("hand", slice(ARM_DIM, ARM_DIM + HAND_JOINT_DIM),
                       th["max_hand_step_deg"]))
    if not groups:
        groups = [("state", slice(0, state.shape[1]), th["max_hand_step_deg"])]

    violations = {}
    for label, sl, limit in groups:
        g = steps[:, sl]
        if g.size == 0:
            continue
        per_frame = g.max(axis=1)
        if label == "arm":
            out["max_arm_step"] = float(per_frame.max())
        elif label == "hand":
            out["max_hand_step"] = float(per_frame.max())
        for i in np.where(per_frame > limit)[0]:
            rec = violations.setdefault(int(i), {"after_frame": int(i)})
            rec[label] = float(per_frame[i])
            rec[f"{label}_limit"] = float(limit)

    ordered = [violations[k] for k in sorted(violations)]
    out["n_step_outliers"] = len(ordered)
    out["step_outliers"] = ordered[:20]   # list truncated; count is exact
    return out


def _check_depth(episode_dir, sample_stride=20, roi=None):
    """Validate the stored 16-bit depth PNGs against the RGB they pair with.

    Only every `sample_stride`-th frame is opened: these properties are set by
    the extraction code path, not by the individual frame, so a sample is
    enough to catch a broken run.

    `invalid_pct` is the headline number and is measured over `roi` only. The
    task is pinching a WHITE foil, and homogeneous bright surfaces are exactly
    where stereo matching fails for lack of correspondences - so a high hole
    fraction IN THE WORKSPACE means the depth signal cannot carry slip
    detection. Measured over the full frame the same number is dominated by
    background the policy never uses, which is why that variant is reported
    separately as `invalid_pct_full` rather than being thresholded.

    Keeping both is deliberate: a clean ROI with a full-frame fraction that
    jumps from ~1.5% to 60% means the camera moved or the range changed, which
    is invisible if only the ROI is tracked.
    """
    out = {"available": False, "n_files": 0, "n_sampled": 0, "dtype": None,
           "shape": None, "rgb_shape": None, "shape_matches_rgb": None,
           "invalid_pct": None, "invalid_pct_full": None, "roi": roi,
           "roi_clamped": None, "valid_min_mm": None, "valid_max_mm": None,
           "valid_median_mm": None}

    d = os.path.join(episode_dir, "front_depth")
    if not os.path.isdir(d):
        return out
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    out["n_files"] = len(files)
    if not files:
        return out

    try:
        import cv2
    except ImportError:
        return out

    rgb_files = sorted(glob.glob(os.path.join(episode_dir, "front_rgb",
                                              "*.png")))
    if rgb_files:
        rgb = cv2.imread(rgb_files[0], cv2.IMREAD_UNCHANGED)
        if rgb is not None:
            out["rgb_shape"] = [int(v) for v in rgb.shape[:2]]

    roi_fracs, full_fracs, mins, maxs, medians = [], [], [], [], []
    for p in files[::max(1, sample_stride)]:
        # IMREAD_UNCHANGED is mandatory: the default flag silently converts
        # the 16-bit single channel to 8-bit BGR and destroys the mm values.
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        out["available"] = True
        out["dtype"] = str(img.dtype)
        out["shape"] = [int(v) for v in img.shape[:2]]
        out["n_sampled"] += 1

        # Full frame: reported, never thresholded (see docstring).
        full_fracs.append(1.0 - (img > 0).sum() / img.size)

        patch = img
        if roi:
            h, w = img.shape[:2]
            y0, y1, x0, x1 = roi
            # Clamp instead of trusting the config: a ROI configured for a
            # different depth resolution would otherwise slice to an empty
            # array and report invalid_pct = nan with no explanation.
            cy0, cy1 = max(0, min(y0, h)), max(0, min(y1, h))
            cx0, cx1 = max(0, min(x0, w)), max(0, min(x1, w))
            if cy1 > cy0 and cx1 > cx0:
                patch = img[cy0:cy1, cx0:cx1]
                out["roi_clamped"] = ([cy0, cy1, cx0, cx1]
                                      if [cy0, cy1, cx0, cx1] != list(roi)
                                      else False)
            else:
                # Degenerate ROI: fall back to the full frame rather than
                # silently producing no number at all.
                out["roi_clamped"] = "degenerate, using full frame"

        roi_fracs.append(1.0 - (patch > 0).sum() / patch.size)
        valid = patch[patch > 0]
        if valid.size:
            mins.append(float(valid.min()))
            maxs.append(float(valid.max()))
            medians.append(float(np.median(valid)))

    if roi_fracs:
        out["invalid_pct"] = float(100.0 * np.mean(roi_fracs))
    if full_fracs:
        out["invalid_pct_full"] = float(100.0 * np.mean(full_fracs))
    if mins:
        out["valid_min_mm"] = float(np.min(mins))
        out["valid_max_mm"] = float(np.max(maxs))
        out["valid_median_mm"] = float(np.median(medians))
    if out["shape"] and out["rgb_shape"]:
        out["shape_matches_rgb"] = out["shape"] == out["rgb_shape"]
    return out


def _check_counts(episode_dir, n_frames):
    """Every per-frame artifact must have exactly n_frames entries."""
    out = {"n_frames": int(n_frames), "dirs": {}, "mismatches": []}
    for name, pattern in (("front_rgb", "*.png"), ("front_depth", "*.png"),
                          ("front_pc", "*.npy"), ("wrist_rgb", "*.png")):
        d = os.path.join(episode_dir, name)
        if not os.path.isdir(d):
            out["dirs"][name] = None
            continue
        count = len(glob.glob(os.path.join(d, pattern)))
        out["dirs"][name] = count
        if count != n_frames:
            out["mismatches"].append({"dir": name, "found": count,
                                      "expected": int(n_frames)})
    return out


# ═══════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════

def _verdict(m, th):
    """FAIL = would corrupt training as-is. WARN = worth knowing."""
    fails, warns = [], []

    s, f, tl, w = m["state"], m["force"], m["timeline"], m["wrist"]
    d = m.get("depth", {})

    if s["has_nan"] or s["has_inf"]:
        fails.append("state.npy contains NaN/Inf")
    if f["n_frames"] == 0:
        fails.append("force.npy is empty")
    for mm in m["counts"]["mismatches"]:
        fails.append(f"{mm['dir']}: {mm['found']} files != "
                     f"{mm['expected']} frames")
    if tl["n_frames"] < th["min_episode_frames"]:
        fails.append(f"only {tl['n_frames']} frames "
                     f"(< {th['min_episode_frames']})")

    sg = m["state_sync"]["gap_stats"]
    if sg["n"]:
        if sg["mean"] > th["state_gap_mean_ms"]:
            fails.append(f"ZED-to-ROS mean gap {sg['mean']:.1f}ms "
                         f"> {th['state_gap_mean_ms']}ms")
        if sg["max"] > th["state_gap_max_ms"]:
            warns.append(f"ZED-to-ROS max gap {sg['max']:.1f}ms "
                         f"> {th['state_gap_max_ms']}ms")

    if w["available"]:
        wt = w["gap_stats"]
        if wt and wt["n"]:
            if wt["mean"] > th["wrist_gap_mean_ms"]:
                fails.append(f"ZED-to-wrist mean gap {wt['mean']:.1f}ms "
                             f"> {th['wrist_gap_mean_ms']}ms")
            if wt["max"] > th["wrist_gap_max_ms"]:
                warns.append(f"ZED-to-wrist max gap {wt['max']:.1f}ms "
                             f"> {th['wrist_gap_max_ms']}ms")
        if w["n_stale_start"] > 0:
            fails.append(f"{w['n_stale_start']} leading frame(s) precede the "
                         f"first wrist message (bag {w['lead_ms']:.0f}ms late) "
                         f"and were NOT trimmed")

    n_dup = len(w["duplicate_pairs"])
    if n_dup > th["max_duplicate_wrist"]:
        fails.append(f"{n_dup} byte-identical consecutive wrist image(s), "
                     f"first at frames {w['duplicate_pairs'][0]['frames']}")

    if d.get("available"):
        if d.get("dtype") != "uint16":
            fails.append(f"front_depth dtype is {d.get('dtype')}, expected "
                         f"uint16 (mm)")
        if d.get("shape_matches_rgb") is False:
            fails.append(f"front_depth {d.get('shape')} != front_rgb "
                         f"{d.get('rgb_shape')} - depth is not pixel-aligned "
                         f"with the RGB it is paired with")
        inv = d.get("invalid_pct")
        if inv is not None and inv > th["max_depth_invalid_pct"]:
            full = d.get("invalid_pct_full")
            full_txt = f"{full:.1f}%" if full is not None else "n/a"
            warns.append(f"{inv:.1f}% invalid depth in ROI {d.get('roi')} "
                         f"> {th['max_depth_invalid_pct']}% "
                         f"(full frame {full_txt}) - check whether the holes "
                         f"sit on the foil, which would mean depth cannot "
                         f"carry slip detection")

    if tl["dropped_frames"] > th["max_dropped_frames"]:
        warns.append(f"{tl['dropped_frames']} dropped ZED frame(s) "
                     f"(max period {tl['period_stats']['max']:.0f}ms vs "
                     f"median {tl['median_period_ms']:.0f}ms)")

    if f["n_frames"]:
        if f["abs_max"] == 0:
            fails.append("force.npy is all zeros - no fingertip F/T recorded")
        elif f["coverage_pct"] < th["min_force_coverage_pct"]:
            fails.append(f"force coverage {f['coverage_pct']:.1f}% "
                         f"< {th['min_force_coverage_pct']}%")
        if f["saturated_pct"] > th["max_force_saturation_pct"]:
            warns.append(f"{f['saturated_pct']:.2f}% of F/T samples at the "
                         f"rail (|{f['abs_max']:.1f}|) - possible clipping")
        if f["dead_tips"]:
            fails.append(f"fingertip(s) {f['dead_tips']} never change "
                         f"- sensor offline")
        elif f["dead_channels"]:
            warns.append(f"{len(f['dead_channels'])} constant F/T channel(s): "
                         f"{f['dead_channels'][:6]}")

    if s["n_step_outliers"]:
        first = s["step_outliers"][0]
        detail = ", ".join(f"{k} {first[k]:.3f} > {first[k + '_limit']:.2f}"
                           for k in ("arm", "hand") if k in first)
        warns.append(f"{s['n_step_outliers']} state step(s) above the physical "
                     f"per-frame limit; first after frame "
                     f"{first['after_frame']} ({detail})")
    if s["n_frames"] and s["frozen_frames"] > 0.1 * s["n_frames"]:
        warns.append(f"{s['frozen_frames']}/{s['n_frames']} frames repeat the "
                     f"previous state (carried forward)")

    status = "FAIL" if fails else ("WARN" if warns else "PASS")
    return status, fails, warns


# ═══════════════════════════════════════════════════════════════
# Entry point used by the extractor
# ═══════════════════════════════════════════════════════════════

def assess_episode(episode_dir, state=None, force=None, timestamps=None,
                   state_gaps_ms=None, wrist=None, make_plots=True,
                   verbose=True):
    """Assess one episode and write its reports.

    Arrays are passed in from the extractor to avoid re-reading anything;
    any that are omitted are loaded from disk, which is what the wrist-only
    backfill path needs.

    `wrist` is the dict returned by extract_wrist_camera(), already adjusted
    for any trimming that happened.
    """
    name = os.path.basename(episode_dir.rstrip("/"))

    def _load(arr, fname):
        if arr is not None:
            return np.asarray(arr)
        p = os.path.join(episode_dir, fname)
        return np.load(p) if os.path.exists(p) else np.array([])

    state = _load(state, "state.npy")
    force = _load(force, "force.npy")
    timestamps = _load(timestamps, "frame_timestamps_ns.npy")

    if state.size == 0 or timestamps.size == 0:
        result = {"episode": name, "status": "FAIL",
                  "fails": ["state.npy or frame_timestamps_ns.npy missing "
                            "or empty"],
                  "warns": [], "metrics": None}
        _print_verdict(result, verbose)
        return result

    # sync_gaps.npy has one entry per robot_state update, not per frame, so it
    # is only ever summarised as a distribution - never indexed by frame.
    if state_gaps_ms is None:
        p = os.path.join(episode_dir, "sync_gaps.npy")
        state_gaps_ms = np.load(p) if os.path.exists(p) else np.array([])

    wrist = dict(wrist or {})
    img = _check_wrist_images(episode_dir)
    wrist_m = {
        "available": bool(wrist.get("available", False)),
        "n_wrist_msgs": int(wrist.get("n_msgs", 0)),
        "wrist_period_ms": wrist.get("period_ms"),
        "lead_ms": wrist.get("lead_ms"),
        "n_stale_start": int(wrist.get("n_stale_start", 0)),
        "trimmed_frames": int(wrist.get("trimmed_frames", 0)),
        "gap_stats": _stats(wrist.get("gaps_ms", [])),
        **img,
    }

    timeline = _check_timeline(timestamps,
                               THRESHOLDS["frame_period_tolerance"])

    metrics = {
        "timeline": timeline,
        "wrist": wrist_m,
        "depth": _check_depth(episode_dir, roi=THRESHOLDS["depth_roi"]),
        "force": _check_force(force),
        "state": _check_state(state, THRESHOLDS),
        "counts": _check_counts(episode_dir, len(state)),
        "state_sync": {
            "n_updates": int(len(state_gaps_ms)),
            "gap_stats": _stats(state_gaps_ms),
            "note": "one entry per robot_state update, not per frame",
        },
    }

    status, fails, warns = _verdict(metrics, THRESHOLDS)
    result = {"episode": name, "status": status, "fails": fails,
              "warns": warns, "metrics": metrics}

    _write_episode_report(episode_dir, result)
    if make_plots and PLOTS_AVAILABLE:
        _plot_episode(episode_dir, name, metrics, state, force, timestamps,
                      np.asarray(wrist.get("gaps_ms", []), dtype=np.float64))
    _print_verdict(result, verbose)
    return result


def _print_verdict(result, verbose):
    if not verbose:
        return
    col = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[result["status"]]
    cprint(f"  [QA] {result['status']}", col, attrs=["bold"])
    for x in result["fails"]:
        cprint(f"         FAIL: {x}", "red")
    for x in result["warns"]:
        cprint(f"         WARN: {x}", "yellow")


# ═══════════════════════════════════════════════════════════════
# Reports
# ═══════════════════════════════════════════════════════════════

def _write_episode_report(episode_dir, result):
    m = result["metrics"]
    tl, w, f, s = m["timeline"], m["wrist"], m["force"], m["state"]

    def opt(label, value, fmt="{:.2f}", suffix=""):
        if value is None:
            return f"  {label:<18}: n/a"
        return f"  {label:<18}: {fmt.format(value)}{suffix}"

    L = ["=" * 64, f"QA REPORT  {result['episode']}", "=" * 64,
         f"Verdict: {result['status']}", ""]
    if result["fails"]:
        L.append("FAIL:")
        L += [f"  - {x}" for x in result["fails"]]
        L.append("")
    if result["warns"]:
        L.append("WARN:")
        L += [f"  - {x}" for x in result["warns"]]
        L.append("")

    L += ["-- Timeline " + "-" * 51,
          f"  {'frames':<18}: {tl['n_frames']}",
          opt("duration", tl["duration_s"], suffix=" s"),
          opt("frame period", tl["median_period_ms"], suffix=" ms (median)"),
          opt("effective fps", tl["effective_fps"]),
          f"  {'dropped frames':<18}: {tl['dropped_frames']}"]
    for h in tl["hiccups"][:5]:
        L.append(f"      after frame {h['after_frame']:>5}: "
                 f"{h['gap_ms']:.1f} ms")

    L += ["", "-- ZED-to-ROS (state) " + "-" * 41,
          f"  {'updates':<18}: {m['state_sync']['n_updates']}",
          f"  {'gap':<18}: {_fmt(m['state_sync']['gap_stats'])}",
          f"  {'note':<18}: {m['state_sync']['note']}",
          "", "-- ZED-to-Wrist " + "-" * 47]
    if w["available"]:
        L += [f"  {'wrist msgs':<18}: {w['n_wrist_msgs']}",
              opt("wrist period", w["wrist_period_ms"], suffix=" ms"),
              opt("bag start lead", w["lead_ms"], fmt="{:+.1f}",
                  suffix=" ms (positive = wrist began late)"),
              f"  {'stale lead frames':<18}: {w['n_stale_start']}",
              f"  {'trimmed frames':<18}: {w['trimmed_frames']}",
              f"  {'gap':<18}: {_fmt(w['gap_stats'])}"]
    else:
        L.append("  unavailable (no bag / rosbags missing)")
    img = f"  {'images':<18}: {w['n_images']}"
    if w["n_unique_images"] is not None:
        img += f" ({w['n_unique_images']} unique)"
    L += [img, f"  {'identical pairs':<18}: {len(w['duplicate_pairs'])}"]
    for d in w["duplicate_pairs"][:5]:
        L.append(f"      frames {d['frames']}")

    d = m.get("depth", {})
    L += ["", "-- Front depth " + "-" * 48]
    if d.get("available"):
        L += [f"  {'files':<18}: {d['n_files']} "
              f"({d['n_sampled']} sampled)",
              f"  {'dtype':<18}: {d['dtype']}",
              f"  {'shape (HxW)':<18}: {d['shape']} "
              f"(front_rgb {d['rgb_shape']}, "
              f"match={d['shape_matches_rgb']})",
              f"  {'roi (y0,y1,x0,x1)':<18}: {d.get('roi')}"
              + (f"  [clamped -> {d['roi_clamped']}]"
                 if d.get("roi_clamped") else ""),
              opt("invalid in roi", d["invalid_pct"], suffix=" %"),
              opt("invalid full", d.get("invalid_pct_full"), suffix=" %"),
              opt("valid min", d["valid_min_mm"], fmt="{:.0f}", suffix=" mm"),
              opt("valid median", d["valid_median_mm"], fmt="{:.0f}",
                  suffix=" mm"),
              opt("valid max", d["valid_max_mm"], fmt="{:.0f}", suffix=" mm")]
    else:
        L.append("  absent (front_depth/ not written)")

    L += ["", "-- Force " + "-" * 54,
          f"  {'coverage':<18}: {f['coverage_pct']:.1f}%",
          f"  {'|F/T| max':<18}: {f['abs_max']:.3f}",
          f"  {'at-rail samples':<18}: {f['saturated_pct']:.3f}%",
          f"  {'per-tip |max|':<18}: "
          + (", ".join(f"{v:.2f}" for v in f["per_tip_abs_max"])
             if f["per_tip_abs_max"] else "n/a"),
          f"  {'dead channels':<18}: {f['dead_channels'] or 'none'}",
          f"  {'dead fingertips':<18}: {f['dead_tips'] or 'none'}",
          "", "-- State " + "-" * 54,
          f"  {'shape':<18}: ({s['n_frames']}, {s['dim']})",
          f"  {'NaN / Inf':<18}: {s['has_nan']} / {s['has_inf']}",
          f"  {'max step (any)':<18}: {s['max_step']:.4f}",
          opt("max arm step", s["max_arm_step"], fmt="{:.4f}",
              suffix=f" rad (limit {THRESHOLDS['max_arm_step_rad']})"),
          opt("max hand step", s["max_hand_step"], fmt="{:.3f}",
              suffix=f" deg (limit {THRESHOLDS['max_hand_step_deg']})"),
          f"  {'step outliers':<18}: {s['n_step_outliers']}",
          f"  {'frozen frames':<18}: {s['frozen_frames']}",
          "", "-- File counts " + "-" * 49]
    for name, count in m["counts"]["dirs"].items():
        L.append(f"  {name:<18}: {'absent' if count is None else count}")
    L += ["", "-- Thresholds " + "-" * 49]
    L += [f"  {k:<28}{v}" for k, v in THRESHOLDS.items()]
    L.append("")

    with open(os.path.join(episode_dir, "qa_report.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    payload = dict(result)
    payload["thresholds"] = THRESHOLDS
    with open(os.path.join(episode_dir, "qa_report.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)


def _plot_episode(episode_dir, name, m, state, force, timestamps, wrist_gaps):
    """Per-frame timeline: spikes are located, not merely counted."""
    n = len(timestamps)
    if n < 3:
        return
    x = np.arange(n)
    fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    dt = np.concatenate([[np.nan], np.diff(timestamps) / 1e6])
    med = m["timeline"]["median_period_ms"]
    ax[0].plot(x, dt, lw=0.9, color="steelblue")
    if med:
        ax[0].axhline(med, color="green", ls="--", lw=1,
                      label=f"median {med:.1f} ms")
        ax[0].axhline(med * THRESHOLDS["frame_period_tolerance"], color="red",
                      ls=":", lw=1, label="drop threshold")
        ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_ylabel("frame dt [ms]")
    ax[0].set_title(f"{name} - QA timeline")

    if wrist_gaps.size == n:
        ax[1].plot(x, wrist_gaps, lw=0.9, color="darkorange")
        ax[1].axhline(THRESHOLDS["wrist_gap_max_ms"], color="red", ls=":",
                      lw=1, label=f"{THRESHOLDS['wrist_gap_max_ms']:.0f} ms")
        ax[1].legend(fontsize=8, loc="upper right")
    else:
        ax[1].text(0.5, 0.5, "wrist sync unavailable", ha="center",
                   transform=ax[1].transAxes, color="grey")
    ax[1].set_ylabel("ZED->wrist [ms]")

    if force.ndim == 2 and force.shape[1] == HAND_FT_DIM and len(force) == n:
        tips = force.reshape(n, N_FINGERTIPS, 6)
        for t in range(N_FINGERTIPS):
            ax[2].plot(x, np.linalg.norm(tips[:, t, :3], axis=1), lw=0.9,
                       label=f"tip {t}")
        ax[2].legend(fontsize=8, ncol=5, loc="upper right")
    elif len(force) == n and force.size:
        ax[2].plot(x, np.abs(force).max(axis=1), lw=0.9, color="firebrick")
    ax[2].set_ylabel("|force| [N]")

    if len(state) == n and n > 1:
        steps = np.concatenate([[np.nan],
                                np.abs(np.diff(state, axis=0)).max(axis=1)])
        ax[3].plot(x, steps, lw=0.9, color="purple")
        for o in m["state"]["step_outliers"]:
            ax[3].axvline(o["after_frame"] + 1, color="red", alpha=0.5, lw=0.8)
        if m["state"]["n_step_outliers"] > len(m["state"]["step_outliers"]):
            ax[3].set_title(f"{m['state']['n_step_outliers']} step outliers "
                            f"(first 20 marked)", fontsize=8)
    ax[3].set_ylabel("max |d state|")
    ax[3].set_xlabel("frame")

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(episode_dir, "qa_timeline.png"), dpi=110)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Collection summary
# ═══════════════════════════════════════════════════════════════

def write_summary(data_dir, results, make_plots=True, verbose=True):
    """Write qa_summary.{txt,json} (+png) for a whole collection."""
    if not results:
        return {"PASS": 0, "WARN": 0, "FAIL": 0}

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    rows = []
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        m = r["metrics"]
        if m is None:
            rows.append(f"{r['episode']:<24} {r['status']:<5} (no data)")
            continue
        tl, w = m["timeline"], m["wrist"]
        g = w["gap_stats"]
        lead = f"{w['lead_ms']:>7.1f}ms" if w["lead_ms"] is not None else "    n/a"
        wgap = f"{g['mean']:.1f}/{g['max']:.1f}ms" if g["n"] else "n/a"
        rows.append(f"{r['episode']:<24} {r['status']:<5} "
                    f"n={tl['n_frames']:<5} "
                    f"drop={tl['dropped_frames']:<3} "
                    f"lead={lead} "
                    f"trim={w['trimmed_frames']:<3} "
                    f"dup={len(w['duplicate_pairs']):<3} "
                    f"wgap={wgap:<14} "
                    f"force={m['force']['coverage_pct']:.0f}%")

    L = ["=" * 78, f"QA SUMMARY  {data_dir}", "=" * 78, "",
         f"episodes: {len(results)}   PASS {counts['PASS']}  "
         f"WARN {counts['WARN']}  FAIL {counts['FAIL']}", ""]
    L += rows
    L += ["", "-" * 78, "Recurring findings (systematic if on most episodes):",
          ""]

    # Collapse each RUN of digits (and decimals) to a single placeholder, so
    # "frame 1" and "frame 30" land in the same bucket instead of differing
    # by digit count.
    def _shape(text):
        out, prev_digit = [], False
        for ch in text:
            if ch.isdigit() or (ch == "." and prev_digit):
                if not prev_digit:
                    out.append("N")
                prev_digit = True
            else:
                out.append(ch)
                prev_digit = False
        return "".join(out)

    tally = {}
    for r in results:
        for sev, items in (("FAIL", r["fails"]), ("WARN", r["warns"])):
            for it in items:
                tally.setdefault((sev, _shape(it)), []).append(r["episode"])
    if tally:
        for (sev, key), eps in sorted(tally.items(),
                                      key=lambda kv: -len(kv[1])):
            L.append(f"  [{sev}] {len(eps)}/{len(results)}: {key}")
    else:
        L.append("  none - all episodes clean")
    L.append("")

    text = "\n".join(L) + "\n"
    with open(os.path.join(data_dir, "qa_summary.txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(data_dir, "qa_summary.json"), "w") as fh:
        json.dump({"thresholds": THRESHOLDS, "counts": counts,
                   "episodes": results}, fh, indent=2, default=float)

    if make_plots and PLOTS_AVAILABLE:
        _plot_summary(data_dir, results)
    if verbose:
        print()
        print(text)
    return counts


def _plot_summary(data_dir, results):
    valid = [r for r in results if r["metrics"]]
    if not valid:
        return
    names = [r["episode"].replace("episode_", "").replace("_success", "")
             for r in valid]
    x = np.arange(len(valid))
    colour = {"PASS": "seagreen", "WARN": "goldenrod", "FAIL": "firebrick"}
    cols = [colour[r["status"]] for r in valid]

    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    ax[0].bar(x, [r["metrics"]["timeline"]["n_frames"] for r in valid],
              color=cols)
    ax[0].set_ylabel("frames")
    ax[0].set_title(f"QA summary - {os.path.basename(data_dir.rstrip('/'))}")

    ax[1].bar(x, [r["metrics"]["wrist"]["lead_ms"] or 0 for r in valid],
              color=cols)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel("bag lead [ms]")

    mean_g = [(r["metrics"]["wrist"]["gap_stats"]["mean"] or 0)
              for r in valid]
    max_g = [(r["metrics"]["wrist"]["gap_stats"]["max"] or 0) for r in valid]
    ax[2].bar(x - 0.2, mean_g, 0.4, label="mean", color="steelblue")
    ax[2].bar(x + 0.2, max_g, 0.4, label="max", color="darkorange")
    ax[2].axhline(THRESHOLDS["wrist_gap_max_ms"], color="red", ls=":",
                  label="limit")
    ax[2].set_ylabel("ZED->wrist [ms]")
    ax[2].legend(fontsize=8)

    ax[3].bar(x, [r["metrics"]["force"]["coverage_pct"] for r in valid],
              color=cols)
    ax[3].axhline(THRESHOLDS["min_force_coverage_pct"], color="red", ls=":")
    ax[3].set_ylabel("force coverage [%]")
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(names, rotation=45, ha="right")

    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(data_dir, "qa_summary.png"), dpi=110)
    plt.close(fig)