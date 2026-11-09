#!/usr/bin/env python3
"""Generate, inspect, and manually refine grasp_state phase labels for rollouts.

Supports:
  1. Automatic proposal of 4-class grasp_state labels (0..3) for all rollouts
     from executed hand closure + EE z height + outcome.json verdict/failure_mode.
  2. Storing human-editable phase segments in each rollout's `outcome.json`
     under `"phases"` (with `"phases_manual": true` when edited by hand).
  3. Syncing per-frame `grasp_state` (int64) and ground-truth `progress` (float32)
     into `foil_lifting_rollouts_progress.zarr`.

Classes:
  0 = APPROACH       (open hand approaching or repositioning)
  1 = PINCH_CONTACT  (fingers closing on foil / contact at pallet)
  2 = LIFT_SUCCESS   (foil securely held and lifted)
  3 = MISS_OR_LOST   (failed attempt, empty grasp, or slipped foil)

Usage examples:
  # Propose phase labels for all 89 rollouts, write to outcome.json & sync to Zarr:
  python data/rollouts/tools/annotate_phases.py --sync

  # Show table of all rollouts (or only multi-attempt / retry rollouts):
  python data/rollouts/tools/annotate_phases.py --table
  python data/rollouts/tools/annotate_phases.py --table --retries-only

  # Manually set phase boundaries (in seconds or frames) for a specific rollout:
  python data/rollouts/tools/annotate_phases.py --ep 68 --set "0:0-14.2s, 3:14.2-19.5s, 0:19.5-25.0s, 1:25.0-26.2s, 2:26.2-end"
"""

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr

ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT_DIR / "data" / "datasets"
ROLLOUTS_DIR = ROOT_DIR / "data" / "rollouts"

for p in (DATASETS_DIR / "dataset_builder", ROLLOUTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from processing.progress import (
    GRASP_STATE_NAMES,
    STATE_APPROACH,
    STATE_LIFT_SUCCESS,
    STATE_MISS_OR_LOST,
    STATE_PINCH_CONTACT,
    build_phase_labels,
    build_progress_curve,
    get_curve_params,
    segment_grasp_attempts,
)
from viz.data import Rollout

DEFAULT_ROLLOUT_ZARR = DATASETS_DIR / "zarr_datasets" / "foil_lifting_rollouts_progress.zarr"
DEFAULT_RUNS_ROOT = ROLLOUTS_DIR / "runs"


def labels_to_segments(labels: np.ndarray, times_s: np.ndarray) -> List[Dict]:
    """Compress per-frame integer labels into contiguous phase segments."""
    T = len(labels)
    if T == 0:
        return []
    segs = []
    start = 0
    cur = int(labels[0])
    for i in range(1, T):
        if int(labels[i]) != cur:
            segs.append({
                "state": cur,
                "name": GRASP_STATE_NAMES[cur],
                "start_frame": int(start),
                "end_frame": int(i),
                "start_s": round(float(times_s[start]), 2),
                "end_s": round(float(times_s[i - 1]), 2),
            })
            start = i
            cur = int(labels[i])
    segs.append({
        "state": cur,
        "name": GRASP_STATE_NAMES[cur],
        "start_frame": int(start),
        "end_frame": int(T),
        "start_s": round(float(times_s[start]), 2),
        "end_s": round(float(times_s[T - 1]), 2),
    })
    return segs


def segments_to_labels(segs: List[Dict], num_frames: int, times_s: np.ndarray) -> np.ndarray:
    """Expand phase segments back into a per-frame (num_frames,) int64 array."""
    labels = np.full(num_frames, STATE_APPROACH, dtype=np.int64)
    for s in segs:
        st = int(s["state"])
        if "start_frame" in s and "end_frame" in s:
            a = max(0, min(num_frames, int(s["start_frame"])))
            b = max(a, min(num_frames, int(s["end_frame"])))
            labels[a:b] = st
        elif "start_s" in s and "end_s" in s:
            mask = (times_s >= float(s["start_s"]) - 1e-3) & (times_s <= float(s["end_s"]) + 1e-3)
            labels[mask] = st
    return labels


def parse_set_spec(spec: str, num_frames: int, times_s: np.ndarray) -> List[Dict]:
    """Parse user specification like '0:0-14.2s, 3:14.2-19.5s, 1:19.5-21.0s, 2:21.0-end'."""
    labels = np.full(num_frames, STATE_APPROACH, dtype=np.int64)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if ":" not in part or "-" not in part:
            raise ValueError(f"Invalid segment '{part}'. Expected format STATE:START-END (e.g. '1:12.0-14.5s' or '2:300-end')")
        st_str, span_str = part.split(":", 1)
        st = int(st_str.strip())
        if st not in (0, 1, 2, 3):
            raise ValueError(f"State must be 0, 1, 2, or 3, got {st}")
        a_str, b_str = span_str.strip().split("-", 1)
        is_sec = a_str.endswith("s") or b_str.endswith("s") or ("." in a_str) or ("." in b_str)
        a_clean = a_str.rstrip("s").strip()
        b_clean = b_str.rstrip("s").strip()

        if is_sec:
            t_start = 0.0 if a_clean in ("0", "start") else float(a_clean)
            t_end = float(times_s[-1]) + 1.0 if b_clean == "end" else float(b_clean)
            idx_a = int(np.searchsorted(times_s, t_start, side="left"))
            idx_b = num_frames if b_clean == "end" else int(np.searchsorted(times_s, t_end, side="right"))
        else:
            idx_a = 0 if a_clean in ("0", "start") else int(a_clean)
            idx_b = num_frames if b_clean == "end" else int(b_clean)

        idx_a = max(0, min(num_frames, idx_a))
        idx_b = max(idx_a, min(num_frames, idx_b))
        labels[idx_a:idx_b] = st

    return labels_to_segments(labels, times_s)


def propose_rollout_phases_and_progress(
    ep_dir: pathlib.Path,
    num_frames: int,
    is_success_zarr: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Compute per-frame (labels, progress, times_s, info) for one rollout."""
    outcome_file = ep_dir / "outcome.json"
    outcome = {}
    if outcome_file.exists():
        try:
            outcome = json.loads(outcome_file.read_text())
        except Exception:
            pass

    verdict = outcome.get("verdict", "pass" if is_success_zarr else "fail")
    fail_mode = str(outcome.get("failure_mode", "none"))
    is_success = bool(outcome.get("success", verdict in ("pass", "success")))

    r = Rollout(ep_dir)
    if r.error or len(r.js_t) == 0:
        times_s = np.arange(num_frames, dtype=np.float32) / 30.0
        labels = np.full(num_frames, STATE_LIFT_SUCCESS if is_success else STATE_APPROACH, dtype=np.int64)
        prog = np.linspace(0.0, 1.0 if is_success else 0.0, num_frames, dtype=np.float32)
        return labels, prog, times_s, {"n_attempts": 0, "verdict": verdict, "failure_mode": fail_mode}

    t0 = float(r.js_t[0])
    t1 = float(r.js_t[-1])
    times_s = np.linspace(max(0.0, t0), max(0.1, t1), num_frames, dtype=np.float32)

    # Interpolate executed finger closure and EE z height onto the 30 Hz wrist frame timeline
    sig_30 = np.interp(times_s, r.js_t, r.exec_close)
    if len(r.tf_t) > 0 and len(r.tf_pose) > 0:
        z_30 = np.interp(times_s, r.tf_t, r.tf_pose[:, 2])
    elif len(r.cmd_t) > 0 and len(r.cmd_pose) > 0:
        z_30 = np.interp(times_s, r.cmd_t, r.cmd_pose[:, 2])
    else:
        z_30 = None

    # Pure approach failures (collision before grasp, approach_xy, approach_height, joint_limit)
    if fail_mode in ("collision", "approach_xy", "approach_height", "joint_limit") and not is_success:
        labels = np.full(num_frames, STATE_APPROACH, dtype=np.int64)
        prog = np.linspace(0.0, 0.25, num_frames, dtype=np.float32)
        return labels, prog, times_s, {"n_attempts": 0, "verdict": verdict, "failure_mode": fail_mode, "is_success": is_success}

    # Segment on 30Hz wrist timeline with 1.0s min_open_frames (30 frames), 40 deg min_span, and height z
    seg = segment_grasp_attempts(
        sig_30,
        smooth_w=7,
        closed_frac=0.45,
        min_closed_frames=12,  # ~400 ms at 30 Hz
        min_open_frames=30,    # ~1.0 s open gap required to count as a separate retry
        min_span=40.0,         # >= 40 deg total closure across the 4 index finger joints
        z=z_30,
        max_contact_height=0.035,
    )

    labels = build_phase_labels(
        num_frames=num_frames,
        attempts=seg["attempts"],
        t_lock=seg["t_lock"],
        sig=sig_30,
        z=z_30,
        is_success=is_success,
        lift_delta_z=0.015,
    )

    prog = build_progress_curve(
        num_frames=num_frames,
        attempts=seg["attempts"],
        t_first=seg["t_first"],
        t_last=seg["t_last"],
        t_lock=seg["t_lock"],
        z=z_30,
        params=get_curve_params("conservative"),
        is_success=is_success,
    )

    info = {
        "n_attempts": seg["n_attempts"],
        "verdict": verdict,
        "failure_mode": fail_mode,
        "is_success": is_success,
    }
    return labels, prog, times_s, info


def format_segments_short(segs: List[Dict]) -> str:
    parts = []
    for s in segs:
        parts.append(f"{s['state']}:{s['start_s']:.1f}-{s['end_s']:.1f}s")
    return " -> ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Inspect, propose, and refine grasp_state phase labels on rollouts.")
    parser.add_argument("--zarr", type=str, default=str(DEFAULT_ROLLOUT_ZARR))
    parser.add_argument("--runs_root", type=str, default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--sync", action="store_true", help="Write/update grasp_state and progress in Zarr and save proposed phases to outcome.json.")
    parser.add_argument("--force-auto", action="store_true", help="Overwrite even non-manual auto proposals in outcome.json.")
    parser.add_argument("--table", action="store_true", help="Print table of all rollouts and their phase segments.")
    parser.add_argument("--retries-only", action="store_true", help="In --table mode, only show rollouts with >1 grasp attempt.")
    parser.add_argument("--ep", type=int, default=None, help="Episode index (0..88) in foil_lifting_rollouts_progress.zarr to inspect or edit.")
    parser.add_argument("--set", type=str, default=None, help="Set manual phase segments for --ep, e.g. '0:0-12.5s, 3:12.5-18.0s, 0:18.0-22.0s, 1:22.0-23.5s, 2:23.5-end'")
    args = parser.parse_args()

    root = zarr.open(args.zarr, mode="a")
    group = root["data"] if "data" in root else root
    ends = root["meta/episode_ends"][:]
    sources = [str(s) for s in root["meta/episode_sources"][:]]
    succ_arr = root["meta/is_success"][:]
    total_frames = int(ends[-1])
    n_episodes = len(ends)

    all_states = (
        group["grasp_state"][:].astype(np.int64)
        if "grasp_state" in group and len(group["grasp_state"]) == total_frames
        else np.zeros(total_frames, dtype=np.int64)
    )
    all_progress = (
        group["progress"][:].astype(np.float32)
        if "progress" in group and len(group["progress"]) == total_frames
        else np.zeros(total_frames, dtype=np.float32)
    )

    runs_root = pathlib.Path(args.runs_root)
    rows = []

    for ep_idx in range(n_episodes):
        if args.ep is not None and args.set is not None and ep_idx != args.ep:
            continue

        start = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
        end = int(ends[ep_idx])
        n_f = end - start
        rel_src = sources[ep_idx]
        ep_dir = runs_root / rel_src
        outcome_file = ep_dir / "outcome.json"

        auto_labels, auto_prog, times_s, info = propose_rollout_phases_and_progress(
            ep_dir, n_f, bool(succ_arr[ep_idx])
        )

        outcome = {}
        if outcome_file.exists():
            try:
                outcome = json.loads(outcome_file.read_text())
            except Exception:
                pass

        is_manual = bool(outcome.get("phases_manual", False))

        if args.ep == ep_idx and args.set is not None:
            segs = parse_set_spec(args.set, n_f, times_s)
            labels = segments_to_labels(segs, n_f, times_s)
            outcome["phases"] = segs
            outcome["phases_manual"] = True
            is_manual = True
            outcome_file.write_text(json.dumps(outcome, indent=2) + "\n")
            print(f"[Manual Override] Saved manual phases for ep {ep_idx} ({rel_src}):")
            print(f"  {format_segments_short(segs)}")
        elif is_manual and "phases" in outcome and not args.force_auto:
            segs = outcome["phases"]
            labels = segments_to_labels(segs, n_f, times_s)
        else:
            labels = auto_labels
            segs = labels_to_segments(labels, times_s)
            if args.sync and outcome_file.exists():
                outcome["phases"] = segs
                outcome["phases_manual"] = False
                outcome_file.write_text(json.dumps(outcome, indent=2) + "\n")

        all_states[start:end] = labels
        all_progress[start:end] = auto_prog

        rows.append({
            "ep_idx": ep_idx,
            "source": rel_src,
            "frames": n_f,
            "dur_s": float(times_s[-1]),
            "verdict": info["verdict"],
            "failure_mode": info["failure_mode"],
            "n_attempts": info["n_attempts"],
            "manual": is_manual,
            "segments": segs,
        })

    if args.sync or (args.ep is not None and args.set is not None):
        if "grasp_state" in group and group["grasp_state"].shape == (total_frames,):
            group["grasp_state"][:] = all_states
        else:
            group.create_dataset(
                "grasp_state",
                data=all_states,
                shape=(total_frames,),
                chunks=(min(2048, total_frames),),
                dtype="int64",
                overwrite=True,
            )
        if "progress" in group and group["progress"].shape == (total_frames,):
            group["progress"][:] = all_progress
        else:
            group.create_dataset(
                "progress",
                data=all_progress,
                shape=(total_frames,),
                chunks=(min(2048, total_frames),),
                dtype="float32",
                overwrite=True,
            )
        print(f"\n[Zarr Sync] Updated '{args.zarr}' ({total_frames} frames across {n_episodes} rollouts).")
        for cid, cname in enumerate(GRASP_STATE_NAMES):
            cnt = int((all_states == cid).sum())
            pct = 100.0 * cnt / max(1, total_frames)
            print(f"  [{cid}] {cname:<15s}: {cnt:7d} frames ({pct:5.1f}%)")

    if args.table or (args.ep is not None and args.set is None):
        print("\n" + "=" * 125)
        print(f"{'Ep':>3} | {'Rollout Source':<55} | {'Verd':<7} | {'Fail Mode':<12} | {'Att':>3} | {'Src':<4} | {'Phase Segments (0=App, 1=Pinch, 2=Lift, 3=Miss)'}")
        print("-" * 125)
        for r in rows:
            if args.ep is not None and r["ep_idx"] != args.ep:
                continue
            if args.retries_only and r["n_attempts"] <= 1:
                continue
            tag = "MAN" if r["manual"] else "AUTO"
            att_str = f"{r['n_attempts']}*" if r["n_attempts"] > 1 else f"{r['n_attempts']}"
            seg_str = format_segments_short(r["segments"])
            print(f"{r['ep_idx']:3d} | {r['source']:<55} | {r['verdict']:<7} | {r['failure_mode']:<12} | {att_str:>3} | {tag:<4} | {seg_str}")
        print("=" * 125)


if __name__ == "__main__":
    main()
