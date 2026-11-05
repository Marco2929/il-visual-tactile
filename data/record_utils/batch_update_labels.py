#!/usr/bin/env python3
"""Batch update outcome.json and progress.npz for reviewed rollout episodes."""
import datetime
import json
import pathlib
import sys
import numpy as np
from mcap.reader import make_reader

# Ensure dataset_builder is in path
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
BUILDER_DIR = ROOT_DIR / "data" / "dataset_builder"
if str(BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(BUILDER_DIR))

from processing.progress import (  # noqa: E402
    extract_grasp_signal,
    segment_grasp_attempts,
    build_progress_curve,
    CONSERVATIVE,
)

WRIST_ROOT = ROOT_DIR / "data" / "rollout_logs" / "foil_lifting_wrist_e40"

# Mapping: relative_subpath -> (verdict, failure_mode, comment, tags)
ANNOTATIONS = {
    # Smoothness rollouts (22 episodes)
    "Smoothness rollouts/rollout_051": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_050": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_049": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_048": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_047": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_046": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_045": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_044": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_043": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_042": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_041": ("fail", "grasp_miss", "", []),
    "Smoothness rollouts/rollout_040": ("pass", "none", "Hand Accuracy and Arm speed test - reached top", []),
    "Smoothness rollouts/rollout_039": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_038": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_037": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_036": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_035": ("fail", "grasp_miss", "", []),
    "Smoothness rollouts/rollout_034": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_033": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_032": ("pass", "none", "", []),
    "Smoothness rollouts/rollout_031": ("fail", "grasp_miss", "", []),
    "Smoothness rollouts/rollout_030": ("pass", "none", "actually success but operator removed foil at top to test reactivity", ["reactivity_test"]),

    # Not systematic rollouts (25 episodes)
    "Not_systematic_tests_findings_stated/rollout_029": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_027": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_025": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_024": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_023": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_022": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_019": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_018": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_017": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_015": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_014": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_013": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_012": ("pass", "none", "", []),
    "Not_systematic_tests_findings_stated/rollout_011": ("pass", "none", "actually success but operator removed foil at top to test reactivity", ["reactivity_test"]),
    "Not_systematic_tests_findings_stated/rollout_010": ("pass", "none", "actually success but operator removed foil at top to test reactivity", ["reactivity_test"]),
    "Not_systematic_tests_findings_stated/rollout_009": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_008": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_007": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_006": ("fail", "grasp_miss", "", []),
    "Not_systematic_tests_findings_stated/rollout_005": ("pass", "none", "", []),
    "Not_systematic_tests_findings_stated/rollout_004": ("pass", "none", "actually success but operator removed foil at top to test reactivity", ["reactivity_test"]),
    "Not_systematic_tests_findings_stated/rollout_003": ("pass", "none", "actually success but operator removed foil at top to test reactivity", ["reactivity_test"]),
    "Not_systematic_tests_findings_stated/rollout_002": ("pass", "none", "", []),
    "Not_systematic_tests_findings_stated/rollout_001": ("pass", "none", "actually success but operator removed foil at top to test reactivity", ["reactivity_test"]),
    "Not_systematic_tests_findings_stated/rollout_000": ("pass", "none", "actually success but operator removed foil at top to test reactivity", ["reactivity_test"]),

    # Root systematic rollouts that needed correction
    "rollout_010": ("fail", "grasp_miss", "missed grasp, drops to 0.0", []),
}


def process_rollout(ep: pathlib.Path, verdict: str, failure_mode: str, comment: str, tags: list):
    meta = {}
    if (ep / "meta.json").exists():
        try:
            meta = json.loads((ep / "meta.json").read_text())
        except Exception:
            pass

    summary = {}
    if (ep / "summary.json").exists():
        try:
            summary = json.loads((ep / "summary.json").read_text())
        except Exception:
            pass

    started_unix = meta.get("started_unix", 0.0)
    z_data = np.load(ep / "rollout.npz")
    js_t = z_data["js_t"]
    js_pos = z_data["js_pos"]

    # Extract timestamps from wrist bag if present
    bag_dir = ep / "wrist_bag"
    mcap_files = list(bag_dir.glob("*.mcap")) if bag_dir.exists() else []

    if mcap_files:
        with open(mcap_files[0], "rb") as f:
            reader = make_reader(f)
            timestamps = [
                m.log_time
                for s, c, m in reader.iter_messages(topics=["/cam_gemini/color/image_raw"])
            ]
        if timestamps:
            times = [(ts / 1e9) - started_unix for ts in timestamps]
            n = len(times)
            resampled_js = np.zeros((n, js_pos.shape[1]), dtype=np.float32)
            for j in range(js_pos.shape[1]):
                resampled_js[:, j] = np.interp(times, js_t, js_pos[:, j])
            raw = resampled_js
            if "tf_pose" in z_data and "tf_t" in z_data:
                z = np.interp(times, z_data["tf_t"], z_data["tf_pose"][:, 2])
            else:
                z = raw[:, 2]
        else:
            raw = js_pos
            n = len(raw)
            z = raw[:, 2]
    else:
        raw = js_pos
        n = len(raw)
        z = raw[:, 2]

    # Grasp segmentation
    sig = extract_grasp_signal(joint_states=raw)
    seg = segment_grasp_attempts(sig)

    is_success = (verdict in ("pass", "success"))

    # Compute progress curve
    if seg.get("has_grasp", False) and seg.get("attempts"):
        prog = build_progress_curve(
            num_frames=n,
            attempts=seg["attempts"],
            t_first=seg["t_first"],
            t_last=seg["t_last"],
            t_lock=seg["t_lock"],
            z=z,
            params=CONSERVATIVE,
            is_success=is_success,
        )
    else:
        prog = np.zeros(n, dtype=np.float32)

    # 1. Update outcome.json
    outcome_path = ep / "outcome.json"
    outcome = {}
    if outcome_path.exists():
        try:
            outcome = json.loads(outcome_path.read_text())
        except Exception:
            pass

    outcome["rollout"] = ep.name
    try:
        outcome["rollout_index"] = int(ep.name.split("_")[-1])
    except Exception:
        pass
    if "checkpoint" in meta:
        outcome["checkpoint"] = meta["checkpoint"]
    outcome["verdict"] = verdict
    outcome["success"] = is_success
    outcome["failure_mode"] = failure_mode
    if comment:
        outcome["comment"] = comment
    elif "comment" not in outcome:
        outcome["comment"] = ""
    if tags:
        outcome["tags"] = tags
    elif "tags" not in outcome:
        outcome["tags"] = []

    if "duration_s" in summary:
        outcome["duration_s"] = summary["duration_s"]
    if "stop_reason" in summary:
        outcome["stop_reason"] = summary["stop_reason"]
    if "started_wall" in meta:
        outcome["started_wall"] = meta["started_wall"]
    if "num_inference_steps" in meta:
        outcome["num_inference_steps"] = meta["num_inference_steps"]
    if "counts" in summary and "chunks" in summary["counts"]:
        outcome["chunks"] = summary["counts"]["chunks"]

    outcome["annotator"] = "user"
    outcome["annotated_wall"] = datetime.datetime.now().isoformat()
    outcome["t_first"] = int(seg.get("t_first", -1))
    outcome["t_lock"] = int(seg.get("t_lock", -1))
    outcome["t_last"] = int(seg.get("t_last", -1))
    outcome["attempts"] = [list(a) for a in seg.get("attempts", [])]

    outcome_path.write_text(json.dumps(outcome, indent=2))

    # 2. Update progress.npz
    np.savez_compressed(
        ep / "progress.npz",
        t=np.arange(n),
        progress=prog,
        max_progress=float(np.max(prog)),
        final_progress=float(prog[-1]),
        reached_080=bool(np.max(prog) >= 0.8),
        peak_time_s=float(summary.get("duration_s", (n - 1) / 30.0)),
    )
    log_dir = ROOT_DIR / "data" / "rollout_logs"
    rel_display = ep.relative_to(log_dir) if ep.is_relative_to(log_dir) else ep.name
    print(f"[{'PASS' if is_success else 'FAIL'}] {rel_display}: n={n}, max_prog={prog.max():.2f}, final_prog={prog[-1]:.2f}")


def process_all_existing_outcomes(root_dir: pathlib.Path):
    """Scan root_dir for any rollout with outcome.json and update progress.npz."""
    count = 0
    for outcome_file in sorted(root_dir.glob("**/outcome.json")):
        ep = outcome_file.parent
        try:
            d = json.loads(outcome_file.read_text())
            verdict = d.get("verdict", "pass" if d.get("success", False) else "fail")
            failure_mode = d.get("failure_mode", "none")
            comment = d.get("comment", "")
            tags = d.get("tags", [])
            process_rollout(ep, verdict, failure_mode, comment, tags)
            count += 1
        except Exception as err:
            print(f"Error processing {ep}: {err}")
    return count


def main():
    print(f"Applying annotations to {len(ANNOTATIONS)} rollouts under {WRIST_ROOT}...")
    success_count = 0
    fail_count = 0
    missing = []

    for rel_path, (verdict, failure_mode, comment, tags) in ANNOTATIONS.items():
        ep = WRIST_ROOT / rel_path
        if not ep.exists():
            missing.append(rel_path)
            continue
        process_rollout(ep, verdict, failure_mode, comment, tags)
        if verdict in ("pass", "success"):
            success_count += 1
        else:
            fail_count += 1

    print("\nSummary:")
    print(f"  Successfully processed: {success_count + fail_count} ({success_count} pass, {fail_count} fail)")
    if missing:
        print(f"  Missing directories ({len(missing)}): {missing}")

    rel_root = ROOT_DIR / "data" / "rollout_logs" / "foil_lifting_relative_e40"
    if rel_root.exists():
        print(f"\nProcessing relative rollouts under {rel_root}...")
        rel_count = process_all_existing_outcomes(rel_root)
        print(f"Updated {rel_count} relative rollouts!")


if __name__ == "__main__":
    main()

