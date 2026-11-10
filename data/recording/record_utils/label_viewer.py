#!/usr/bin/env python3
"""Interactive check of dataset builder progress labels against camera frames.

Visualizes:
  top     front_rgb and wrist_rgb of the current frame
  middle  grasp signal with threshold, attempts (failed = red, final = green)
  bottom  progress curve with theta, value at cursor

Keys:
  left / right      -1 / +1 frame        (shift: -10 / +10)
  a / d             previous / next event (attempt start/end, t_first, t_last, theta)
  home / end        first / last frame
  n / p             next / previous episode / rollout
  [ / ]             skip 5 episodes backward / forward
  g                 jump to next recording group / directory
  click on plot     jump to that frame
  v                 toggle outcome (pass / fail)
  1                 set current frame as t_first (grasp start)
  2                 set current frame as t_lock (solid grip / peel start)
  r                 split / add retry attempt at current frame
  R (shift+r)       reset attempts to automatic grasp detection
  0 (zero)          toggle no-grasp / total miss (progress flat 0.0)
  f                 cycle failure mode (grasp_miss, approach_xy, approach_height, slip, stall)
  s                 SAVE updated labels & progress curve to disk (outcome.json + progress.npz)
  q                 quit

Usage (inside ros2_dev, needs a display):
    cd /ros2_ws/data/record_utils
    export DISPLAY=:1
    python label_viewer.py --rollouts all
    python label_viewer.py --rollouts data/rollout_logs/foil_lifting_wrist_e40 data/rollout_logs/foil_lifting_relative_e40
    python label_viewer.py --rollouts --only rollout_007
"""

import argparse
import json
import os
import pathlib
import sys

# Auto-detect active X11 display socket if DISPLAY is unset or pointing to a nonexistent socket (e.g. :0 instead of :1)
def _ensure_display():
    cur = os.environ.get("DISPLAY", "")
    x11_dir = pathlib.Path("/tmp/.X11-unix")
    if x11_dir.exists():
        sockets = sorted(x11_dir.glob("X*"))
        if sockets:
            valid_nums = [s.name[1:] for s in sockets]
            cur_num = cur.lstrip(":").split(".")[0] if cur.startswith(":") else None
            if cur_num not in valid_nums:
                chosen = "1" if "1" in valid_nums else valid_nums[0]
                os.environ["DISPLAY"] = f":{chosen}"

_ensure_display()

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Ensure dataset_builder and rollouts are in path
_RECORD_UTILS_DIR = pathlib.Path(__file__).resolve().parent
_DATA_DIR = _RECORD_UTILS_DIR.parent.parent
_BUILDER_DIR = _DATA_DIR / "datasets" / "dataset_builder"
_ROLLOUTS_DIR = _DATA_DIR / "rollouts"
for d in (_BUILDER_DIR, _ROLLOUTS_DIR, _DATA_DIR / "recording" / "record_utils"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

try:
    import viz.wrist as wrist
except ImportError:
    wrist = None

from processing.progress import (  # noqa: E402
    compute_progress,
    segment_grasp_attempts,
    build_progress_curve,
    build_pchip_progress_curve,
    build_phase_labels,
    build_progress_from_phases,
    extract_grasp_signal,
    smooth,
    CONSERVATIVE,
    CurveParams,
    GRASP_STATE_NAMES,
    STATE_APPROACH,
    STATE_PINCH_CONTACT,
    STATE_LIFT_SUCCESS,
    STATE_MISS_OR_LOST,
)

# Disable matplotlib default keymap conflicts
for _k in list(plt.rcParams):
    if _k.startswith("keymap."):
        plt.rcParams[_k] = []

def _find_external_dir():
    candidates = [
        pathlib.Path("/mnt/data/data"),
        pathlib.Path("/mnt/data"),
        _DATA_DIR / "recording" / "episodes" / "external",
        pathlib.Path("/ros2_ws/data/recording/episodes/external"),
    ]
    for c in candidates:
        try:
            if c.exists() and any(c.iterdir()):
                return c
        except Exception:
            pass
    return candidates[0]

EXTERNAL = _find_external_dir()
_RUNS_DIR = _ROLLOUTS_DIR / "runs"
DEFAULT_ROLLOUT_ROOTS = [
    _RUNS_DIR / "foil_lifting_wrist_e40" if (_RUNS_DIR / "foil_lifting_wrist_e40").exists() else _ROLLOUTS_DIR / "foil_lifting_wrist_e40",
    _RUNS_DIR / "foil_lifting_relative_e40" if (_RUNS_DIR / "foil_lifting_relative_e40").exists() else _ROLLOUTS_DIR / "foil_lifting_relative_e40",
]
DEFAULT_ROLLOUT_ROOT = DEFAULT_ROLLOUT_ROOTS[0]
CAMERAS = ("front_rgb", "wrist_rgb")

REVIEW_EPISODES = [
    "folien_unten_b_sauele/episode_041_success",
    "folien_unten_a_sauele/episode_046_success",
    "folien_unten_a_sauele/episode_013_success",
    "folien_unten_a_sauele/episode_012_success",
    "folien_oben_b_saeule/episode_030_success",
    "folien_oben_a_saeule/episode_026_success",
    "folien_oben_a_saeule/episode_023_success",
    "folien_oben_a_saeule/episode_017_success",
    "folien_mitte_b_saeule/episode_027_success",
    "folien_mitte_a_saeule/episode_040_success",
]

PROBLEM_ROLLOUT_PATTERNS = [
    "Not_systematic_tests_findings_stated/rollout_000",
    "Not_systematic_tests_findings_stated/rollout_001",
    "Not_systematic_tests_findings_stated/rollout_002",
    "Not_systematic_tests_findings_stated/rollout_003",
    "Not_systematic_tests_findings_stated/rollout_004",
    "Not_systematic_tests_findings_stated/rollout_005",
    "Not_systematic_tests_findings_stated/rollout_006",
    "Not_systematic_tests_findings_stated/rollout_010",
    "Not_systematic_tests_findings_stated/rollout_011",
    "Not_systematic_tests_findings_stated/rollout_013",
    "Not_systematic_tests_findings_stated/rollout_014",
    "Smoothness rollouts/rollout_030",
    "Smoothness rollouts/rollout_031",
    "Smoothness rollouts/rollout_034",
    "Smoothness rollouts/rollout_035",
    "Smoothness rollouts/rollout_040",
    "Smoothness rollouts/rollout_042",
    "Smoothness rollouts/rollout_048",
    "Smoothness rollouts/rollout_051",
    "relative_e40/rollout_008",
    "relative_e40/rollout_017",
]


def find_rollouts(roots, direct_only=False):
    """[(key, path)] for all rollout dirs holding rollout.npz across one or multiple roots."""
    if isinstance(roots, (str, pathlib.Path)):
        roots = [roots]
    elif not isinstance(roots, (list, tuple)):
        roots = list(roots)

    all_roots = [pathlib.Path(r).resolve() for r in roots]
    out = []
    is_multi_root = len(all_roots) > 1

    for root in all_roots:
        if not root.exists():
            continue
        if root.name.startswith("rollout_") and (root / "rollout.npz").exists():
            out.append((root.name, root))
            continue

        if direct_only:
            candidates = sorted(p for p in root.glob("rollout_*") if p.is_dir() and (p / "rollout.npz").exists())
        else:
            # Put root direct rollouts first (systematic tests), then subdirectories
            direct = sorted(p for p in root.glob("rollout_*") if p.is_dir() and (p / "rollout.npz").exists())
            subdirs = sorted(p for p in root.glob("**/rollout_*") if p.is_dir() and (p / "rollout.npz").exists() and p.parent != root)
            candidates = direct + subdirs

        for r in candidates:
            if r.is_dir() and (r / "rollout.npz").exists():
                try:
                    rel = r.relative_to(root)
                    rel_str = str(rel)
                except ValueError:
                    rel_str = r.name
                
                key = f"{root.name}/{rel_str}" if is_multi_root else rel_str
                out.append((key, r))
    return out


def find_episodes(root, prefixes):
    """[(key, path)] for all episodes holding state.npy."""
    out = []
    root = pathlib.Path(root)
    if not root.exists():
        return out
    for coll in sorted(p for p in root.iterdir() if p.is_dir()):
        if prefixes and not any(coll.name.startswith(p) for p in prefixes):
            continue
        for ep in sorted(coll.glob("episode_*")):
            if (ep / "state.npy").exists():
                out.append(("%s/%s" % (coll.name, ep.name), ep))
    return out


def load_frame(ep, cam, i):
    if getattr(ep, "is_zarr", False):
        return None
    try:
        path = ep / cam / ("frame_%04d.png" % i)
        return mpimg.imread(path) if path.exists() else None
    except Exception:
        return None


def labels_to_segments(labels: np.ndarray, times_s: np.ndarray):
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


def segments_to_labels(segs, num_frames: int, times_s: np.ndarray) -> np.ndarray:
    """Expand phase segments back into a per-frame (num_frames,) int64 array."""
    labels = np.full(num_frames, STATE_APPROACH, dtype=np.int64)
    if not segs:
        return labels
    last_end = segs[-1].get("end_frame", -1)
    frame_aligned = (last_end == num_frames)
    times_arr = np.asarray(times_s, dtype=np.float64) if times_s is not None else None
    for s in segs:
        st = int(s["state"])
        if frame_aligned and "start_frame" in s and "end_frame" in s:
            a = max(0, min(num_frames, int(s["start_frame"])))
            b = max(a, min(num_frames, int(s["end_frame"])))
            labels[a:b] = st
        elif times_arr is not None and len(times_arr) == num_frames and "start_s" in s and "end_s" in s:
            mask = (times_arr >= float(s["start_s"]) - 1e-3) & (times_arr <= float(s["end_s"]) + 1e-3)
            labels[mask] = st
        elif "start_frame" in s and "end_frame" in s and last_end > 0:
            scale = float(num_frames) / float(last_end)
            a = max(0, min(num_frames, int(round(int(s["start_frame"]) * scale))))
            b = max(a, min(num_frames, int(round(int(s["end_frame"]) * scale))))
            labels[a:b] = st
    return labels


class ZarrEpisode:
    def __init__(self, root, grp, ep_idx, ends):
        self.root = root
        self.grp = grp
        self.ep_idx = ep_idx
        self.start = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
        self.end = int(ends[ep_idx])
        self.is_zarr = True
        self.name = f"ep_{ep_idx:03d}"

    def __truediv__(self, other):
        class FakePath:
            def exists(self):
                return False
            def __truediv__(self, other):
                return self
        return FakePath()


def find_zarr_crosscheck_episodes(z_name="foil_lifting_box_progress.zarr", n_samples=10):
    """Find retry episodes and representative clean samples from the large Zarr dataset."""
    import zarr
    cand_paths = [
        _DATA_DIR / "datasets" / "zarr_datasets" / z_name,
        pathlib.Path("/ros2_ws/data/datasets/zarr_datasets") / z_name,
    ]
    z_path = None
    for c in cand_paths:
        if c.exists():
            z_path = c
            break
    if not z_path:
        return [], []

    z_root = zarr.open(str(z_path), mode="a")
    z_grp = z_root["data"] if "data" in z_root else z_root
    ends = z_root["meta/episode_ends"][:] if "meta/episode_ends" in z_root else z_grp["meta/episode_ends"][:]
    sources = z_root["meta/episode_sources"][:] if "meta/episode_sources" in z_root else [f"ep_{i:03d}" for i in range(len(ends))]
    states = z_root["grasp_state"][:] if "grasp_state" in z_root else None
    progress = z_root["progress"][:] if "progress" in z_root else None

    retry_eps = []
    clean_candidates = []
    start = 0
    for i, end in enumerate(ends):
        src = str(sources[i])
        ep_obj = ZarrEpisode(z_root, z_grp, i, ends)
        if states is not None:
            ep_st = states[start:end]
            ep_pr = progress[start:end] if progress is not None else None
            has_miss = (ep_st == STATE_MISS_OR_LOST).any()
            pinches = np.count_nonzero((ep_st[:-1] != STATE_PINCH_CONTACT) & (ep_st[1:] == STATE_PINCH_CONTACT))
            if len(ep_st) and ep_st[0] == STATE_PINCH_CONTACT:
                pinches += 1
            pr_dips = np.count_nonzero(np.diff(ep_pr) < -0.05) if ep_pr is not None else 0
            if has_miss or pinches > 1 or pr_dips > 0:
                retry_eps.append((f"[2_Zarr_Retries] ep_{i:03d}: {src}", ep_obj))
            elif not has_miss and pinches == 1 and ep_pr is not None and ep_pr[-1] >= 0.95:
                clean_candidates.append((i, src, ep_obj))
        start = end

    sample_eps = []
    if clean_candidates:
        idxs = np.linspace(0, len(clean_candidates) - 1, min(n_samples, len(clean_candidates)), dtype=int)
        for idx in idxs:
            ep_idx, src, ep_obj = clean_candidates[idx]
            sample_eps.append((f"[3_Zarr_Samples] ep_{ep_idx:03d}: {src}", ep_obj))

    return retry_eps, sample_eps


class Viewer:
    def __init__(self, episodes, params=CONSERVATIVE, max_frames=1500, no_video=False, show_pchip=False):
        self.episodes = episodes
        self.params = params
        self.max_frames = max_frames
        self.no_video = no_video
        self.show_pchip = show_pchip
        self.ei = 0

        # Clean 3-row layout: Wrist Image (top), Classify (middle), Regression (bottom)
        self.fig = plt.figure(figsize=(13, 9.5))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[2.2, 0.9, 1.0], hspace=0.30)
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_state = self.fig.add_subplot(gs[1, 0])
        self.ax_y = self.fig.add_subplot(gs[2, 0], sharex=self.ax_state)

        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.load_episode()

    def load_episode(self):
        # Explicit garbage collection to free image frames between episodes
        self.rollout_frames = None
        import gc
        gc.collect()

        key, ep = self.episodes[self.ei]
        self.key, self.ep = key, ep
        self.is_zarr = getattr(ep, "is_zarr", False)
        self.is_rollout = (ep / "rollout.npz").exists() if not self.is_zarr else False

        if self.is_zarr:
            start, end = ep.start, ep.end
            self.n = end - start
            self.times_s = np.arange(self.n) / 15.0

            if "state" in ep.grp:
                st = ep.grp["state"][start:end]
                self.raw_data = st
                self.z = st[:, 2] if st.shape[-1] >= 3 else np.zeros(self.n)
                sig = extract_grasp_signal(hand_state=st[:, 7:15]) if st.shape[-1] >= 15 else st[:, -1]
                self.g = smooth(sig, 5)
                self.seg = segment_grasp_attempts(sig, z=self.z)
            else:
                self.raw_data = np.zeros((self.n, 1))
                self.z = np.zeros(self.n)
                self.g = np.zeros(self.n)
                self.seg = {"has_grasp": False, "attempts": [], "t_first": -1, "t_last": -1, "t_lock": -1}

            if "wrist_rgb" in ep.grp and not self.no_video:
                self.rollout_frames = ep.grp["wrist_rgb"][start:end]
            else:
                self.rollout_frames = None

            if "grasp_state" in ep.grp:
                self.phase_labels = ep.grp["grasp_state"][start:end].astype(np.int64)
            else:
                self.phase_labels = np.zeros(self.n, dtype=np.int64)

            if "progress" in ep.grp:
                self.curves = {self.params.name: ep.grp["progress"][start:end].astype(np.float32)}
            else:
                self._compute_curves()

            is_succ_arr = ep.root["meta/is_success"][:] if "meta/is_success" in ep.root else None
            self.is_success = bool(is_succ_arr[ep.ep_idx]) if is_succ_arr is not None and ep.ep_idx < len(is_succ_arr) else True
            self.verdict = "pass" if self.is_success else "fail"
            self.fail_mode = ""
            self._update_events()
            self.i = 0
            self.draw_static()
            self.draw_frame()
            return

        if self.is_rollout:
            z_data = np.load(ep / "rollout.npz")
            outcome = {}
            if (ep / "outcome.json").exists():
                try:
                    outcome = json.loads((ep / "outcome.json").read_text())
                except Exception:
                    pass
            self.is_success = outcome.get("success", outcome.get("verdict") in ("pass", "success"))
            self.verdict = outcome.get("verdict", "unknown")
            self.fail_mode = outcome.get("failure_mode", "")

            # Load wrist camera frames
            meta = {}
            if (ep / "meta.json").exists():
                try:
                    meta = json.loads((ep / "meta.json").read_text())
                except Exception:
                    pass
            started_unix = meta.get("started_unix", 0.0)

            times, frames = [], []
            if not self.no_video and wrist is not None:
                try:
                    times, frames, note = wrist.load_frames(
                        ep, started_unix, max_frames=self.max_frames, size=(640, 360)
                    )
                except Exception as e:
                    print(f"Warning loading wrist bag: {e}")

            if frames:
                self.rollout_frames = frames
                self.n = len(frames)
                js_t = z_data["js_t"]
                js_pos = z_data["js_pos"]
                resampled_js = np.zeros((self.n, js_pos.shape[1]), dtype=np.float32)
                for j in range(js_pos.shape[1]):
                    resampled_js[:, j] = np.interp(times, js_t, js_pos[:, j])
                raw = resampled_js
                if "tf_pose" in z_data and "tf_t" in z_data:
                    tf_t = z_data["tf_t"]
                    ee_z = z_data["tf_pose"][:, 2]
                    self.z = np.interp(times, tf_t, ee_z)
                else:
                    self.z = smooth(-(raw[:, 1].astype(np.float64) + raw[:, 2].astype(np.float64)), 5)
                self.times_s = times
            else:
                raw = z_data["js_pos"]
                self.n = len(raw)
                if "tf_pose" in z_data and len(z_data["tf_pose"]) == self.n:
                    self.z = z_data["tf_pose"][:, 2]
                else:
                    self.z = smooth(-(raw[:, 1].astype(np.float64) + raw[:, 2].astype(np.float64)), 5)
                self.times_s = z_data["js_t"] if "js_t" in z_data else (np.arange(self.n) / 30.0)
        else:
            self.is_success = True
            self.verdict = "pass"
            self.fail_mode = ""
            raw = np.load(ep / "state.npy").astype(np.float32)
            self.n = len(raw)
            z = -(raw[:, 1].astype(np.float64) + raw[:, 2].astype(np.float64))
            self.z = smooth(z, 5)
            self.times_s = np.arange(self.n) / 15.0

        self.raw_data = raw
        sig = extract_grasp_signal(joint_states=raw)
        self.g = smooth(sig, 5)
        self.seg = segment_grasp_attempts(sig, z=self.z)

        self.lift_peak = None
        self.segment_peaks = {}
        outcome_end = -1
        if self.is_rollout and outcome:
            self.lift_peak = outcome.get("lift_peak", None)
            raw_sp = outcome.get("segment_peaks", {})
            if isinstance(raw_sp, dict):
                self.segment_peaks = {int(k): float(v) for k, v in raw_sp.items()}
            if outcome.get("phases"):
                outcome_end = outcome["phases"][-1].get("end_frame", -1)

        if self.is_rollout and outcome and outcome_end == self.n:
            if outcome.get("attempts"):
                cl_att = []
                for a, b in outcome["attempts"]:
                    if int(a) < self.n:
                        cl_att.append((max(0, int(a)), min(int(b), self.n - 1)))
                if cl_att:
                    self.seg["attempts"] = cl_att
                    self.seg["has_grasp"] = True
                    self.seg["n_attempts"] = len(cl_att)
            if outcome.get("t_first") is not None and 0 <= outcome["t_first"] < self.n:
                self.seg["t_first"] = int(outcome["t_first"])
            if outcome.get("t_lock") is not None and 0 <= outcome["t_lock"] < self.n:
                self.seg["t_lock"] = int(outcome["t_lock"])
            if outcome.get("t_last") is not None and 0 <= outcome["t_last"] < self.n:
                self.seg["t_last"] = int(outcome["t_last"])

        self.phase_labels = np.zeros(self.n, dtype=np.int64)
        self.has_manual_phases = False
        if self.is_rollout and outcome and outcome.get("phases"):
            is_man = bool(outcome.get("phases_manual", False))
            if is_man or outcome_end == self.n:
                self.phase_labels = segments_to_labels(outcome["phases"], self.n, self.times_s)
                self.has_manual_phases = is_man

        try:
            self._compute_curves(force_phases=not self.has_manual_phases)
        except Exception as e:
            print(f"[Warning] Error computing curves for {self.key}: {e}")
            self.curves = {self.params.name: np.zeros(self.n, dtype=np.float32)}

        self._update_events()
        self.i = 0
        self.draw_static()
        self.draw_frame()

    def _update_events(self):
        ev = {0, self.n - 1}
        for a, b in self.seg.get("attempts", []):
            ev.update((a, b))
        for t in (self.seg.get("t_first"), self.seg.get("t_last"), self.seg.get("t_lock")):
            if t is not None and t >= 0:
                ev.add(int(t))
        self.events = sorted(x for x in ev if 0 <= x < self.n)

    def _get_group(self, key):
        if key.startswith("["):
            return key.split("]")[0] + "]"
        parts = key.split("/")
        if len(parts) > 2:
            return f"{parts[0]}/{parts[1]}"
        elif len(parts) == 2:
            return parts[0]
        return "root"

    def _compute_curves(self, force_phases=False):
        self.curves = {}
        if getattr(self, "has_manual_phases", False):
            # Derive continuous regression progress directly from manual classification phases
            y = build_progress_from_phases(
                self.phase_labels,
                is_success=self.is_success,
                params=self.params,
                lift_peak=getattr(self, "lift_peak", None),
                segment_peaks=getattr(self, "segment_peaks", None),
            )
            self.curves[self.params.name] = y
        else:
            if self.seg.get("has_grasp", False) and self.seg.get("attempts"):
                y = build_progress_curve(
                    num_frames=self.n,
                    attempts=self.seg["attempts"],
                    t_first=self.seg["t_first"],
                    t_last=self.seg["t_last"],
                    t_lock=self.seg["t_lock"],
                    z=self.z,
                    params=self.params,
                    is_success=self.is_success,
                )
                self.curves[self.params.name] = y
            else:
                self.curves[self.params.name] = np.zeros(self.n, dtype=np.float32)

            if force_phases or not getattr(self, "has_manual_phases", False):
                self.phase_labels = build_phase_labels(
                    num_frames=self.n,
                    attempts=self.seg.get("attempts", []),
                    t_lock=self.seg.get("t_lock"),
                    sig=self.g,
                    z=self.z,
                    is_success=self.is_success,
                )

        # Optional 4-point PCHIP spline for comparison
        if getattr(self, "show_pchip", False):
            y_pchip = build_pchip_progress_curve(
                num_frames=self.n,
                t_first=self.seg.get("t_first", -1),
                z=self.z,
                is_success=self.is_success,
            )
            self.curves["initial_pchip"] = y_pchip
        elif "initial_pchip" in self.curves:
            del self.curves["initial_pchip"]

    def draw_static(self):
        seg, n = self.seg, self.n
        t = np.arange(n)
        for ax in (self.ax_state, self.ax_y):
            ax.clear()
            ax.grid(alpha=0.25)

        # Ensure phase_labels strictly matches frame count n
        if len(self.phase_labels) != n:
            new_p = np.full(n, STATE_APPROACH, dtype=np.int64)
            copy_len = min(n, len(self.phase_labels))
            new_p[:copy_len] = self.phase_labels[:copy_len]
            self.phase_labels = new_p

        # 1. Classify Plot: Grasp State Head
        STATE_BG_COLORS = {
            0: "#e3f2fd",  # Approach (light blue)
            1: "#fff3e0",  # Pinch (light orange)
            2: "#e8f5e9",  # Lift (light green)
            3: "#ffebee",  # Miss/Lost (light red)
        }
        STATE_BORDER_COLORS = {
            0: "#1976d2",
            1: "#f57c00",
            2: "#388e3c",
            3: "#d32f2f",
        }

        for s_idx in range(4):
            self.ax_state.axhspan(s_idx - 0.45, s_idx + 0.45, color=STATE_BG_COLORS[s_idx], alpha=0.35, lw=0)

        # Step curve of the discrete state
        self.ax_state.step(t, self.phase_labels, where="post", color="#1a237e", lw=2.2)

        # Segment colored spans
        segs = labels_to_segments(self.phase_labels, self.times_s[:n])
        for s in segs:
            st = s["state"]
            self.ax_state.axvspan(s["start_frame"], s["end_frame"], color=STATE_BORDER_COLORS[st], alpha=0.18, lw=0)

        self.ax_state.set_yticks([0, 1, 2, 3])
        self.ax_state.set_yticklabels(["0: APPROACH", "1: PINCH", "2: LIFT", "3: MISS"], fontsize=8.5, fontweight="medium")
        self.ax_state.set_ylim(-0.4, 3.4)
        self.ax_state.set_ylabel("Classify\n(State 0..3)", fontsize=9, fontweight="bold")

        # 2. Regression Plot: Progress Head
        colors = {"conservative": "#1f77b4", "initial_pchip": "#ff7f0e", "legacy80": "#2ca02c"}
        for name, y in self.curves.items():
            if len(y) != n:
                new_y = np.zeros(n, dtype=np.float32)
                copy_len = min(n, len(y))
                new_y[:copy_len] = y[:copy_len]
                y = new_y
            col = colors.get(name, "#1f77b4")
            self.ax_y.plot(t, y, color=col, lw=2.0, label=name)
        self.ax_y.axhline(self.params.theta, color="#888888", ls="--", lw=0.9, label="theta=%.2f" % self.params.theta)

        t_peel = seg.get("t_lock", seg.get("t_last", -1))
        if seg.get("has_grasp", False) and seg.get("attempts") and t_peel >= 0:
            self.ax_y.axvspan(t_peel, n, color="#2ca02c", alpha=0.08, lw=0)
            if t_peel > seg.get("t_last", -1):
                self.ax_y.axvline(t_peel, color="#2ca02c", ls="--", lw=1.0, alpha=0.7)

        self.ax_y.set_ylim(-0.03, 1.05)
        self.ax_y.set_xlim(0, max(n - 1, 1))
        self.ax_y.set_ylabel("Regression\n(Progress 0..1)", fontsize=9, fontweight="bold")
        self.ax_y.set_xlabel("Frame", fontsize=9)
        self.ax_y.legend(loc="upper left", fontsize=8)

        self.cursors = [ax.axvline(0, color="k", lw=1.2) for ax in (self.ax_state, self.ax_y)]

    def draw_frame(self):
        i = self.i
        self.ax_img.clear()
        self.ax_img.axis("off")

        # Display Wrist Image
        if self.rollout_frames is not None and len(self.rollout_frames) > 0 and i < len(self.rollout_frames):
            self.ax_img.imshow(self.rollout_frames[i])
        else:
            img = load_frame(self.ep, "wrist_rgb", i)
            if img is not None:
                self.ax_img.imshow(img)
            else:
                msg = "Video disabled (--no-video)" if self.no_video else f"Wrist frame {i} not available"
                self.ax_img.text(0.5, 0.5, msg, ha="center", va="center",
                                 transform=self.ax_img.transAxes, fontsize=11, color="#666666")

        cur_phase = int(self.phase_labels[i]) if hasattr(self, "phase_labels") and i < len(self.phase_labels) else 0
        cur_prog = float(self.curves[self.params.name][i]) if self.curves else 0.0
        p_name = GRASP_STATE_NAMES[cur_phase]

        STATE_COLORS = {
            0: "#1976d2",
            1: "#e65100",
            2: "#2e7d32",
            3: "#c62828",
        }
        color = STATE_COLORS.get(cur_phase, "#333333")

        t_sec = float(self.times_s[i]) if hasattr(self, "times_s") and i < len(self.times_s) else (i / 30.0)
        v_tag = f"[{self.verdict.upper()}]" if self.verdict else ""

        title_str = (
            f"[{self.ei + 1}/{len(self.episodes)}] {self.key}   |   "
            f"Frame {i}/{self.n - 1} ({t_sec:.2f}s)   |   "
            f"Phase: {cur_phase} ({p_name})   |   "
            f"Progress: {cur_prog:.3f}   {v_tag}"
        )
        self.ax_img.set_title(title_str, fontsize=11, fontweight="bold", pad=8, color=color)

        for c in self.cursors:
            c.set_xdata([i, i])

        status_hint = getattr(
            self,
            "status_hint",
            "Keys: [1] Pinch  [2] Lift  [3] Miss/Slip  [0] Approach  [m] Cycle  [x] DelAttempt  [s] Save  [n/p] Ep  [q] Quit",
        )
        self.fig.suptitle(status_hint, fontsize=8.5, color="#555555" if "SAVED" not in status_hint else "#1976d2", y=0.015)
        self.fig.canvas.draw_idle()

    def goto(self, i):
        self.i = int(np.clip(i, 0, self.n - 1))
        self.draw_frame()

    def on_key(self, event):
        raw_k = event.key or ""
        k = raw_k.lower()
        if k in ("right", " "):
            self.goto(self.i + 1)
        elif k == "left":
            self.goto(self.i - 1)
        elif k in ("shift+right", "up"):
            self.goto(self.i + 10)
        elif k in ("shift+left", "down"):
            self.goto(self.i - 10)
        elif k == "home":
            self.goto(0)
        elif k == "end":
            self.goto(self.n - 1)
        elif k == "d":
            nxt = [x for x in self.events if x > self.i]
            if nxt:
                self.goto(nxt[0])
        elif k == "a":
            prv = [x for x in self.events if x < self.i]
            if prv:
                self.goto(prv[-1])
        elif k == "v":
            self.is_success = not self.is_success
            self.verdict = "pass" if self.is_success else "fail"
            if not self.is_success:
                if not self.fail_mode or self.fail_mode in ("none", "unknown"):
                    self.fail_mode = "slip"
            else:
                self.fail_mode = "none"
            self.has_manual_phases = True
            self.status_hint = f"Verdict: {self.verdict.upper()} (fail_mode={self.fail_mode}). Press [s] to save."
            self._compute_curves(force_phases=False)
            self.draw_static()
            self.draw_frame()
        elif k in ("0", "1", "2", "3", "4") or raw_k in ("!", "@", '"', "#", "§", "paragraph", "section", "$", ")", "=") or k in ("shift+0", "shift+1", "shift+2", "shift+3", "shift+4"):
            # Unified non-destructive phase setting:
            # Never modifies frames BEFORE cursor self.i!
            # Sets self.i .. segment_end (or whole segment if self.i == start_frame).
            key_map = {
                "0": (STATE_APPROACH, None),
                ")": (STATE_APPROACH, None),
                "=": (STATE_APPROACH, None),
                "shift+0": (STATE_APPROACH, None),
                "1": (STATE_PINCH_CONTACT, None),
                "!": (STATE_PINCH_CONTACT, None),
                "shift+1": (STATE_PINCH_CONTACT, None),
                "2": (STATE_LIFT_SUCCESS, "lift"),
                "@": (STATE_LIFT_SUCCESS, "lift"),
                '"': (STATE_LIFT_SUCCESS, "lift"),
                "shift+2": (STATE_LIFT_SUCCESS, "lift"),
                "3": (STATE_MISS_OR_LOST, "slip"),
                "#": (STATE_MISS_OR_LOST, "slip"),
                "§": (STATE_MISS_OR_LOST, "slip"),
                "paragraph": (STATE_MISS_OR_LOST, "slip"),
                "section": (STATE_MISS_OR_LOST, "slip"),
                "shift+3": (STATE_MISS_OR_LOST, "slip"),
                "4": (STATE_MISS_OR_LOST, "removed"),
                "$": (STATE_MISS_OR_LOST, "removed"),
                "shift+4": (STATE_MISS_OR_LOST, "removed"),
            }
            tgt_st, mode = key_map.get(k, key_map.get(raw_k, (STATE_APPROACH, None)))
            segs = labels_to_segments(self.phase_labels, self.times_s)

            for idx_s, s in enumerate(segs):
                if s["start_frame"] <= self.i < s["end_frame"] or (s["start_frame"] == self.i == self.n - 1):
                    end_f = s["end_frame"]
                    if not hasattr(self, "segment_peaks") or self.segment_peaks is None:
                        self.segment_peaks = {}

                    # Find which LIFT segment (if any) belongs to this attempt for per-segment peak (3 vs 4)
                    lift_start_key = None
                    if s["state"] == STATE_LIFT_SUCCESS:
                        lift_start_key = s["start_frame"]
                    elif idx_s > 0 and segs[idx_s - 1]["state"] == STATE_LIFT_SUCCESS:
                        lift_start_key = segs[idx_s - 1]["start_frame"]

                    # If cursor is ALREADY inside a MISS (3) segment and user presses [3] or [4],
                    # do NOT alter any classification phase boundaries — only switch the preceding lift's peak!
                    if s["state"] == STATE_MISS_OR_LOST and tgt_st == STATE_MISS_OR_LOST and lift_start_key is not None:
                        start_f = s["start_frame"]
                    else:
                        if tgt_st == STATE_LIFT_SUCCESS and s["state"] == STATE_PINCH_CONTACT and idx_s + 1 < len(segs):
                            if segs[idx_s + 1]["state"] == STATE_MISS_OR_LOST:
                                end_f = segs[idx_s + 1]["end_frame"]
                        start_f = self.i if self.i > s["start_frame"] else s["start_frame"]
                        self.phase_labels[start_f:end_f] = tgt_st
                        if tgt_st == STATE_LIFT_SUCCESS:
                            lift_start_key = start_f

                    if mode == "slip":
                        if lift_start_key is not None:
                            self.segment_peaks[int(lift_start_key)] = 0.55
                        if end_f >= self.n - 5:
                            self.is_success = False
                            self.verdict = "fail"
                            self.fail_mode = "slip"
                    elif mode == "removed":
                        if lift_start_key is not None:
                            self.segment_peaks[int(lift_start_key)] = 1.0
                        if end_f >= self.n - 5:
                            self.is_success = False
                            self.verdict = "fail"
                            self.fail_mode = "operator_removed"
                    elif mode == "lift":
                        if lift_start_key is not None:
                            self.segment_peaks[int(lift_start_key)] = 1.0
                        if end_f >= self.n - 5:
                            self.is_success = True
                            self.verdict = "pass"
                            self.fail_mode = "none"

                    # Keep verdict synced with final phase of episode
                    if int(self.phase_labels[-1]) == STATE_LIFT_SUCCESS:
                        self.is_success = True
                        self.verdict = "pass"
                        self.fail_mode = "none"

                    self.has_manual_phases = True
                    extra = ""
                    if mode == "slip":
                        extra = f" (Lift @{lift_start_key} peak = 0.55)"
                    elif mode == "removed":
                        extra = f" (Lift @{lift_start_key} peak = 1.00 / 100%)"
                    self.status_hint = (
                        f"Set f={start_f}..{end_f} -> {tgt_st} ({GRASP_STATE_NAMES[tgt_st]}){extra}. Press [s] to save."
                    )
                    break

            self._update_events()
            self._compute_curves(force_phases=False)
            self.draw_static()
            self.draw_frame()

        elif raw_k in ("+", "-") or k in ("+", "-", "plus", "minus"):
            if not hasattr(self, "segment_peaks") or self.segment_peaks is None:
                self.segment_peaks = {}
            segs = labels_to_segments(self.phase_labels, self.times_s)
            lift_key = None
            for idx_s, s in enumerate(segs):
                if s["start_frame"] <= self.i < s["end_frame"] or (s["start_frame"] == self.i == self.n - 1):
                    if s["state"] == STATE_LIFT_SUCCESS:
                        lift_key = int(s["start_frame"])
                    elif idx_s > 0 and segs[idx_s - 1]["state"] == STATE_LIFT_SUCCESS:
                        lift_key = int(segs[idx_s - 1]["start_frame"])
                    break
            step = 0.05 if (raw_k == "+" or k in ("+", "plus")) else -0.05
            if lift_key is not None:
                cur_pk = self.segment_peaks.get(lift_key, getattr(self, "lift_peak", None))
                if cur_pk is None:
                    cur_pk = 1.0 if self.is_success else 0.55
                new_pk = round(float(np.clip(cur_pk + step, 0.40, 1.00)), 2)
                self.segment_peaks[lift_key] = new_pk
                self.status_hint = f"Adjusted Lift @{lift_key} Peak = {new_pk:.2f}. Press [s] to save."
            else:
                cur_pk = getattr(self, "lift_peak", None)
                if cur_pk is None:
                    cur_pk = 1.0 if self.is_success else 0.55
                self.lift_peak = round(float(np.clip(cur_pk + step, 0.40, 1.00)), 2)
                self.status_hint = f"Adjusted Global Lift Peak = {self.lift_peak:.2f}. Press [s] to save."
            self.has_manual_phases = True
            self._compute_curves(force_phases=False)
            self.draw_static()
            self.draw_frame()

        elif k == "c":
            # Clear from cursor frame i to end of episode -> 0: APPROACH (for clean left-to-right relabeling)
            self.phase_labels[self.i:] = STATE_APPROACH
            self.has_manual_phases = True
            self.status_hint = f"Cleared f={self.i}..{self.n} to 0 (APPROACH). Ready to label left-to-right with [1]->[2]->[3/4]->[0]."
            self._update_events()
            self._compute_curves(force_phases=False)
            self.draw_static()
            self.draw_frame()

        elif k in ("x", "backspace", "delete"):
            # Delete/reset current segment under cursor to 0: APPROACH without losing other manual edits!
            segs = labels_to_segments(self.phase_labels, self.times_s)
            for idx_s, s in enumerate(segs):
                if s["start_frame"] <= self.i < s["end_frame"] or (s["start_frame"] == self.i == self.n - 1):
                    sf, ef = s["start_frame"], s["end_frame"]
                    # If deleting a PINCH segment followed immediately by MISS, clear both (whole failed attempt)
                    if s["state"] == STATE_PINCH_CONTACT and idx_s + 1 < len(segs) and segs[idx_s + 1]["state"] == STATE_MISS_OR_LOST:
                        ef = segs[idx_s + 1]["end_frame"]
                    elif s["state"] == STATE_MISS_OR_LOST and idx_s > 0 and segs[idx_s - 1]["state"] == STATE_PINCH_CONTACT:
                        sf = segs[idx_s - 1]["start_frame"]
                    self.phase_labels[sf:ef] = STATE_APPROACH
                    self.has_manual_phases = True
                    self.status_hint = f"Deleted segment f={sf}..{ef} -> set to 0 (APPROACH). Press [s] to save."
                    break
            self._update_events()
            self._compute_curves(force_phases=False)
            self.draw_static()
            self.draw_frame()

        elif k == "m":
            # Cycle state of current segment: 0 -> 1 -> 2 -> 3 -> 0
            segs = labels_to_segments(self.phase_labels, self.times_s)
            cycle_map = {
                STATE_APPROACH: STATE_PINCH_CONTACT,
                STATE_PINCH_CONTACT: STATE_LIFT_SUCCESS,
                STATE_LIFT_SUCCESS: STATE_MISS_OR_LOST,
                STATE_MISS_OR_LOST: STATE_APPROACH,
            }
            for s in segs:
                if s["start_frame"] <= self.i < s["end_frame"] or (s["start_frame"] == self.i == self.n - 1):
                    nxt_st = cycle_map.get(s["state"], STATE_APPROACH)
                    start_f = self.i if self.i > s["start_frame"] else s["start_frame"]
                    self.phase_labels[start_f:s["end_frame"]] = nxt_st
                    self.has_manual_phases = True
                    self.status_hint = f"f={start_f}..{s['end_frame']} -> {nxt_st} ({GRASP_STATE_NAMES[nxt_st]}). Press [s] to save."
                    break
            self._update_events()
            self._compute_curves(force_phases=False)
            self.draw_static()
            self.draw_frame()

        elif raw_k in ("R", "shift+r") or k == "r":
            sig = extract_grasp_signal(joint_states=self.raw_data)
            self.seg = segment_grasp_attempts(sig, z=self.z)
            self.has_manual_phases = False
            self.lift_peak = None
            self.status_hint = "Reset episode to automatic grasp segmentation."
            self._compute_curves(force_phases=True)
            self._update_events()
            self.draw_static()
            self.draw_frame()
        elif k == "f":
            modes = ["grasp_miss", "approach_xy", "approach_height", "slip", "stall", "none"]
            curr = self.fail_mode if self.fail_mode in modes else modes[-1]
            nxt = modes[(modes.index(curr) + 1) % len(modes)]
            self.fail_mode = nxt
            self.status_hint = f"Fail mode set to: {self.fail_mode} (press [s] to save)"
            self.draw_frame()
        elif k == "s":
            self.save_labels()
        elif k == "g":
            curr_grp = self._get_group(self.key)
            found = None
            for offset in range(1, len(self.episodes)):
                idx = (self.ei + offset) % len(self.episodes)
                other_key = self.episodes[idx][0]
                if self._get_group(other_key) != curr_grp:
                    found = idx
                    break
            if found is not None:
                self.ei = found
                nxt_key = self.episodes[self.ei][0]
                nxt_grp = self._get_group(nxt_key)
                print(f"\nJumping to group '{nxt_grp}': [{self.ei + 1}/{len(self.episodes)}] {nxt_key}...")
                self.fig.suptitle(f"Loading [{self.ei + 1}/{len(self.episodes)}] {nxt_key}...", fontsize=10, fontweight="bold")
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.load_episode()
        elif k in ("n", "p", "pageup", "pagedown", "[", "]"):
            if k in ("pageup", "["):
                delta = -5
            elif k in ("pagedown", "]"):
                delta = 5
            elif k == "n":
                delta = 1
            else:
                delta = -1
            self.status_hint = "[1]=Pinch  [2]=Lift  [3]=Miss/Slip  [0]=Approach  [m]=Cycle  [v]=Pass/Fail  [s]=Save  [n/p]=Ep  [q]=Quit"
            self.ei = (self.ei + delta) % len(self.episodes)
            nxt_key = self.episodes[self.ei][0]
            print(f"\nLoading [{self.ei + 1}/{len(self.episodes)}] {nxt_key}...")
            self.fig.suptitle(f"Loading [{self.ei + 1}/{len(self.episodes)}] {nxt_key}...", fontsize=10, fontweight="bold")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.load_episode()
        elif k == "q":
            plt.close(self.fig)

    def save_labels(self):
        """Save updated labels, outcome, and progress curve back to disk."""
        if getattr(self, "is_zarr", False):
            # Save into Zarr dataset slice
            start, end = self.ep.start, self.ep.end
            try:
                if "grasp_state" in self.ep.grp:
                    self.ep.grp["grasp_state"][start:end] = self.phase_labels
                if "progress" in self.ep.grp and self.curves and self.params.name in self.curves:
                    self.ep.grp["progress"][start:end] = self.curves[self.params.name]
                self.status_hint = f"SAVED {self.key} directly into Zarr!"
                print(f"\n>>> [SAVED] {self.key} -> grasp_state and progress updated in Zarr!")
            except Exception as e:
                self.status_hint = f"Error saving to Zarr: {e}"
                print(f"\n>>> [ERROR saving to Zarr]: {e}")
            self.draw_frame()
            return

        if self.is_rollout:
            outcome_path = self.ep / "outcome.json"
            outcome = {}
            if outcome_path.exists():
                try:
                    outcome = json.loads(outcome_path.read_text())
                except Exception:
                    pass
            outcome["verdict"] = self.verdict
            outcome["success"] = bool(self.is_success)
            outcome["failure_mode"] = self.fail_mode
            if getattr(self, "lift_peak", None) is not None:
                outcome["lift_peak"] = float(self.lift_peak)
            elif "lift_peak" in outcome:
                del outcome["lift_peak"]
            if getattr(self, "segment_peaks", None):
                outcome["segment_peaks"] = {str(k): float(v) for k, v in self.segment_peaks.items()}
            elif "segment_peaks" in outcome:
                del outcome["segment_peaks"]
            outcome["t_first"] = int(self.seg.get("t_first", -1))
            outcome["t_lock"] = int(self.seg.get("t_lock", -1))
            outcome["t_last"] = int(self.seg.get("t_last", -1))
            outcome["attempts"] = [list(a) for a in self.seg.get("attempts", [])]
            outcome["phases"] = labels_to_segments(self.phase_labels, self.times_s)
            outcome["phases_manual"] = True
            outcome_path.write_text(json.dumps(outcome, indent=2))

            # Save progress.npz
            if self.curves and self.params.name in self.curves:
                prog = self.curves[self.params.name]
                np.savez_compressed(
                    self.ep / "progress.npz",
                    t=np.arange(self.n),
                    progress=prog,
                    max_progress=float(np.max(prog)),
                    final_progress=float(prog[-1]),
                    reached_080=bool(np.max(prog) >= 0.8),
                    peak_time_s=float(self.events[-1] / 20.0) if self.events else 0.0,
                )
            self.status_hint = f"SAVED {self.key}: {self.verdict.upper()} (t_lock={outcome['t_lock']}, {len(outcome['phases'])} phases)!"
            print(f"\n>>> [SAVED] {self.key} -> verdict={self.verdict}, failure_mode={self.fail_mode}, phases saved to outcome.json!")
            self.draw_frame()
        else:
            # External dataset episode
            meta_path = self.ep / "meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    pass
            meta["success"] = bool(self.is_success)
            meta["t_first"] = int(self.seg.get("t_first", -1))
            meta["t_lock"] = int(self.seg.get("t_lock", -1))
            meta["t_last"] = int(self.seg.get("t_last", -1))
            meta["attempts"] = [list(a) for a in self.seg.get("attempts", [])]
            meta["phases"] = labels_to_segments(self.phase_labels, self.times_s)
            meta_path.write_text(json.dumps(meta, indent=2))
            if self.curves and self.params.name in self.curves:
                prog = self.curves[self.params.name]
                np.save(self.ep / "progress.npy", prog)
            self.status_hint = f"SAVED {self.key} (meta.json & progress.npy)!"
            print(f"\n>>> [SAVED] {self.key} -> meta.json & progress.npy updated!")
            self.draw_frame()

    def on_click(self, event):
        if event.inaxes in (self.ax_state, self.ax_y) and event.xdata is not None:
            self.goto(round(event.xdata))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=EXTERNAL)
    ap.add_argument(
        "--rollouts",
        nargs="*",
        default=None,
        help="one or more rollout directories, or 'all' to load both wrist_e40 and relative_e40 (default if --rollouts passed alone: all)"
    )
    ap.add_argument(
        "--zarr",
        type=str,
        default=None,
        help="path to a Zarr dataset to inspect directly (e.g. foil_lifting_box_progress.zarr)"
    )
    ap.add_argument("--direct-only", action="store_true",
                    help="only search rollouts directly in root (skip subdirectories)")
    ap.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["all", "systematic", "root", "not_systematic", "smoothness"],
        help="filter rollouts by group (systematic, not_systematic, smoothness, or all; default: all, systematic first)"
    )
    ap.add_argument("--prefix", nargs="*", default=["folien_", "foil_lifting_collection_"],
                    help="prefix filters for collection folders (default: folien_ and foil_lifting_collection_)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="collection/episode keys or rollout names")
    ap.add_argument("--review", action="store_true",
                    help="load the 10 reviewed episodes")
    ap.add_argument("--problems", action="store_true",
                    help="only load the 21 problem rollouts identified during review")
    ap.add_argument("--crosscheck", action="store_true",
                    help="load 21 problem rollouts + 30 Zarr retries + 10 Zarr samples for final crosscheck")
    ap.add_argument("--retries", action="store_true",
                    help="load only the 30 retry episodes from the large Zarr dataset")
    ap.add_argument("--max-frames", type=int, default=1500,
                    help="max wrist camera frames to decode per episode (default: 1500; 0 for uncapped)")
    ap.add_argument("--no-video", action="store_true",
                    help="skip decoding wrist camera frames (fast trajectory review)")
    ap.add_argument("--show-pchip", action="store_true",
                    help="plot optional 4-point PCHIP spline curve for comparison")
    args = ap.parse_args()

    # Check GUI backend availability
    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        print(f"\n[ERROR] Matplotlib is running in non-interactive '{backend}' backend.", file=sys.stderr)
        print(f"Current DISPLAY is set to: {os.environ.get('DISPLAY')!r}", file=sys.stderr)
        print("To fix this inside Docker, make sure your host X11 server is accessible and run:", file=sys.stderr)
        print("    export DISPLAY=:1", file=sys.stderr)
        sys.exit(1)

    if args.crosscheck:
        roots = DEFAULT_ROLLOUT_ROOTS
        all_rollouts = find_rollouts(roots)
        episodes = []
        for pat in PROBLEM_ROLLOUT_PATTERNS:
            for k, e in all_rollouts:
                if pat.lower() in k.lower():
                    episodes.append((f"[1_Problem_Rollouts] {k}", e))
                    break
        retries, samples = find_zarr_crosscheck_episodes()
        episodes.extend(retries)
        episodes.extend(samples)
        print(f"[Viewer] Loaded {len(episodes)} episodes for Crosscheck: "
              f"{len(PROBLEM_ROLLOUT_PATTERNS)} problem rollouts, {len(retries)} Zarr retries, {len(samples)} Zarr samples.")
    elif args.retries:
        retries, _ = find_zarr_crosscheck_episodes()
        episodes = retries
        print(f"[Viewer] Loaded {len(episodes)} Zarr retry episodes from large dataset.")
    elif args.problems:
        roots = DEFAULT_ROLLOUT_ROOTS
        all_rollouts = find_rollouts(roots)
        episodes = []
        for pat in PROBLEM_ROLLOUT_PATTERNS:
            for k, e in all_rollouts:
                if pat.lower() in k.lower():
                    episodes.append((k, e))
                    break
        print(f"[Viewer] Loaded {len(episodes)} problem rollouts for targeted review.")
    elif args.zarr is not None:
        import zarr
        z_path = pathlib.Path(args.zarr)
        if not z_path.exists():
            for cand in (
                _DATA_DIR / "datasets" / "zarr_datasets" / args.zarr,
                pathlib.Path("/ros2_ws/data/datasets/zarr_datasets") / args.zarr,
            ):
                if cand.exists():
                    z_path = cand
                    break
        if not z_path.exists():
            raise FileNotFoundError(f"Zarr dataset not found: {args.zarr}")
        print(f"[Viewer] Loading Zarr dataset: {z_path}")
        z_root = zarr.open(str(z_path), mode="a")
        z_grp = z_root["data"] if "data" in z_root else z_root
        ends = z_root["meta/episode_ends"][:] if "meta/episode_ends" in z_root else z_grp["meta/episode_ends"][:]
        sources = z_root["meta/episode_sources"][:] if "meta/episode_sources" in z_root else [f"ep_{i:03d}" for i in range(len(ends))]
        episodes = []
        for i in range(len(ends)):
            src = str(sources[i])
            if args.only:
                if not any(cand in src or cand == str(i) for cand in args.only):
                    continue
            episodes.append((src, ZarrEpisode(z_root, z_grp, i, ends)))
    elif args.rollouts is not None:
        raw_list = args.rollouts
        if not raw_list or any(x.lower() in ("all", "*") for x in raw_list):
            roots = DEFAULT_ROLLOUT_ROOTS
        else:
            roots = []
            for item in raw_list:
                s = str(item).strip()
                s_lower = s.lower()
                if s_lower in ("all", "*"):
                    roots.extend(DEFAULT_ROLLOUT_ROOTS)
                elif s_lower in ("wrist", "wrist_e40", "foil_lifting_wrist_e40"):
                    p = _RUNS_DIR / "foil_lifting_wrist_e40"
                    roots.append(p if p.exists() else _ROLLOUTS_DIR / "foil_lifting_wrist_e40")
                elif s_lower in ("relative", "rel", "relative_e40", "foil_lifting_relative_e40"):
                    p = _RUNS_DIR / "foil_lifting_relative_e40"
                    roots.append(p if p.exists() else _ROLLOUTS_DIR / "foil_lifting_relative_e40")
                elif pathlib.Path(s).exists():
                    roots.append(pathlib.Path(s))
                elif (_RUNS_DIR / s).exists():
                    roots.append(_RUNS_DIR / s)
                elif (_ROLLOUTS_DIR / s).exists():
                    roots.append(_ROLLOUTS_DIR / s)
                elif (_DATA_DIR / s).exists():
                    roots.append(_DATA_DIR / s)
                else:
                    search_dir = _RUNS_DIR if _RUNS_DIR.exists() else _ROLLOUTS_DIR
                    matched = [d for d in search_dir.iterdir() if d.is_dir() and s_lower in d.name.lower()]
                    if not matched and search_dir != _ROLLOUTS_DIR:
                        matched = [d for d in _ROLLOUTS_DIR.iterdir() if d.is_dir() and s_lower in d.name.lower()]
                    if matched:
                        roots.extend(matched)
                    else:
                        roots.append(pathlib.Path(s))
        episodes = find_rollouts(roots, direct_only=args.direct_only)
        if args.group and args.group.lower() != "all":
            grp = args.group.lower()
            if grp in ("systematic", "root"):
                episodes = [(k, e) for k, e in episodes if e.parent.name.startswith("foil_lifting_")]
            elif grp in ("not_systematic", "non_systematic"):
                episodes = [(k, e) for k, e in episodes if "not_systematic" in str(e).lower()]
            elif grp == "smoothness":
                episodes = [(k, e) for k, e in episodes if "smoothness" in str(e).lower()]
        if args.only:
            only_set = set(args.only)
            episodes = [
                (k, e) for k, e in episodes
                if k in only_set or e.name in only_set or any(cand in k for cand in only_set)
            ]
    else:
        episodes = find_episodes(args.root, args.prefix)
        if args.review:
            ep_map = dict(episodes)
            episodes = [(k, ep_map[k]) for k in REVIEW_EPISODES if k in ep_map]
        elif args.only:
            only_set = set(args.only)
            episodes = [
                (k, e) for k, e in episodes
                if k in only_set or e.name in only_set or any(cand in k for cand in only_set)
            ]

    if not episodes:
        raise SystemExit("no episodes selected")

    print("%d episodes loaded. Keys: arrows (frames), a/d (events), n/p (episodes), g (next group), [/] (skip 5), q (quit)"
          % len(episodes))
    viewer = Viewer(episodes, max_frames=args.max_frames, no_video=args.no_video, show_pchip=args.show_pchip)
    plt.show()
    return viewer


if __name__ == "__main__":
    main()

