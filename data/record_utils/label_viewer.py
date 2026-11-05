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

# Ensure dataset_builder and rollout_logs are in path
_RECORD_UTILS_DIR = pathlib.Path(__file__).resolve().parent
_DATA_DIR = _RECORD_UTILS_DIR.parent
_BUILDER_DIR = _DATA_DIR / "dataset_builder"
_ROLLOUT_LOGS_DIR = _DATA_DIR / "rollout_logs"
for d in (_BUILDER_DIR, _ROLLOUT_LOGS_DIR):
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
    extract_grasp_signal,
    smooth,
    CONSERVATIVE,
    CurveParams,
)

# Disable matplotlib default keymap conflicts
for _k in list(plt.rcParams):
    if _k.startswith("keymap."):
        plt.rcParams[_k] = []

EXTERNAL = pathlib.Path("/ros2_ws/data/episodes/external")
DEFAULT_ROLLOUT_ROOTS = [
    _ROLLOUT_LOGS_DIR / "foil_lifting_wrist_e40",
    _ROLLOUT_LOGS_DIR / "foil_lifting_relative_e40",
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
    path = ep / cam / ("frame_%04d.png" % i)
    return mpimg.imread(path) if path.exists() else None


class Viewer:
    def __init__(self, episodes, params=CONSERVATIVE, max_frames=1500, no_video=False, show_pchip=False):
        self.episodes = episodes
        self.params = params
        self.max_frames = max_frames
        self.no_video = no_video
        self.show_pchip = show_pchip
        self.ei = 0
        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.6, 0.8, 1.0],
                                   hspace=0.25, wspace=0.05)
        self.ax_img = [self.fig.add_subplot(gs[0, k]) for k in range(2)]
        self.ax_g = self.fig.add_subplot(gs[1, :])
        self.ax_y = self.fig.add_subplot(gs[2, :], sharex=self.ax_g)
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
        self.is_rollout = (ep / "rollout.npz").exists()

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
                    self.z = raw[:, 2]
            else:
                raw = z_data["js_pos"]
                self.n = len(raw)
                if "tf_pose" in z_data and len(z_data["tf_pose"]) == self.n:
                    self.z = z_data["tf_pose"][:, 2]
                else:
                    self.z = raw[:, 2]
        else:
            self.is_success = True
            self.verdict = "pass"
            self.fail_mode = ""
            raw = np.load(ep / "state.npy").astype(np.float32)
            self.n = len(raw)
            z = -(raw[:, 1].astype(np.float64) + raw[:, 2].astype(np.float64))
            self.z = smooth(z, 5)

        self.raw_data = raw
        sig = extract_grasp_signal(joint_states=raw)
        self.g = smooth(sig, 5)
        self.seg = segment_grasp_attempts(sig)

        if self.is_rollout and outcome:
            if outcome.get("attempts"):
                self.seg["attempts"] = [tuple(a) for a in outcome["attempts"]]
                self.seg["has_grasp"] = True
                self.seg["n_attempts"] = len(self.seg["attempts"])
            if outcome.get("t_first") is not None and outcome["t_first"] >= 0:
                self.seg["t_first"] = int(outcome["t_first"])
            if outcome.get("t_lock") is not None and outcome["t_lock"] >= 0:
                self.seg["t_lock"] = int(outcome["t_lock"])
            if outcome.get("t_last") is not None and outcome["t_last"] >= 0:
                self.seg["t_last"] = int(outcome["t_last"])

        self._compute_curves()
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
        parts = key.split("/")
        if len(parts) > 2:
            return f"{parts[0]}/{parts[1]}"
        elif len(parts) == 2:
            return parts[0]
        return "root"

    def _compute_curves(self):
        self.curves = {}
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

            # Optional 4-point PCHIP spline for comparison
            if getattr(self, "show_pchip", False):
                y_pchip = build_pchip_progress_curve(
                    num_frames=self.n,
                    t_first=self.seg["t_first"],
                    z=self.z,
                    is_success=self.is_success,
                )
                self.curves["initial_pchip"] = y_pchip
            elif "initial_pchip" in self.curves:
                del self.curves["initial_pchip"]
        else:
            self.curves[self.params.name] = np.zeros(self.n, dtype=np.float32)

    def draw_static(self):
        seg, n = self.seg, self.n
        t = np.arange(n)
        for ax in (self.ax_g, self.ax_y):
            ax.clear()
            ax.grid(alpha=0.25)

        # Grasp signal plot
        self.ax_g.plot(t, self.g[:n], color="#333333", lw=1.2)
        thr = seg.get("threshold")
        if thr is not None and np.isfinite(thr):
            self.ax_g.axhline(thr, color="#ff7f0e", ls="--", lw=1.0, label="thresh=%.1f" % thr)

        for a, b in seg.get("attempts", []):
            final = a >= seg.get("t_last", 0)
            self.ax_g.axvspan(a, b, color="#2ca02c" if final else "#d62728",
                              alpha=0.16, lw=0)

        t_peel = seg.get("t_lock", seg.get("t_last", -1))
        if seg.get("has_grasp", False) and t_peel > seg.get("t_last", -1):
            self.ax_g.axvline(t_peel, color="#2ca02c", ls="--", lw=1.0, alpha=0.7)

        self.ax_g.set_ylabel("grasp angle (deg)")
        self.ax_g.legend(loc="upper left", fontsize=8)

        # Progress curve plot
        colors = {"conservative": "#1f77b4", "initial_pchip": "#ff7f0e", "legacy80": "#2ca02c"}
        styles = {"conservative": "-", "initial_pchip": "--", "legacy80": ":"}
        for name, y in self.curves.items():
            col = colors.get(name, "#1f77b4")
            ls = styles.get(name, "-")
            self.ax_y.plot(t, y, color=col, ls=ls, lw=1.5, label=name)
        self.ax_y.axhline(self.params.theta, color="#888888", ls="--", lw=0.8, label="theta=%.2f" % self.params.theta)

        if seg.get("has_grasp", False) and seg.get("attempts"):
            self.ax_y.axvspan(t_peel, n, color="#2ca02c", alpha=0.06, lw=0)
            if t_peel > seg.get("t_last", -1):
                self.ax_y.axvline(t_peel, color="#2ca02c", ls="--", lw=1.0, alpha=0.7, label="peel start (f=%d)" % t_peel)
        else:
            self.ax_y.text(0.5, 0.5, "no grasp / 0.0 progress",
                           transform=self.ax_y.transAxes, ha="center", fontsize=11, color="#888888")

        self.ax_y.set_ylim(-0.03, 1.05)
        self.ax_y.set_xlim(0, max(n - 1, 1))
        self.ax_y.set_ylabel("progress")
        self.ax_y.set_xlabel("frame")
        self.ax_y.legend(loc="upper left", fontsize=8)
        self.cursors = [ax.axvline(0, color="k", lw=1.0) for ax in (self.ax_g, self.ax_y)]

    def draw_frame(self):
        i = self.i
        if self.is_rollout:
            self.ax_img[0].clear()
            self.ax_img[0].axis("off")
            if self.rollout_frames and i < len(self.rollout_frames):
                self.ax_img[0].imshow(self.rollout_frames[i])
                self.ax_img[0].set_title(f"wrist_rgb (f={i}/{self.n-1})", fontsize=9)
            else:
                msg = "Video disabled (--no-video)" if self.no_video else "No wrist camera frames decoded"
                self.ax_img[0].text(0.5, 0.5, msg, ha="center", va="center",
                                    transform=self.ax_img[0].transAxes, fontsize=10, color="#666666")
                self.ax_img[0].set_title("wrist_rgb", fontsize=9)

            self.ax_img[1].clear()
            self.ax_img[1].axis("off")
            v_col = "#2ca02c" if self.is_success else "#d62728"
            self.ax_img[1].text(0.05, 0.78, f"Outcome: {self.verdict.upper()}", fontsize=12, fontweight="bold", color=v_col)
            if self.fail_mode:
                self.ax_img[1].text(0.05, 0.60, f"Failure Mode: {self.fail_mode}", fontsize=10, color="#d62728")
            self.ax_img[1].text(0.05, 0.44, f"EE Z: {self.z[i]:.4f} m", fontsize=10, color="#1f77b4")
            thr = self.seg.get("threshold", 0)
            self.ax_img[1].text(0.05, 0.28, f"Grasp: {self.g[i]:.2f}° (thr={thr:.2f}°)", fontsize=10)
            status_hint = getattr(self, "status_hint", "[v]=Pass/Fail  [1]=Grasp  [2]=Lock/Peel  [r]=Retry  [0]=NoGrasp  [f]=FailMode  [s]=Save  [n/p]=Ep  [g]=Group  [q]=Quit")
            self.ax_img[1].text(0.05, 0.08, status_hint, fontsize=8.2, color="#1f77b4" if "SAVED" in status_hint else "#555555", style="italic", fontweight="bold" if "SAVED" in status_hint else "normal")
            self.ax_img[1].set_title("rollout metadata & controls", fontsize=9)
        else:
            for ax, cam in zip(self.ax_img, CAMERAS):
                ax.clear()
                ax.axis("off")
                img = load_frame(self.ep, cam, i)
                if img is None:
                    ax.text(0.5, 0.5, "%s/frame_%04d.png missing" % (cam, i),
                            ha="center", transform=ax.transAxes)
                else:
                    ax.imshow(img)
                ax.set_title(cam, fontsize=9)

        for c in self.cursors:
            c.set_xdata([i, i])

        seg = self.seg
        phase = "approach"
        t_peel = seg.get("t_lock", seg.get("t_last", -1))
        for a, b in seg["attempts"]:
            if a <= i <= b:
                if a >= seg["t_last"]:
                    phase = "FINAL grip closing" if i < t_peel else ("peel" if self.is_success else "fail/empty lift")
                else:
                    phase = "failed attempt"
        if seg["has_grasp"] and i >= t_peel:
            phase = "peel" if self.is_success else "fail/empty lift"

        val_str = "progress=%.3f" % (self.curves[self.params.name][i]) if self.curves else ""
        ev = "  <EVENT>" if i in self.events else ""
        title = "[%d/%d] %s   frame %d/%d   phase: %s   %s%s" % (
            self.ei + 1, len(self.episodes), self.key, i, self.n - 1,
            phase, val_str, ev)
        self.fig.suptitle(title, fontsize=10, fontweight="bold")
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
            self.status_hint = f"Verdict toggled to: {self.verdict.upper()} (press [s] to save)"
            self._compute_curves()
            self.draw_static()
            self.draw_frame()
        elif k == "1":
            self.seg["t_first"] = self.i
            if not self.seg.get("attempts"):
                self.seg["attempts"] = [(self.i, min(self.i + 20, self.n - 1))]
                self.seg["has_grasp"] = True
            self.status_hint = f"Set t_first = frame {self.i} (press [s] to save)"
            self._compute_curves()
            self.draw_static()
            self.draw_frame()
        elif k == "2":
            self.seg["t_lock"] = self.i
            self.seg["t_last"] = self.i
            self.status_hint = f"Set t_lock/peel = frame {self.i} (press [s] to save)"
            self._compute_curves()
            self.draw_static()
            self.draw_frame()
        elif raw_k in ("R", "shift+r"):
            sig = extract_grasp_signal(joint_states=self.raw_data)
            self.seg = segment_grasp_attempts(sig)
            self.status_hint = "Reset grasp attempts to automatic segmentation."
            self._update_events()
            self._compute_curves()
            self.draw_static()
            self.draw_frame()
        elif k == "r":
            # Split current attempt or add a retry attempt at cursor frame i
            curr_attempts = list(self.seg.get("attempts", []))
            split_idx = None
            for idx, (a, b) in enumerate(curr_attempts):
                if a < self.i < b:
                    split_idx = idx
                    break

            if split_idx is not None:
                a, b = curr_attempts[split_idx]
                sub_sig = self.g[a:self.i]
                if len(sub_sig) > 20:
                    pk_idx = a + int(np.argmax(sub_sig))
                    if self.i > pk_idx + 10:
                        val_idx = pk_idx + int(np.argmin(self.g[pk_idx:self.i]))
                        end1 = max(a + 10, min((pk_idx + val_idx) // 2, self.i - 10))
                    else:
                        end1 = max(a + 10, self.i - 10)
                else:
                    end1 = max(a + 1, self.i - 1)
                new_attempts = curr_attempts[:split_idx] + [(a, end1), (self.i, b)] + curr_attempts[split_idx + 1:]
                self.seg["attempts"] = new_attempts
                self.seg["t_last"] = self.i
                thr_lock = self.seg.get("threshold", 0.5)
                hit_lock = np.flatnonzero(self.g[self.i:b] >= thr_lock)
                t_lock = self.i + int(hit_lock[0]) if len(hit_lock) else self.i
                self.seg["t_lock"] = min(t_lock, min(self.i + 25, self.n - 1))
                self.status_hint = f"Retry added at f={self.i} (total attempts: {len(new_attempts)}). Press [s] to save."
            else:
                new_attempts = sorted(curr_attempts + [(self.i, self.n - 1)])
                self.seg["attempts"] = new_attempts
                self.seg["t_last"] = new_attempts[-1][0]
                self.seg["t_lock"] = min(self.seg["t_last"] + 25, self.n - 1)
                self.status_hint = f"Added attempt starting at f={self.i}. Press [s] to save."

            self.seg["has_grasp"] = bool(self.seg["attempts"])
            self.seg["n_attempts"] = len(self.seg["attempts"])
            self.seg["t_first"] = self.seg["attempts"][0][0] if self.seg["attempts"] else -1
            self._update_events()
            self._compute_curves()
            self.draw_static()
            self.draw_frame()
        elif k == "0":
            if self.seg.get("has_grasp", False) and self.seg.get("attempts"):
                self.seg["has_grasp"] = False
                self.seg["attempts"] = []
                self.seg["t_first"] = -1
                self.seg["t_last"] = -1
                self.seg["t_lock"] = -1
                self.fail_mode = "grasp_miss"
                self.is_success = False
                self.verdict = "fail"
                self.status_hint = "Marked as NO GRASP / total miss (progress = 0.0). Press [s] to save."
            else:
                sig = extract_grasp_signal(joint_states=self.raw_data)
                self.seg = segment_grasp_attempts(sig)
                self.status_hint = "Restored automatic grasp detection."
            self._update_events()
            self._compute_curves()
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
            self.status_hint = "[v]=Pass/Fail  [1]=Grasp  [2]=Lock/Peel  [r]=Retry  [0]=NoGrasp  [f]=FailMode  [s]=Save  [n/p]=Ep  [g]=Group  [q]=Quit"
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
            outcome["t_first"] = int(self.seg.get("t_first", -1))
            outcome["t_lock"] = int(self.seg.get("t_lock", -1))
            outcome["t_last"] = int(self.seg.get("t_last", -1))
            outcome["attempts"] = [list(a) for a in self.seg.get("attempts", [])]
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
            self.status_hint = f"SAVED {self.key}: {self.verdict.upper()} (t_lock={outcome['t_lock']})!"
            print(f"\n>>> [SAVED] {self.key} -> verdict={self.verdict}, failure_mode={self.fail_mode}, progress.npz updated!")
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
            meta_path.write_text(json.dumps(meta, indent=2))
            if self.curves and self.params.name in self.curves:
                prog = self.curves[self.params.name]
                np.save(self.ep / "progress.npy", prog)
            self.status_hint = f"SAVED {self.key} (meta.json & progress.npy)!"
            print(f"\n>>> [SAVED] {self.key} -> meta.json & progress.npy updated!")
            self.draw_frame()

    def on_click(self, event):
        if event.inaxes in (self.ax_g, self.ax_y) and event.xdata is not None:
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
    ap.add_argument("--direct-only", action="store_true",
                    help="only search rollouts directly in root (skip subdirectories)")
    ap.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["all", "systematic", "root", "not_systematic", "smoothness"],
        help="filter rollouts by group (systematic, not_systematic, smoothness, or all; default: all, systematic first)"
    )
    ap.add_argument("--prefix", nargs="*", default=["folien_"])
    ap.add_argument("--only", nargs="*", default=None,
                    help="collection/episode keys or rollout names")
    ap.add_argument("--review", action="store_true",
                    help="load the 10 reviewed episodes")
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

    if args.rollouts is not None:
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
                    roots.append(_ROLLOUT_LOGS_DIR / "foil_lifting_wrist_e40")
                elif s_lower in ("relative", "rel", "relative_e40", "foil_lifting_relative_e40"):
                    roots.append(_ROLLOUT_LOGS_DIR / "foil_lifting_relative_e40")
                elif pathlib.Path(s).exists():
                    roots.append(pathlib.Path(s))
                elif (_ROLLOUT_LOGS_DIR / s).exists():
                    roots.append(_ROLLOUT_LOGS_DIR / s)
                elif (_DATA_DIR / s).exists():
                    roots.append(_DATA_DIR / s)
                else:
                    matched = [d for d in _ROLLOUT_LOGS_DIR.iterdir() if d.is_dir() and s_lower in d.name.lower()]
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

