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
  n / p             next / previous episode
  click on plot     jump to that frame
  q                 quit

Usage (inside ros2_dev, needs a display):
    cd /ros2_ws/data/record_utils
    export DISPLAY=:1
    /ros2_ws/venv/bin/python label_viewer.py --review
    /ros2_ws/venv/bin/python label_viewer.py --only folien_unten_b_sauele/episode_041_success
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import numpy as np

# Switch to GUI backend if available
try:
    plt.switch_backend("TkAgg")
except Exception:
    pass

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
DEFAULT_ROLLOUT_ROOT = _ROLLOUT_LOGS_DIR / "foil_lifting_wrist_e40"
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


def find_rollouts(root):
    """[(key, path)] for all rollout dirs holding rollout.npz."""
    out = []
    root = pathlib.Path(root).resolve()
    if not root.exists():
        return out
    candidates = sorted(root.glob("**/rollout_*")) if not root.name.startswith("rollout_") else [root]
    for r in candidates:
        if r.is_dir() and (r / "rollout.npz").exists():
            try:
                rel = r.relative_to(root)
                key = str(rel) if len(rel.parts) > 1 else r.name
            except ValueError:
                key = r.name
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
    def __init__(self, episodes, params=CONSERVATIVE):
        self.episodes = episodes
        self.params = params
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
        key, ep = self.episodes[self.ei]
        self.key, self.ep = key, ep
        self.is_rollout = (ep / "rollout.npz").exists()
        self.rollout_frames = None

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
            if wrist is not None:
                try:
                    times, frames, _ = wrist.load_frames(ep, started_unix, max_frames=0)
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
                self.z = raw[:, 2]
        else:
            self.is_success = True
            self.verdict = "pass"
            self.fail_mode = ""
            raw = np.load(ep / "state.npy").astype(np.float32)
            self.n = len(raw)
            z = -(raw[:, 1].astype(np.float64) + raw[:, 2].astype(np.float64))
            self.z = smooth(z, 5)

        sig = extract_grasp_signal(joint_states=raw)
        self.g = smooth(sig, 5)
        self.seg = segment_grasp_attempts(sig)

        self._compute_curves()

        ev = {0, self.n - 1}
        for a, b in self.seg["attempts"]:
            ev.update((a, b))
        for t in (self.seg["t_first"], self.seg["t_last"], self.seg.get("t_lock")):
            if t is not None and t >= 0:
                ev.add(int(t))
        self.events = sorted(x for x in ev if 0 <= x < self.n)
        self.i = 0
        self.draw_static()
        self.draw_frame()

    def _compute_curves(self):
        self.curves = {}
        if self.seg["has_grasp"]:
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

        for a, b in seg["attempts"]:
            final = a >= seg["t_last"]
            self.ax_g.axvspan(a, b, color="#2ca02c" if final else "#d62728",
                              alpha=0.16, lw=0)

        t_peel = seg.get("t_lock", seg.get("t_last", -1))
        if seg["has_grasp"] and t_peel > seg["t_last"]:
            self.ax_g.axvline(t_peel, color="#2ca02c", ls="--", lw=1.0, alpha=0.7)

        self.ax_g.set_ylabel("grasp angle (deg)")
        self.ax_g.legend(loc="upper left", fontsize=8)

        # Progress curve plot
        for name, y in self.curves.items():
            self.ax_y.plot(t, y, color="#1f77b4", lw=1.5, label=name)
        self.ax_y.axhline(self.params.theta, color="#888888", ls="--", lw=0.8, label="theta=%.2f" % self.params.theta)

        if seg["has_grasp"]:
            self.ax_y.axvspan(t_peel, n, color="#2ca02c", alpha=0.06, lw=0)
            if t_peel > seg["t_last"]:
                self.ax_y.axvline(t_peel, color="#2ca02c", ls="--", lw=1.0, alpha=0.7, label="peel start (f=%d)" % t_peel)
        else:
            self.ax_y.text(0.5, 0.5, "no grasp detected - no label",
                           transform=self.ax_y.transAxes, ha="center")

        self.ax_y.set_ylim(-0.03, 1.05)
        self.ax_y.set_xlim(0, max(n - 1, 1))
        self.ax_y.set_ylabel("progress")
        self.ax_y.set_xlabel("frame")
        self.ax_y.legend(loc="upper left", fontsize=8)
        self.cursors = [ax.axvline(0, color="k", lw=1.0) for ax in (self.ax_g, self.ax_y)]

    def draw_frame(self):
        i = self.i
        if self.is_rollout and self.rollout_frames:
            self.ax_img[0].clear()
            self.ax_img[0].axis("off")
            self.ax_img[0].imshow(self.rollout_frames[i])
            self.ax_img[0].set_title(f"wrist_rgb (f={i}/{self.n-1})", fontsize=9)

            self.ax_img[1].clear()
            self.ax_img[1].axis("off")
            v_col = "#2ca02c" if self.is_success else "#d62728"
            self.ax_img[1].text(0.05, 0.78, f"Outcome: {self.verdict.upper()}", fontsize=12, fontweight="bold", color=v_col)
            if self.fail_mode:
                self.ax_img[1].text(0.05, 0.60, f"Failure Mode: {self.fail_mode}", fontsize=10, color="#d62728")
            self.ax_img[1].text(0.05, 0.44, f"EE Z: {self.z[i]:.4f} m", fontsize=10, color="#1f77b4")
            thr = self.seg.get("threshold", 0)
            self.ax_img[1].text(0.05, 0.28, f"Grasp: {self.g[i]:.2f}° (thr={thr:.2f}°)", fontsize=10)
            self.ax_img[1].text(0.05, 0.10, "[v] = Toggle Success/Fail Curve   [q] = Quit", fontsize=9, color="#555555", style="italic")
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
        k = event.key
        if k in ("right", " "):
            self.goto(self.i + 1)
        elif k == "left":
            self.goto(self.i - 1)
        elif k == "shift+right":
            self.goto(self.i + 10)
        elif k == "shift+left":
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
            self._compute_curves()
            self.draw_static()
            self.draw_frame()
        elif k in ("n", "p"):
            self.ei = (self.ei + (1 if k == "n" else -1)) % len(self.episodes)
            self.load_episode()
        elif k == "q":
            plt.close(self.fig)

    def on_click(self, event):
        if event.inaxes in (self.ax_g, self.ax_y) and event.xdata is not None:
            self.goto(round(event.xdata))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=EXTERNAL)
    ap.add_argument("--rollouts", nargs="?", const=str(DEFAULT_ROLLOUT_ROOT), default=None,
                    help="directory containing rollout_* folders (e.g. data/rollout_logs/foil_lifting_wrist_e40)")
    ap.add_argument("--prefix", nargs="*", default=["folien_"])
    ap.add_argument("--only", nargs="*", default=None,
                    help="collection/episode keys or rollout names")
    ap.add_argument("--review", action="store_true",
                    help="load the 10 reviewed episodes")
    args = ap.parse_args()

    if args.rollouts:
        episodes = find_rollouts(args.rollouts)
        if args.only:
            episodes = [(k, e) for k, e in episodes if k in set(args.only)]
    else:
        episodes = find_episodes(args.root, args.prefix)
        if args.review:
            ep_map = dict(episodes)
            episodes = [(k, ep_map[k]) for k in REVIEW_EPISODES if k in ep_map]
        elif args.only:
            episodes = [(k, e) for k, e in episodes if k in set(args.only)]

    if not episodes:
        raise SystemExit("no episodes selected")

    print("%d episodes loaded. Keys: arrows (frames), a/d (events), n/p (episodes), q (quit)"
          % len(episodes))
    viewer = Viewer(episodes)
    plt.show()
    return viewer


if __name__ == "__main__":
    main()

