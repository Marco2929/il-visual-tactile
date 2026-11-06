#!/usr/bin/env python3
"""Shared loading layer for logged rollouts.

The LWW reconstruction below used to be duplicated in four scripts. It is the
one piece of logic that is easy to get wrong and silently wrong when it is:
concatenating all chunk rows instead counts every step ~3x and inflates every
peak, which is exactly the bug that made an earlier script report 144.5 deg
where the truth was 141.4.

Units, stated once so no caller has to rediscover them:
  * joint_states.jsonl / js_pos  -> radians, as they appear on the bus
  * chunk actions, hand block    -> DEGREES (deploy.py converts on the way in)
  * chunk actions, pose block    -> metres + quaternion xyzw
The executed hand is converted to degrees here so both sides are comparable.
"""
import json
import os
import pathlib

import numpy as np

ARM_DOF = 6
POSE_DIM = 7
FPS = 20.0

# rj_dg_2_1..2_4 within the 20-joint hand block. Finger 2 (index) does the
# grasping, finger 1 is the opposing thumb and barely moves - measured, see
# ANALYSIS_TODO.md.
GRASP_LOCAL = [4, 5, 6, 7]

HAND_JOINT_NAMES = [f"rj_dg_{f}_{j}"
                    for f in range(1, 6) for j in range(1, 5)]

# Same env var as log_rollout.py and deploy.py, so logger and viewer cannot
# end up looking at different directories.
DEFAULT_ROOT = pathlib.Path(
    os.environ.get("ROLLOUT_LOG_DIR", "/ros2_ws/data/rollout_logs"))

# Generated HTML/rrd go here rather than next to the rollouts, so the log
# directory stays a directory of DATA and `ls` stays readable.
REPORT_DIRNAME = "reports"


def report_dir(root=DEFAULT_ROOT):
    d = pathlib.Path(root) / REPORT_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def lww_sequence(z):
    """Overlapping chunks -> the sequence the executor actually ran.

    Mirrors ActionSchedule.merge_chunk: actions are keyed by absolute step
    and a newer chunk overwrites an older one for the same step.

    Returns (seq, steps) with seq (T, 27) and steps the absolute indices, or
    (None, None) when the rollout holds no chunks.
    """
    acts = z["chunk_actions"]
    offs = z["chunk_offsets"]
    anchors = z["chunk_anchors"]
    sched = {}
    for c in range(len(offs) - 1):
        a, b = int(offs[c]), int(offs[c + 1])
        base = int(anchors[c])
        for k in range(b - a):
            sched[base + k] = acts[a + k]
    if not sched:
        return None, None
    steps = sorted(sched)
    return np.stack([sched[s] for s in steps], axis=0), np.asarray(steps)


class Rollout:
    """One logged rollout, with commanded and executed sides aligned."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.name = self.path.name
        self.error = None
        try:
            z = np.load(self.path / "rollout.npz", allow_pickle=True)
            self.meta = json.loads((self.path / "meta.json").read_text())
            self.summary = json.loads(
                (self.path / "summary.json").read_text())
        except Exception as e:
            self.error = str(e)
            return

        seq, steps = lww_sequence(z)
        if seq is None or len(z["js_pos"]) == 0:
            self.error = "no chunks or no joint states"
            return

        self.steps = steps
        self.chunk_t = z["chunk_t"] if "chunk_t" in z else np.zeros(0)
        # Commanded, on the SAME clock as everything else.
        #
        # The obvious `(steps - steps[0]) / FPS` is wrong, and wrong in a way
        # that only shows up once the wrist video is on the timeline: it
        # assumes the executor ticks at exactly FPS. It does not - measured
        # over these rollouts it ranges from 17.7 to 20.0 Hz, because the
        # tick shares a thread with inference. The error is cumulative, so by
        # the end of rollout_010 the commanded curve ran 3.2 s ahead of the
        # camera (rollout_006: 6.7 s) and the closure appeared to happen
        # while the hand was still descending.
        #
        # chunk_t[c] is the arrival time of the chunk anchored at
        # chunk_anchors[c], both logged against the gate clock that js_t,
        # pc_t and the bag timestamps use. That pairing IS the step->time
        # map, so use it instead of assuming the rate.
        self.cmd_t, self.step_hz = self._step_clock(steps, z)
        self.cmd_pose = seq[:, :POSE_DIM]
        self.cmd_hand = seq[:, POSE_DIM:]
        self.cmd_close = self.cmd_hand[:, GRASP_LOCAL].sum(axis=1)

        # The executor's output: cmd_pose after the Butterworth, the clamp
        # and the approach ramp. This is what the cartesian controller is
        # actually given, so a lag between cmd_pose and js_arm can now be
        # split into "the executor held it back" and "the controller did not
        # follow". Empty for rollouts recorded before /target_frame was
        # logged, hence the guard.
        self.tf_t = z["tf_t"] if "tf_t" in z else np.zeros(0)
        self.tf_pose = (z["tf_pose"] if "tf_pose" in z
                        else np.zeros((0, POSE_DIM)))
        self.has_target = len(self.tf_t) > 0

        # Executed.
        self.js_t = z["js_t"]
        self.js_arm = z["js_pos"][:, :ARM_DOF]
        self.exec_hand = np.degrees(z["js_pos"][:, ARM_DOF:])
        self.exec_close = self.exec_hand[:, GRASP_LOCAL].sum(axis=1)

        # Executed closure resampled onto the commanded clock. /joint_states
        # runs at ~470 Hz and the chunks at 20 Hz, so a pointwise difference
        # needs one of the two moved onto the other's grid; doing it here
        # keeps every plot that shows a tracking error consistent.
        self.exec_on_cmd = np.interp(self.cmd_t, self.js_t, self.exec_close)
        self.track_err = self.exec_on_cmd - self.cmd_close

        self.pc_t = z["pc_t"] if "pc_t" in z else np.zeros(0)
        self.has_pc = "pc" in z
        self._z = z

        # Progress prediction (from Spatio-Temporal ProgressHeadST if evaluated)
        prog_file = self.path / "progress.npz"
        if prog_file.exists():
            try:
                pz = np.load(prog_file)
                raw_t = pz["t"].astype(np.float64)
                # Convert frame indices to seconds if needed so Rerun timeline aligns
                dur_s = float(self.summary.get("duration_s", self.js_t[-1] if len(self.js_t) else 1.0))
                if len(raw_t) > 1 and raw_t[-1] > dur_s * 2.5:
                    raw_t = np.linspace(0.0, dur_s, len(raw_t))
                self.prog_t = raw_t
                self.prog_val = pz["progress"]
                self.prog_gt = pz["gt_progress"] if "gt_progress" in pz else None
                self.prog_split = str(pz["split"]) if "split" in pz else None
                self.has_progress = len(self.prog_val) > 0
            except Exception:
                self.has_progress = False
                self.prog_t = np.zeros(0)
                self.prog_val = np.zeros(0)
                self.prog_gt = None
                self.prog_split = None
        else:
            self.has_progress = False
            self.prog_t = np.zeros(0)
            self.prog_val = np.zeros(0)
            self.prog_gt = None
            self.prog_split = None

    @staticmethod
    def _step_clock(steps, z):
        """Map absolute step index -> seconds on the gate clock.

        Returns (times, hz). Least-squares through the (anchor, arrival)
        pairs rather than a per-chunk interpolation: arrival times carry
        inference jitter of a few tens of ms, and a monotone commanded clock
        matters more here than following that jitter. The fit is over the
        whole run, so a slow tick shows up as a slope, which is the thing
        that was being missed.

        Falls back to the nominal FPS when a rollout has too few chunks to
        fit (early aborts), where the drift cannot have accumulated anyway.
        """
        anchors = np.asarray(z["chunk_anchors"], dtype=float)
        ct = np.asarray(z["chunk_t"], dtype=float) if "chunk_t" in z \
            else np.zeros(0)
        steps = np.asarray(steps, dtype=float)

        if len(anchors) < 3 or len(ct) != len(anchors):
            return (steps - steps[0]) / FPS, FPS

        slope, icept = np.polyfit(anchors, ct, 1)
        # A nonsensical fit (duplicate anchors, clock jump) is worse than the
        # nominal rate, so sanity-check it against the order of magnitude we
        # know is right.
        if not (0.5 / FPS < slope < 2.0 / FPS):
            return (steps - steps[0]) / FPS, FPS
        return slope * steps + icept, 1.0 / slope

    # -- lazily loaded, because 1187 clouds x 1024 points add up over 28 --
    def point_clouds(self):
        return self._z["pc"] if self.has_pc else None

    @property
    def ok(self):
        return self.error is None

    @property
    def notes(self):
        return (self.meta.get("notes") or "").strip()

    def stats(self):
        """Scalars for cross-rollout comparison."""
        kc = int(np.argmax(self.cmd_close))
        ke = int(np.argmax(self.exec_close))
        peak_cmd = float(self.cmd_close[kc])
        peak_exec = float(self.exec_close[ke])

        post = self.exec_close[ke:]
        reopen = float(peak_exec - post.min()) if len(post) > 5 else 0.0

        return {
            "name": self.name,
            "short": self.name.replace("rollout_", ""),
            "dur": float(self.summary["duration_s"]),
            "steps": int(len(self.steps)),
            "notes": self.notes,
            "infer": self.meta.get("num_inference_steps"),
            "stop": self.summary.get("stop_reason", ""),
            "peak_cmd": peak_cmd,
            "peak_exec": peak_exec,
            "deficit": peak_cmd - peak_exec,
            "ratio": peak_exec / max(peak_cmd, 1e-6),
            "deficit_pct": 100.0 * (peak_cmd - peak_exec) / max(peak_cmd, 1e-6),
            "reopen": reopen,
            "t_cmd_close": float(self.cmd_t[kc]),
            "t_exec_close": float(self.js_t[ke]),
            "lag": float(self.js_t[ke]) - float(self.cmd_t[kc]),
            "z_at_close": float(self.cmd_pose[kc, 2]),
            "z_min": float(self.cmd_pose[:, 2].min()),
            "z_max": float(self.cmd_pose[:, 2].max()),
            "z_rise": float(self.cmd_pose[-1, 2] - self.cmd_pose[kc, 2]),
            "pc_frames": int(len(self.pc_t)),
            "max_prog": float(np.max(self.prog_val)) if self.has_progress and len(self.prog_val) > 0 else None,
            "final_prog": float(self.prog_val[-1]) if self.has_progress and len(self.prog_val) > 0 else None,
            "reached_80": bool(np.any(self.prog_val >= 0.80)) if self.has_progress and len(self.prog_val) > 0 else False,
        }


def load_all(root=DEFAULT_ROOT, ckpt=None):
    """Load every rollout, grouped by checkpoint directory.

    Returns {checkpoint_name: [Rollout, ...]} with failed loads kept in the
    list; the caller decides whether to show or drop them, rather than having
    them silently vanish.
    """
    root = pathlib.Path(root)
    out = {}
    skip = {"__pycache__", "viz", REPORT_DIRNAME}
    for ck in sorted(p for p in root.iterdir() if p.is_dir()):
        if ck.name in skip:
            continue
        if ckpt and ck.name != ckpt:
            continue
        rs = [Rollout(d) for d in sorted(ck.glob("rollout_*")) if d.is_dir()]
        if rs:
            out[ck.name] = rs
    return out


def pick_group(root, ckpt):
    """(checkpoint_name, [ok rollouts]) for the single-checkpoint commands."""
    groups = load_all(root, ckpt)
    if not groups:
        raise SystemExit(f"no rollouts under {root}")
    ck = ckpt or sorted(groups)[0]
    if ck not in groups:
        raise SystemExit(f"checkpoint {ck!r} not in {sorted(groups)}")
    return ck, [r for r in groups[ck] if r.ok]


def flag_outliers(rollouts):
    """Mark rollouts that never really grasped.

    Not a quality judgement: these are runs where the hand never closed on
    anything, so a closure deficit is not defined for them. Including them
    distorts every aggregate - the run this was first measured on moved
    corr(commanded, deficit) from +0.496 to +0.934 on its own. (That rollout,
    021, has since been deleted from the log directory; the effect is kept
    here because it is the reason the filter exists.)

    A rollout is flagged when the executed closure EXCEEDS the commanded one
    (the hand was already closed before the policy asked, i.e. nothing was
    grasped) or when the run is too short to contain an approach at all.
    """
    flagged = {}
    for r in rollouts:
        if not r.ok:
            continue
        s = r.stats()
        why = []
        if s["deficit"] < 0:
            why.append("executed exceeds commanded (no grasp)")
        if s["dur"] < 10.0:
            why.append(f"only {s['dur']:.0f}s")
        if why:
            flagged[r.name] = "; ".join(why)
    return flagged


def decimate(x, y, n=600):
    """Uniform subsample for plotting.

    Peaks are reported numerically in every label, so losing a sample here
    cannot change a stated figure - but 28 rollouts at the full 470 Hz
    /joint_states rate would be ~900k points in one HTML file.
    """
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx], y[idx]