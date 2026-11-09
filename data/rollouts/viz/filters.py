#!/usr/bin/env python3
"""Replay the executor's filter chains over a logged rollout.

The chunks in the log are what deploy.py PUBLISHED. Between that topic and
the hardware goal, control_ur_tesollo_cartesian.py applies several stateful
stages. They feed back on their own output, so a sequence of same-signed
deltas just under a clamp cannot be evaluated frame by frame - it has to be
replayed, which is what this module does, on the actually executed values
rather than on a synthetic ramp.

What it answers: of the motion the policy asked for, how much survives?

The hand and the pose side are kept in one file because they share the
Butterworth implementation, which is the part that must not drift apart
between the two.

Three structural differences make the pose side worth checking separately
rather than assuming the hand's ~1 deg loss carries over:

  a) The Butterworth runs over the QUATERNION COMPONENTS as independent
     scalars. A low-pass of q is not a rotation interpolation: the result
     leaves the unit sphere, and the renormalisation afterwards is not the
     same rotation as a proper slerp. Error grows with angular rate.
  b) The clamp and validate stages can SKIP A TICK entirely (`return`
     without publishing), not just attenuate it.
  c) The workspace box was derived from the training set. A rollout that
     reaches past it is silently frozen, with only a throttled warning.
"""
import numpy as np

CONTROL_HZ = 20.0

# --- values as they stand in control_ur_tesollo_cartesian.py ---
HAND_CUTOFF_HZ = 3.0
HAND_DEADBAND_DEG = 0.3
HAND_MAX_STEP_DEG = 6.0

POSE_CUTOFF_HZ = 8.0
MAX_POSITION_STEP = 0.1
MAX_ROTATION_STEP = 0.15

WORKSPACE_MARGIN_M = 0.25
WORKSPACE = {
    "x": (-0.523 - WORKSPACE_MARGIN_M, 0.123 + WORKSPACE_MARGIN_M),
    "y": (-1.230 - WORKSPACE_MARGIN_M, -0.535 + WORKSPACE_MARGIN_M),
    "z": (-0.080 - WORKSPACE_MARGIN_M, 0.587 + WORKSPACE_MARGIN_M),
}


def butter2_coeffs(cutoff_hz, sample_hz):
    """The same bilinear-transform 2nd-order Butterworth the executor uses.

    Deliberately reimplemented rather than taken from scipy: the point is to
    reproduce the deployed code's arithmetic, not to filter well.
    """
    nyq = 0.5 * sample_hz
    if cutoff_hz >= nyq:
        cutoff_hz = 0.99 * nyq
    wc = np.tan(np.pi * cutoff_hz / sample_hz)
    k1, k2 = np.sqrt(2.0) * wc, wc * wc
    a0 = 1.0 + k1 + k2
    b0 = k2 / a0
    a1 = (2.0 * (k2 - 1.0)) / a0
    a2 = (1.0 - k1 + k2) / a0
    return (b0, 2.0 * b0, b0), (a1, a2)


class Butter2:
    """Stateful 2nd-order section, initialised from the first sample.

    The executor seeds its history with the first value instead of zeros, so
    there is no startup transient to mistake for a tracking error.
    """

    def __init__(self, cutoff_hz, sample_hz=CONTROL_HZ):
        (self.b0, self.b1, self.b2), (self.a1, self.a2) = butter2_coeffs(
            cutoff_hz, sample_hz)
        self.x1 = self.x2 = self.y1 = self.y2 = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x1 is None:
            self.x1 = self.x2 = self.y1 = self.y2 = x.copy()
            return x.copy()
        y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
             - self.a1 * self.y1 - self.a2 * self.y2)
        self.x2, self.x1 = self.x1, x.copy()
        self.y2, self.y1 = self.y1, y.copy()
        return y


def quat_angle_deg(q1, q2):
    """Angle between two quaternions in degrees, hemisphere-corrected."""
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0))))


# ---------------------------------------------------------------- hand ----
def run_hand_chain(cmd, use_butter=True, use_deadband=True, use_clamp=True):
    """cmd: (T, J) commanded degrees -> ((T, J) filtered, stats).

    Stages, in the executor's order:
        1. ButterworthLowPass(3 Hz @ 20 Hz) per joint
        2. deadband   |delta| < 0.3 deg      -> hold previous
        3. step clamp |delta| > 6.0 deg/tick -> clip to 6.0
    """
    cmd = np.asarray(cmd, dtype=np.float64)
    T, J = cmd.shape
    butter = Butter2(HAND_CUTOFF_HZ)
    out = np.zeros_like(cmd)
    stats = {"deadbanded": 0, "clamped": 0}
    last = None

    for t in range(T):
        y = butter(cmd[t]) if use_butter else cmd[t].copy()

        if last is not None:
            for j in range(J):
                d = y[j] - last[j]
                if use_deadband and abs(d) < HAND_DEADBAND_DEG:
                    y[j] = last[j]
                    stats["deadbanded"] += 1
                elif use_clamp and abs(d) > HAND_MAX_STEP_DEG:
                    y[j] = last[j] + np.sign(d) * HAND_MAX_STEP_DEG
                    stats["clamped"] += 1

        out[t] = y
        last = y.copy()
    return out, stats


HAND_VARIANTS = [
    ("commanded (no filter)", dict(use_butter=False, use_deadband=False,
                                   use_clamp=False)),
    ("butter only", dict(use_butter=True, use_deadband=False,
                         use_clamp=False)),
    ("butter + deadband", dict(use_butter=True, use_deadband=True,
                               use_clamp=False)),
    ("full chain (as deployed)", dict(use_butter=True, use_deadband=True,
                                      use_clamp=True)),
]


# ---------------------------------------------------------------- pose ----
def run_pose_chain(cmd, use_butter=True, use_clamp=True, use_validate=True):
    """cmd: (T, 7) -> ((T, 7) filtered, stats). Mirrors _execute_step."""
    cmd = np.asarray(cmd, dtype=np.float64)
    butter = Butter2(POSE_CUTOFF_HZ)
    out = np.zeros_like(cmd)
    stats = {"pos_clamped": 0, "rot_clamped": 0, "rejected": 0,
             "held": 0, "quat_norm_min": 1.0, "quat_norm_max": 1.0}
    last = None

    for t in range(cmd.shape[0]):
        y = butter(cmd[t]) if use_butter else cmd[t].copy()

        # The filtered quaternion is off the unit sphere; record how far.
        n = float(np.linalg.norm(y[3:7]))
        stats["quat_norm_min"] = min(stats["quat_norm_min"], n)
        stats["quat_norm_max"] = max(stats["quat_norm_max"], n)

        pose = y.copy()

        if use_clamp and last is not None:
            delta = pose[:3] - last[:3]
            dist = float(np.linalg.norm(delta))
            if dist > MAX_POSITION_STEP:
                pose[:3] = last[:3] + (delta / dist) * MAX_POSITION_STEP
                stats["pos_clamped"] += 1

            q, lq = pose[3:7].copy(), last[3:7]
            if float(np.dot(q, lq)) < 0.0:
                q = -q
            nq = float(np.linalg.norm(q))
            if nq > 0:
                q = q / nq
            chord = float(np.linalg.norm(q - lq))
            if chord > MAX_ROTATION_STEP:
                q = lq + (MAX_ROTATION_STEP / chord) * (q - lq)
                nq = float(np.linalg.norm(q))
                if nq > 0:
                    q = q / nq
                stats["rot_clamped"] += 1
            pose[3:7] = q

        if use_validate:
            inbox = all(WORKSPACE[ax][0] <= pose[i] <= WORKSPACE[ax][1]
                        for i, ax in enumerate("xyz"))
            qn = float(np.linalg.norm(pose[3:7]))
            if not inbox or not (0.90 <= qn <= 1.10):
                # The executor returns without publishing: the arm holds its
                # previous target for this tick.
                stats["rejected"] += 1
                stats["held"] += 1
                out[t] = last if last is not None else pose
                continue

        out[t] = pose
        last = pose.copy()
    return out, stats


POSE_VARIANTS = [
    ("commanded (no filter)", dict(use_butter=False, use_clamp=False,
                                   use_validate=False)),
    ("butter only", dict(use_butter=True, use_clamp=False,
                         use_validate=False)),
    ("butter + clamp", dict(use_butter=True, use_clamp=True,
                            use_validate=False)),
    ("full chain (as deployed)", dict(use_butter=True, use_clamp=True,
                                      use_validate=True)),
]