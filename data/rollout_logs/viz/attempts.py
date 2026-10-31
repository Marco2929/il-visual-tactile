#!/usr/bin/env python3
"""Split a rollout into individual grasp attempts.

Lifted out of deficit_split.py, which had the only copy. Three analyses now
need it - the mid-layer diagnosis, the XY scatter (013/014/015 each contain
SEVERAL tries, so a per-rollout number would average a hit with a miss) and
the height comparison - and three copies of a threshold is three chances for
the reports to disagree without anyone noticing.

The segmentation is on the COMMANDED closure, not the executed one. The
commanded signal is what the policy decided; the executed one stalls
wherever the finger meets resistance, so thresholding it would merge an
attempt that was blocked early with one that never started.

THRESHOLD: 0.45 of the commanded range, as in deficit_split.py. It is
relative because the absolute closure level differs between runs, and it is
reported in every table so a reader can see what it resolved to.
"""
import numpy as np

from .data import GRASP_LOCAL

FRAC = 0.45

# An attempt shorter than this is a blip in the commanded signal, not a
# grasp: at 20 Hz, 5 samples is 250 ms, less than the hand needs to travel
# anywhere. Kept explicit so it shows up in the reports.
MIN_SAMPLES = 5


class Attempt:
    """One commanded closure episode, with where and when it peaked."""

    def __init__(self, rollout, index, start, end, peak):
        self.r = rollout
        self.index = index          # 1-based, as humans count tries
        self.start = start
        self.end = end
        self.peak = peak            # index into the commanded arrays

    # -- timing ------------------------------------------------------------
    @property
    def t_peak(self):
        return float(self.r.cmd_t[self.peak])

    @property
    def duration(self):
        return float(self.r.cmd_t[self.end - 1] - self.r.cmd_t[self.start])

    # -- hand --------------------------------------------------------------
    @property
    def peak_cmd_close(self):
        return float(self.r.cmd_close[self.peak])

    @property
    def peak_exec_close(self):
        """Executed closure at the commanded peak, on the commanded clock.

        `exec_on_cmd` already carries the interpolation, so this cannot drift
        from what the rest of the reports use.
        """
        return float(self.r.exec_on_cmd[self.peak])

    @property
    def deficit(self):
        return self.peak_cmd_close - self.peak_exec_close

    # -- where the grasp happened -----------------------------------------
    @property
    def xyz(self):
        """Commanded TCP position at the closure peak, in metres.

        The commanded pose is used rather than forward kinematics on the
        measured joints: it needs no URDF, and for the overfitting question
        it is the right signal anyway - we are asking where the policy AIMED.
        """
        return np.asarray(self.r.cmd_pose[self.peak, :3], dtype=float)

    @property
    def quat(self):
        return np.asarray(self.r.cmd_pose[self.peak, 3:7], dtype=float)

    @property
    def z_min(self):
        """Lowest commanded z WITHIN this attempt.

        Per attempt, not per rollout: in a run that retries, the first dip
        and the last can be centimetres apart, and the per-rollout minimum
        reports only the deepest.
        """
        return float(self.r.cmd_pose[self.start:self.end, 2].min())

    def __repr__(self):
        return (f"<Attempt {self.r.name}#{self.index} "
                f"t={self.t_peak:.1f}s peak={self.peak_cmd_close:.0f}deg>")


def threshold(cmd_close, frac=FRAC):
    c = np.asarray(cmd_close, dtype=float)
    return float(c.min() + frac * (c.max() - c.min()))


def segments(rollout, frac=FRAC, min_samples=MIN_SAMPLES):
    """-> [Attempt]. Runs of commanded closure above the relative threshold.

    Edge handling matters and is the reason this is not a one-liner: a run
    whose closure is already above the threshold at t=0, or still above it
    when the gate closes, has no rising/falling edge to find. Both are real -
    the gate is opened by hand - so the boundaries are treated as implicit
    edges rather than dropped.
    """
    c = np.asarray(rollout.cmd_close, dtype=float)
    if len(c) < min_samples or not np.isfinite(c).all():
        return []
    if c.max() - c.min() < 1e-9:
        return []

    above = c > threshold(c, frac)
    edges = np.diff(above.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(c)]

    out = []
    for a, b in zip(starts, ends):
        if b - a < min_samples:
            continue
        peak = a + int(np.argmax(c[a:b]))
        out.append(Attempt(rollout, len(out) + 1, a, b, peak))
    return out


def best(rollout, frac=FRAC):
    """The attempt with the highest commanded closure, or None."""
    segs = segments(rollout, frac)
    return max(segs, key=lambda s: s.peak_cmd_close) if segs else None


def reopenings(rollout, min_deg=5.0):
    """How far the EXECUTED closure fell back between consecutive attempts.

    This is the reactivity signal: a regrasp after losing the foil shows up
    as the hand opening again between two commanded closures. Measured on the
    executed side because that is what physically let go.
    """
    segs = segments(rollout)
    out = []
    for prev, nxt in zip(segs, segs[1:]):
        gap = rollout.exec_on_cmd[prev.peak:nxt.peak]
        if len(gap) < 2:
            continue
        drop = float(rollout.exec_on_cmd[prev.peak] - gap.min())
        if drop >= min_deg:
            out.append((prev.index, nxt.index, drop))
    return out


def summary(rollout, frac=FRAC):
    """Per-rollout attempt statistics, for the group tables."""
    segs = segments(rollout, frac)
    return {
        "name": rollout.name,
        "n_attempts": len(segs),
        "thresh": threshold(rollout.cmd_close, frac) if len(segs) else None,
        "n_reopen": len(reopenings(rollout)),
        "attempts": segs,
    }