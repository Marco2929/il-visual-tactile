# dataset_builder/tests/test_trim.py
"""Tests for the approach trim.

Run inside the container:
    /ros2_ws/src/il_main/venv/bin/python -m pytest \\
        dataset_builder/tests/test_trim.py -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from processing.grasp_detection import (  # noqa: E402
    GRASP_DIMS, STATE_DIM, descent_anchor, detect_episode_grasps)
from processing.trim import compute_trim, trim_metadata  # noqa: E402

OPEN, CLOSED = 5.0, 50.0


def make_episode(z, closed_spans=(), n_dims=STATE_DIM):
    """Synthetic (T, 15) state: z trajectory + pinch open/closed spans."""
    z = np.asarray(z, dtype=np.float32)
    state = np.zeros((len(z), n_dims), dtype=np.float32)
    state[:, 2] = z
    state[:, list(GRASP_DIMS)] = OPEN
    for a, b in closed_spans:
        state[a:b, list(GRASP_DIMS)] = CLOSED
    return state


def descend_hold(n_down=100, n_hold=60, z_hi=0.5, z_lo=0.0):
    """Linear descent then a flat stretch at working height."""
    return np.concatenate([
        np.linspace(z_hi, z_lo, n_down),
        np.full(n_hold, z_lo),
    ])


class TestDescentAnchor:
    def test_anchor_sits_margin_above_lowest_point(self):
        z = descend_hold()
        state = make_episode(z)
        idx = descent_anchor(state, margin_m=0.10, smooth_w=1)
        assert z[idx] - z.min() <= 0.10 + 1e-6
        # and the frame before it is still above the margin
        assert z[idx - 1] - z.min() > 0.10

    def test_margin_is_metric_not_fractional(self):
        """Doubling the z span must not move the metric cut height.

        This is the property the fraction-of-span anchor lacked.
        """
        for z_hi in (0.3, 0.6, 1.2):
            z = descend_hold(z_hi=z_hi)
            idx = descent_anchor(make_episode(z), margin_m=0.10, smooth_w=1)
            assert abs((z[idx] - z.min()) - 0.10) < 0.01

    def test_pallet_height_offset_is_irrelevant(self):
        """A different pallet height shifts z but not the cut frame."""
        z = descend_hold()
        a = descent_anchor(make_episode(z), margin_m=0.10, smooth_w=1)
        b = descent_anchor(make_episode(z + 0.25), margin_m=0.10,
                           smooth_w=1)
        assert a == b

    def test_forward_scan_ignores_later_re_descent(self):
        """A re-approach after a failed pinch must not move the anchor.

        Scanning backwards from the grasp would land in the second descent
        and cut the recovery away.
        """
        z = np.concatenate([
            np.linspace(0.5, 0.0, 80),   # first descent
            np.full(20, 0.0),            # first pinch
            np.linspace(0.0, 0.3, 20),   # lift, foil slips
            np.linspace(0.3, 0.0, 30),   # re-approach
            np.full(40, 0.0),            # second pinch
        ])
        idx = descent_anchor(make_episode(z), margin_m=0.10, smooth_w=1)
        assert idx < 80, "anchor must be in the FIRST descent"


class TestGraspDetection:
    def test_detects_single_pinch(self):
        state = make_episode(descend_hold(), closed_spans=[(110, 160)])
        info = detect_episode_grasps(state, smooth_w=1)
        assert info["has_grasp"]
        assert info["n_attempts"] == 1

    def test_short_blip_is_not_an_attempt(self):
        """3-5 frame blips before the real pinch pulled the anchor early."""
        state = make_episode(descend_hold(),
                             closed_spans=[(95, 99), (120, 160)])
        info = detect_episode_grasps(state, smooth_w=1)
        assert info["n_attempts"] == 1
        assert info["first_grasp"] >= 100

    def test_retry_is_reported_but_not_split(self):
        state = make_episode(descend_hold(n_hold=120),
                             closed_spans=[(110, 140), (160, 210)])
        info = detect_episode_grasps(state, smooth_w=1)
        assert info["n_attempts"] == 2
        # first_grasp anchors on the FIRST attempt, so the retry is kept
        assert info["first_grasp"] < 140

    def test_no_grasp(self):
        info = detect_episode_grasps(make_episode(descend_hold()),
                                     smooth_w=1)
        assert not info["has_grasp"]
        assert info["n_attempts"] == 0

    def test_wrong_state_width_is_rejected(self):
        """GRASP_DIMS would point at unrelated columns on a 27-dim state."""
        with pytest.raises(ValueError, match="pinch state"):
            detect_episode_grasps(np.zeros((50, 27), dtype=np.float32))


class TestComputeTrim:
    def test_disabled_keeps_everything(self):
        state = make_episode(descend_hold(), closed_spans=[(110, 160)])
        r = compute_trim(state, {"enabled": False})
        assert (r.start, r.end, r.skip) == (0, len(state), False)

    def test_keeps_grasp_and_everything_after(self):
        state = make_episode(descend_hold(), closed_spans=[(110, 160)])
        r = compute_trim(state, {"enabled": True, "margin_m": 0.10})
        assert not r.skip
        assert r.end == len(state)          # never cuts the tail
        assert r.start < 110                # grasp survives

    def test_retry_segment_stays_contiguous(self):
        state = make_episode(descend_hold(n_hold=120),
                             closed_spans=[(110, 140), (160, 210)])
        r = compute_trim(state, {"enabled": True, "margin_m": 0.10})
        assert not r.skip
        assert r.start < 110 and r.end == len(state)
        assert r.info["n_attempts"] == 2

    def test_skips_when_no_grasp(self):
        r = compute_trim(make_episode(descend_hold()),
                         {"enabled": True, "require_grasp": True})
        assert r.skip and r.reason == "no_grasp_detected"

    def test_skips_when_segment_too_short(self):
        # Descent that only reaches working height at the very end.
        z = np.concatenate([np.linspace(0.5, 0.0, 150), np.full(10, 0.0)])
        state = make_episode(z, closed_spans=[(150, 160)])
        r = compute_trim(state, {"enabled": True, "margin_m": 0.10,
                                 "min_frames": 50})
        assert r.skip and "segment_too_short" in r.reason

    def test_larger_margin_keeps_more(self):
        state = make_episode(descend_hold(), closed_spans=[(110, 160)])
        tight = compute_trim(state, {"enabled": True, "margin_m": 0.05})
        wide = compute_trim(state, {"enabled": True, "margin_m": 0.20})
        assert wide.start < tight.start
        assert wide.n_frames > tight.n_frames


class TestTrimMetadata:
    def test_grasp_index_is_relative_to_segment(self):
        state = make_episode(descend_hold(), closed_spans=[(110, 160)])
        r = compute_trim(state, {"enabled": True, "margin_m": 0.10})
        meta = trim_metadata(r, orig_frames=len(state))
        assert meta["orig_frames"] == len(state)
        assert meta["n_frames"] == r.n_frames
        # absolute first_grasp minus the cut offset
        assert meta["first_grasp"] == r.info["first_grasp"] - r.start
        assert 0 <= meta["first_grasp"] < meta["n_frames"]