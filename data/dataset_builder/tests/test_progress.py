# dataset_builder/tests/test_progress.py
"""Tests for progress labeling."""

import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from processing.progress import (
    compute_progress,
    segment_grasp_attempts,
    build_progress_curve,
    extract_grasp_signal,
    CONSERVATIVE,
)


def test_empty_episode():
    res = compute_progress(num_frames=0)
    assert len(res) == 0


def test_no_grasp_fallback():
    sig = np.zeros(100, dtype=np.float32)
    res = compute_progress(num_frames=100, finger_sensor=sig)
    assert len(res) == 100
    assert np.allclose(res, np.linspace(0.0, 1.0, 100))


def test_single_grasp_monotonic():
    # Approach 0..50 (open=3), grasp 50..100 (closed=50)
    sig = np.full(100, 3.0, dtype=np.float32)
    sig[50:] = 50.0
    res = compute_progress(num_frames=100, finger_sensor=sig)
    assert len(res) == 100
    assert res[0] == 0.0
    assert res[-1] == 1.0
    assert np.all(np.diff(res) >= -1e-5)  # strictly monotone


def test_retry_grasp_recovery():
    # Attempt 1: 30..45, Attempt 2: 70..100
    sig = np.full(120, 3.0, dtype=np.float32)
    sig[30:45] = 45.0
    sig[70:120] = 55.0
    seg = segment_grasp_attempts(sig)
    assert seg["n_attempts"] == 2
    assert seg["attempts"][0][0] == 30
    assert seg["attempts"][1][0] == 70

    res = compute_progress(num_frames=120, finger_sensor=sig)
    assert len(res) == 120
    assert res[0] == 0.0
    assert res[-1] == 1.0
    # Peak of attempt 1 should reach y_grasp (0.40)
    assert np.isclose(np.max(res[30:50]), CONSERVATIVE.y_grasp, atol=0.01)
    # Retreat should drop toward y_retry (0.20)
    assert np.min(res[45:70]) < 0.25
    # Peel ramp from second grasp to end reaches 1.0
    assert res[-1] == 1.0


def test_progress_variants_and_overrides():
    sig = np.full(100, 3.0, dtype=np.float32)
    sig[50:] = 50.0

    res_cons = compute_progress(num_frames=100, finger_sensor=sig, progress_dict={"variant": "conservative"})
    res_leg = compute_progress(num_frames=100, finger_sensor=sig, progress_dict={"variant": "legacy80"})

    assert np.isclose(res_cons[50], 0.32, atol=1e-4)
    assert np.isclose(res_leg[50], 0.72, atol=1e-4)


if __name__ == "__main__":
    test_empty_episode()
    test_no_grasp_fallback()
    test_single_grasp_monotonic()
    test_retry_grasp_recovery()
    test_progress_variants_and_overrides()
    print("All progress tests passed successfully!")

