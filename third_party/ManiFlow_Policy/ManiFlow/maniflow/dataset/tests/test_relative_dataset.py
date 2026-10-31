"""Tests for RelativeRealDexDataset against a real zarr.

    /ros2_ws/src/il_main/venv/bin/python -m pytest \\
        maniflow/dataset/tests/test_relative_dataset.py -q
"""

import os

import numpy as np
import pytest

from maniflow.dataset.relative_dataset import RelativeRealDexDataset
from maniflow.dataset.relative_pose import (
    from_relative, matrix_to_rot6d, quat_to_matrix, rot6d_to_matrix,
    to_relative,
)

ZARR = "/ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr"
N_OBS = 2
HORIZON = 64

pytestmark = pytest.mark.skipif(
    not os.path.exists(ZARR), reason=f"{ZARR} not available")


@pytest.fixture(scope="module")
def ds():
    return RelativeRealDexDataset(
        zarr_path=ZARR,
        horizon=HORIZON,
        pad_before=N_OBS - 1,
        pad_after=HORIZON - N_OBS,
        n_obs_steps=N_OBS,
        val_ratio=0.0,
        normalizer_samples=50,
    )


class TestLayout:
    def test_vector_width_is_17(self, ds):
        item = ds[0]
        assert item["obs"]["agent_pos"].shape == (HORIZON, 17)
        assert item["action"].shape == (HORIZON, 17)

    def test_base_row_is_identity(self, ds):
        ap = ds[0]["obs"]["agent_pos"].numpy()
        base = ap[N_OBS - 1]
        assert np.allclose(base[0:3], 0.0, atol=1e-6)
        assert np.allclose(base[3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)

    def test_hand_joints_stay_absolute(self, ds):
        sample = ds.sampler.sample_sequence(0)
        raw = np.asarray(sample["state"])
        enc = ds[0]["obs"]["agent_pos"].numpy()
        assert np.allclose(enc[:, 9:17], raw[:, 7:15], atol=1e-4)


class TestDecode:
    def test_action_decodes_to_absolute_poses(self, ds):
        idx = 1234
        sample = ds.sampler.sample_sequence(idx)
        state = np.asarray(sample["state"], dtype=np.float64)
        action = np.asarray(sample["action"], dtype=np.float64)

        base = N_OBS - 1
        T_base = state[base, 0:3]
        R_base = quat_to_matrix(state[base, 3:7])

        rel = ds[idx]["action"].numpy().astype(np.float64)
        T_dec, R_dec = from_relative(
            rel[:, 0:3], rot6d_to_matrix(rel[:, 3:9]), T_base, R_base)

        assert np.abs(T_dec - action[:, 0:3]).max() < 1e-5
        assert np.abs(R_dec - quat_to_matrix(action[:, 3:7])).max() < 1e-5


class TestWindowEdgeCases:
    """The sampler pads by EDGE REPLICATION, not with zeros.

    If that ever changes to zero padding, the base pose at the start of an
    episode becomes an all-zero row whose quaternion is invalid, and the
    identity invariant breaks. These tests pin the behaviour down.
    """

    def _episode_bounds(self, ds):
        ends = np.asarray(ds.replay_buffer.episode_ends[:])
        return np.concatenate([[0], ends[:-1]]), ends

    def test_episode_start_window_is_valid(self, ds):
        starts, _ = self._episode_bounds(ds)
        first = int(np.flatnonzero(
            np.asarray([t[0] for t in ds.sampler.indices]) == starts[0])[0])
        ap = ds[first]["obs"]["agent_pos"].numpy()
        assert np.isfinite(ap).all()
        assert np.allclose(ap[N_OBS - 1, 0:3], 0.0, atol=1e-6)

    def test_episode_end_window_is_valid(self, ds):
        item = ds[len(ds) - 1]
        ap = item["obs"]["agent_pos"].numpy()
        act = item["action"].numpy()
        assert np.isfinite(ap).all() and np.isfinite(act).all()
        assert np.allclose(ap[N_OBS - 1, 0:3], 0.0, atol=1e-6)

    def test_padding_replicates_edges(self, ds):
        sample = ds.sampler.sample_sequence(0)
        state = np.asarray(sample["state"])
        assert np.isfinite(state).all()
        assert not np.allclose(state[0, 3:7], 0.0), \
            "zero-padded quaternion would be invalid"

    def test_every_window_satisfies_the_invariant(self, ds):
        for i in np.linspace(0, len(ds) - 1, 300).astype(int):
            ap = ds[int(i)]["obs"]["agent_pos"].numpy()
            base = ap[N_OBS - 1]
            assert np.allclose(base[0:3], 0.0, atol=1e-6)
            assert np.allclose(base[3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)


class TestFixedBaseIsRejected:
    def test_builder_style_fixed_base_fails_the_invariant(self, ds):
        """Negative test for the failure this whole design avoids.

        Encoding against a base that is not the last observed pose - e.g. a
        per-frame base precomputed in the builder - leaves P_rel[t] != (0, I).
        """
        sample = ds.sampler.sample_sequence(500)
        state = np.asarray(sample["state"], dtype=np.float64)

        T = state[:, 0:3]
        R = quat_to_matrix(state[:, 3:7])

        wrong_base = 0
        T_rel, R_rel = to_relative(T, R, T[wrong_base], R[wrong_base])

        base = N_OBS - 1
        assert not (np.allclose(T_rel[base], 0.0, atol=1e-6)
                    and np.allclose(R_rel[base], np.eye(3), atol=1e-6))

    def test_asserts_catch_a_wrong_base(self, ds):
        from maniflow.dataset.relative_pose import assert_identity_pose

        sample = ds.sampler.sample_sequence(500)
        state = np.asarray(sample["state"], dtype=np.float64)
        T, R = state[:, 0:3], quat_to_matrix(state[:, 3:7])
        T_rel, R_rel = to_relative(T, R, T[0], R[0])

        with pytest.raises(AssertionError):
            assert_identity_pose(T_rel[N_OBS - 1], R_rel[N_OBS - 1])


class TestNormalizer:
    def test_fitted_on_relative_values(self, ds):
        norm = ds.get_normalizer()
        scale = norm.params_dict["action"]["scale"].numpy()
        assert np.isfinite(scale).all()
        assert (scale != 0).all()

    def test_normalised_output_is_in_range(self, ds):
        import torch
        norm = ds.get_normalizer()
        item = ds[0]
        out = norm.normalize({
            "agent_pos": item["obs"]["agent_pos"],
            "action": item["action"],
        })
        for v in out.values():
            assert bool(torch.isfinite(v).all())

    def test_absolute_stats_would_differ(self, ds):
        """Guards against get_normalizer() silently falling back to super()."""
        norm = ds.get_normalizer()
        rel_scale = norm.params_dict["action"]["scale"].numpy()
        raw = np.asarray(ds.replay_buffer["action"])
        abs_range = raw.max(axis=0) - raw.min(axis=0)
        abs_scale = 2.0 / np.maximum(abs_range, 1e-4)
        assert rel_scale.shape[0] == 17
        assert abs_scale.shape[0] == 15
