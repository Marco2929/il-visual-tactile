"""Tests for the base-relative pose maths.

    /ros2_ws/src/il_main/venv/bin/python -m pytest \\
        maniflow/dataset/tests/test_relative_pose.py -q
"""

import numpy as np
import pytest

from maniflow.dataset.relative_pose import (
    assert_identity_pose, assert_valid_rotation, from_relative,
    matrix_to_quat, matrix_to_rot6d, quat_to_matrix, rot6d_to_matrix,
    to_relative,
)


def random_quats(n, seed=0):
    """Uniform random unit quaternions, xyzw."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def random_rotations(n, seed=0):
    return quat_to_matrix(random_quats(n, seed))


class TestQuaternionMatrix:
    def test_quat_to_matrix_is_valid_rotation(self):
        R = random_rotations(64, seed=1)
        assert_valid_rotation(R)

    def test_roundtrip_quat_matrix_quat(self):
        q = random_quats(64, seed=2)
        # Canonicalise the input sign too: q and -q are the same rotation,
        # and matrix_to_quat returns the w >= 0 branch.
        q = np.where(q[:, 3:4] < 0, -q, q)
        assert np.allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)

    def test_unnormalised_quaternion_still_gives_valid_rotation(self):
        """FK output is not bit-exactly normalised."""
        q = random_quats(16, seed=3) * 1.0004
        assert_valid_rotation(quat_to_matrix(q))

    def test_matrix_to_quat_handles_small_w(self):
        """180 deg rotations have w = 0, where the naive formula divides by 0."""
        for axis in np.eye(3):
            q = np.concatenate([axis, [0.0]])
            R = quat_to_matrix(q)
            assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R,
                               atol=1e-9)


class TestRot6D:
    def test_roundtrip_matrix_rot6d_matrix(self):
        R = random_rotations(64, seed=4)
        assert np.allclose(rot6d_to_matrix(matrix_to_rot6d(R)), R, atol=1e-9)

    def test_gram_schmidt_fixes_non_orthogonal_input(self):
        """Network output is never orthonormal; it must still decode."""
        rng = np.random.default_rng(5)
        r6 = rng.normal(size=(32, 6)) * 2.0
        assert_valid_rotation(rot6d_to_matrix(r6))

    def test_rot6d_uses_columns_not_rows(self):
        """Guards against a silent transpose.

        A non-symmetric rotation is needed: on a symmetric one, rows and
        columns agree and the bug would pass unnoticed.
        """
        R = quat_to_matrix(np.array([0.1, 0.2, 0.3, 0.9]))
        r6 = matrix_to_rot6d(R)
        assert np.allclose(r6[0:3], R[:, 0])
        assert np.allclose(r6[3:6], R[:, 1])

    def test_batched_shapes_are_preserved(self):
        R = random_rotations(12, seed=6).reshape(3, 4, 3, 3)
        r6 = matrix_to_rot6d(R)
        assert r6.shape == (3, 4, 6)
        assert rot6d_to_matrix(r6).shape == (3, 4, 3, 3)


class TestRelativeRoundTrip:
    def test_roundtrip_recovers_absolute_pose(self):
        rng = np.random.default_rng(7)
        T = rng.normal(size=(50, 3))
        R = random_rotations(50, seed=8)
        T_base, R_base = T[10].copy(), R[10].copy()

        T_rel, R_rel = to_relative(T, R, T_base, R_base)
        T_back, R_back = from_relative(T_rel, R_rel, T_base, R_base)

        assert np.allclose(T_back, T, atol=1e-9)
        assert np.allclose(R_back, R, atol=1e-9)

    def test_base_pose_maps_to_identity(self):
        """The defining invariant: P_rel[t] == (0, I)."""
        T = np.array([[0.3, -0.9, 0.42]])
        R = random_rotations(1, seed=9)
        T_rel, R_rel = to_relative(T, R, T[0], R[0])
        assert_identity_pose(T_rel, R_rel)

    def test_relative_rotations_are_valid(self):
        T = np.random.default_rng(10).normal(size=(20, 3))
        R = random_rotations(20, seed=11)
        _, R_rel = to_relative(T, R, T[5], R[5])
        assert_valid_rotation(R_rel)

    def test_inputs_are_not_mutated(self):
        T = np.random.default_rng(12).normal(size=(8, 3))
        R = random_rotations(8, seed=13)
        T0, R0 = T.copy(), R.copy()
        to_relative(T, R, T[0], R[0])
        from_relative(T, R, T[0], R[0])
        assert np.array_equal(T, T0) and np.array_equal(R, R0)


class TestEquivariance:
    def test_rigid_transform_of_whole_window_leaves_relative_unchanged(self):
        """The property that makes this worth doing.

        Move the entire window by an arbitrary rigid transform - a different
        pallet position, say - and the base-relative encoding is identical.
        That is why the policy cannot read the pallet position out of the
        proprioception any more and has to use the image.
        """
        rng = np.random.default_rng(14)
        T = rng.normal(size=(30, 3))
        R = random_rotations(30, seed=15)
        base = 7

        T_rel, R_rel = to_relative(T, R, T[base], R[base])

        # Arbitrary rigid motion applied to every pose in the window.
        R_shift = random_rotations(1, seed=16)[0]
        T_shift = np.array([1.3, -0.7, 0.25])
        T_moved = np.einsum("ij,nj->ni", R_shift, T) + T_shift
        R_moved = np.einsum("ij,njk->nik", R_shift, R)

        T_rel2, R_rel2 = to_relative(
            T_moved, R_moved, T_moved[base], R_moved[base])

        assert np.allclose(T_rel2, T_rel, atol=1e-9)
        assert np.allclose(R_rel2, R_rel, atol=1e-9)

    def test_absolute_encoding_is_NOT_invariant(self):
        """Control for the test above: absolute poses do change.

        Without this, the equivariance test could pass for a trivial reason.
        """
        rng = np.random.default_rng(17)
        T = rng.normal(size=(10, 3))
        T_moved = T + np.array([1.0, 0.0, 0.0])
        assert not np.allclose(T_moved, T)


class TestAsserts:
    def test_identity_assert_rejects_wrong_base(self):
        """Negative test: encoding against the WRONG pose must be caught.

        This is the 'fixed base' failure mode - using a pose other than the
        last observed one, e.g. a precomputed per-frame base from the
        builder. The invariant check is what makes it loud instead of silent.
        """
        T = np.random.default_rng(18).normal(size=(10, 3))
        R = random_rotations(10, seed=19)
        # Base taken from frame 0 but the window claims frame 5 is the base.
        T_rel, R_rel = to_relative(T, R, T[0], R[0])
        with pytest.raises(AssertionError, match="not zero|not identity"):
            assert_identity_pose(T_rel[5], R_rel[5])

    def test_valid_rotation_rejects_reflection(self):
        R = np.eye(3).copy()
        R[:, 0] *= -1.0          # det = -1
        with pytest.raises(AssertionError, match="det"):
            assert_valid_rotation(R)

    def test_valid_rotation_rejects_non_orthonormal(self):
        with pytest.raises(AssertionError, match="orthonormal"):
            assert_valid_rotation(np.eye(3) * 1.5)

    def test_valid_rotation_rejects_wrong_shape(self):
        with pytest.raises(AssertionError, match="expected"):
            assert_valid_rotation(np.zeros((3, 4)))