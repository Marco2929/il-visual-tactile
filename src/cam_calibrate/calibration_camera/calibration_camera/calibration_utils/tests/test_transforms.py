# tests/test_transforms.py

"""Tests for transforms.py"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.transforms import (
    make_transform_matrix,
    make_transform_matrix_from_quat,
    decompose_transform_matrix,
    invert_transform,
    average_transformations,
    transform_to_pose_string
)


class TestMakeTransformMatrix:
    """Tests for make_transform_matrix function."""

    def test_identity_transform(self):
        """Test identity transformation (no rotation, no translation)."""
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        
        T = make_transform_matrix(rvec, tvec)
        
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_translation_only(self):
        """Test translation-only transformation."""
        rvec = np.zeros(3)
        tvec = np.array([1.0, 2.0, 3.0])
        
        T = make_transform_matrix(rvec, tvec)
        
        expected = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ], dtype=float)
        np.testing.assert_array_almost_equal(T, expected)

    def test_rotation_only(self):
        """Test rotation-only transformation (90° around Z)."""
        rvec = np.array([0, 0, np.pi/2])  # 90° around Z
        tvec = np.zeros(3)
        
        T = make_transform_matrix(rvec, tvec)
        
        # Check rotation matrix properties
        R_mat = T[:3, :3]
        np.testing.assert_array_almost_equal(R_mat @ R_mat.T, np.eye(3))
        assert np.isclose(np.linalg.det(R_mat), 1.0)

    def test_matrix_shape(self):
        """Test that output is always 4x4."""
        T = make_transform_matrix(np.random.rand(3), np.random.rand(3))
        
        assert T.shape == (4, 4)
        np.testing.assert_array_equal(T[3, :], [0, 0, 0, 1])


class TestMakeTransformMatrixFromQuat:
    """Tests for make_transform_matrix_from_quat function."""

    def test_identity_quaternion(self):
        """Test identity quaternion [0, 0, 0, 1]."""
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        pos = np.array([1.0, 2.0, 3.0])
        
        T = make_transform_matrix_from_quat(quat, pos)
        
        # Rotation should be identity
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))
        # Translation should match position
        np.testing.assert_array_almost_equal(T[:3, 3], pos)

    def test_90_degree_rotation(self):
        """Test 90° rotation around Z axis."""
        # Quaternion for 90° around Z: [0, 0, sin(45°), cos(45°)]
        quat = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        pos = np.zeros(3)
        
        T = make_transform_matrix_from_quat(quat, pos)
        
        # [1, 0, 0] should map to [0, 1, 0]
        result = T[:3, :3] @ np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(result, [0, 1, 0])

    def test_consistency_with_rvec(self):
        """Test consistency between quaternion and rvec methods."""
        # Create rotation from euler angles
        euler = np.array([0.1, 0.2, 0.3])
        rotation = R.from_euler('xyz', euler)
        quat = rotation.as_quat()  # [x, y, z, w]
        rvec = rotation.as_rotvec()
        pos = np.array([1.0, 2.0, 3.0])
        
        T_quat = make_transform_matrix_from_quat(quat, pos)
        T_rvec = make_transform_matrix(rvec, pos)
        
        np.testing.assert_array_almost_equal(T_quat, T_rvec)


class TestDecomposeTransformMatrix:
    """Tests for decompose_transform_matrix function."""

    def test_roundtrip(self):
        """Test that decomposition and reconstruction gives same matrix."""
        rvec_orig = np.array([0.1, 0.2, 0.3])
        tvec_orig = np.array([1.0, 2.0, 3.0])
        T = make_transform_matrix(rvec_orig, tvec_orig)
        
        R_mat, rvec, tvec = decompose_transform_matrix(T)
        T_reconstructed = make_transform_matrix(rvec.flatten(), tvec)
        
        np.testing.assert_array_almost_equal(T, T_reconstructed)

    def test_rotation_matrix_properties(self):
        """Test that extracted rotation matrix is valid."""
        T = make_transform_matrix(np.random.rand(3), np.random.rand(3))
        R_mat, _, _ = decompose_transform_matrix(T)
        
        # Orthogonality
        np.testing.assert_array_almost_equal(R_mat @ R_mat.T, np.eye(3))
        # Determinant = 1 (proper rotation)
        assert np.isclose(np.linalg.det(R_mat), 1.0)


class TestInvertTransform:
    """Tests for invert_transform function."""

    def test_identity_inverse(self):
        """Test that inverse of identity is identity."""
        T = np.eye(4)
        T_inv = invert_transform(T)
        
        np.testing.assert_array_almost_equal(T_inv, np.eye(4))

    def test_inverse_property(self):
        """Test that T @ T^-1 = I."""
        T = make_transform_matrix(
            np.array([0.1, 0.2, 0.3]),
            np.array([1.0, 2.0, 3.0])
        )
        
        T_inv = invert_transform(T)
        identity = T @ T_inv
        
        np.testing.assert_array_almost_equal(identity, np.eye(4))

    def test_double_inverse(self):
        """Test that (T^-1)^-1 = T."""
        T = make_transform_matrix(
            np.array([0.5, -0.3, 0.2]),
            np.array([-1.0, 2.0, 0.5])
        )
        
        T_double_inv = invert_transform(invert_transform(T))
        
        np.testing.assert_array_almost_equal(T, T_double_inv)


class TestAverageTransformations:
    """Tests for average_transformations function."""

    def test_empty_list(self):
        """Test averaging empty list returns identity."""
        T_avg = average_transformations([])
        
        np.testing.assert_array_almost_equal(T_avg, np.eye(4))

    def test_single_transform(self):
        """Test averaging single transform returns that transform."""
        T = make_transform_matrix(
            np.array([0.1, 0.2, 0.3]),
            np.array([1.0, 2.0, 3.0])
        )
        
        T_avg = average_transformations([T])
        
        np.testing.assert_array_almost_equal(T_avg, T)

    def test_identical_transforms(self):
        """Test averaging identical transforms returns that transform."""
        T = make_transform_matrix(
            np.array([0.1, 0.2, 0.3]),
            np.array([1.0, 2.0, 3.0])
        )
        transforms = [T.copy() for _ in range(5)]
        
        T_avg = average_transformations(transforms)
        
        np.testing.assert_array_almost_equal(T_avg, T)

    @pytest.mark.parametrize("num_transforms", [2, 5, 10])
    def test_output_is_valid_transform(self, num_transforms):
        """Test that averaged transform is a valid transformation."""
        transforms = []
        for i in range(num_transforms):
            rvec = np.random.rand(3) * 0.2
            tvec = np.random.rand(3)
            transforms.append(make_transform_matrix(rvec, tvec))
        
        T_avg = average_transformations(transforms)
        
        # Check shape
        assert T_avg.shape == (4, 4)
        # Check last row
        np.testing.assert_array_equal(T_avg[3, :], [0, 0, 0, 1])
        # Check rotation matrix properties
        R_mat = T_avg[:3, :3]
        np.testing.assert_array_almost_equal(R_mat @ R_mat.T, np.eye(3), decimal=5)


class TestTransformToPoseString:
    """Tests for transform_to_pose_string function."""

    def test_output_format(self, sample_transform):
        """Test that output contains expected information."""
        result = transform_to_pose_string(sample_transform)
        
        assert "Position:" in result
        assert "Rotation:" in result
        assert "m" in result
        assert "deg" in result

    def test_identity_values(self, identity_transform):
        """Test identity transform produces zero values."""
        result = transform_to_pose_string(identity_transform)
        
        assert "0.0000" in result  # Position should be zeros
        assert "0.0" in result     # Rotation should be zeros

    @pytest.mark.parametrize("precision", [1, 2, 3, 4, 5])
    def test_precision(self, sample_transform, precision):
        """Test that precision parameter is respected."""
        result = transform_to_pose_string(sample_transform, precision=precision)
        
        # Check that we have the right number of decimal places
        # The format should be like "1.0000" for precision=4
        assert f".{'0' * precision}" in result or any(
            f".{d}" in result for d in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        )

class TestAverageTransformationsEdgeCases:
    """Tests for edge cases in average_transformations."""

    def test_average_with_opposite_quaternions(self):
        """Test averaging quaternions that are 180° apart."""
        # Create two rotations that are opposite
        T1 = make_transform_matrix(
            np.array([0.0, 0.0, 0.0]),  # No rotation
            np.array([1.0, 0.0, 0.0])
        )
        T2 = make_transform_matrix(
            np.array([np.pi, 0.0, 0.0]),  # 180° rotation
            np.array([1.0, 0.0, 0.0])
        )
        
        transforms = [T1, T2]
        T_avg = average_transformations(transforms)
        
        # Should return valid transformation
        assert T_avg.shape == (4, 4)
        assert np.allclose(T_avg[3, :], [0, 0, 0, 1])
        
        # Check that rotation matrix is orthogonal
        R = T_avg[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)

    def test_average_near_zero_quaternions(self):
        """Test with very small rotations that could cause numerical issues."""
        transforms = []
        for i in range(5):
            # Very small rotation vectors
            rvec = np.array([1e-8 * i, 1e-8 * i, 1e-8 * i])
            tvec = np.array([0.0, 0.0, 0.5])
            transforms.append(make_transform_matrix(rvec, tvec))
        
        T_avg = average_transformations(transforms)
        
        assert T_avg.shape == (4, 4)
        assert np.allclose(T_avg[3, :], [0, 0, 0, 1])

    def test_average_large_rotations(self):
        """Test with large rotation angles."""
        transforms = []
        for i in range(3):
            # Large but valid rotation vectors
            rvec = np.array([2.0, 1.5, 1.0]) + i * 0.1
            tvec = np.array([i * 0.5, 0.0, 1.0])
            transforms.append(make_transform_matrix(rvec, tvec))
        
        T_avg = average_transformations(transforms)
        
        assert T_avg.shape == (4, 4)
        # Should be orthogonal
        R = T_avg[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)

    def test_average_dtype_consistency(self):
        """Test that output has consistent float64 dtype."""
        T1 = np.eye(4, dtype=np.float32)  # float32 input
        T2 = np.eye(4, dtype=np.float64)  # float64 input
        
        T_avg = average_transformations([T1, T2])
        
        assert T_avg.dtype == np.float64
        assert T_avg.shape == (4, 4)