# tests/conftest.py

"""
Shared pytest fixtures for all tests.

Fixtures defined here are automatically available in all test files.
"""

import pytest
import numpy as np
import cv2
from cv2 import aruco
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock

from calibration_camera.calibration_camera.calibration_utils.charuco_calibration import (
    BoardConfig, DataPoint, DetectionResult, PoseResult,
    IntrinsicCalibrationResult, ExtrinsicCalibrationResult,
    CharucoDetector, CharucoCalibration
)
from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.transforms import make_transform_matrix, make_transform_matrix_from_quat


# =============================================================================

# Board Configuration Fixtures

# =============================================================================

@pytest.fixture
def default_board_config():
    """Default board configuration."""
    return BoardConfig()


@pytest.fixture
def small_board_config():
    """Smaller board for faster tests."""
    return BoardConfig(
        num_squares=(5, 4),
        square_length=0.025,
        marker_length=0.020,
        dictionary=aruco.DICT_4X4_50
    )

@pytest.fixture
def mock_charuco_board():
    """Mock CharucoBoard that can be used in place of real board."""
    mock_board = Mock()
    
    # Mock matchImagePoints method
    def mock_match_image_points(charuco_corners, charuco_ids):
        # Return some realistic-looking object and image points
        num_corners = len(charuco_corners) if charuco_corners is not None else 10
        object_points = np.random.rand(num_corners, 3).astype(np.float32)
        image_points = np.random.rand(num_corners, 2).astype(np.float32)
        return object_points, image_points
    
    mock_board.matchImagePoints.side_effect = mock_match_image_points
    return mock_board


# =============================================================================

# Detector Fixtures

# =============================================================================

@pytest.fixture
def charuco_detector(small_board_config):
    """CharucoDetector instance with small board."""
    return CharucoDetector(small_board_config)


# =============================================================================

# Image Fixtures

# =============================================================================

@pytest.fixture
def random_image():
    """Random noise image (480x640 BGR)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_image():
    """Random grayscale image."""
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)


@pytest.fixture
def board_image(charuco_detector):
    """Generated ChArUco board image (BGR)."""
    board_gray = charuco_detector.generate_board_image((400, 300))
    return cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)


# =============================================================================

# Transform Fixtures

# =============================================================================

@pytest.fixture
def identity_transform():
    """4x4 identity transformation matrix."""
    return np.eye(4)


@pytest.fixture
def sample_transform():
    """Sample transformation with rotation and translation."""
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([1.0, 2.0, 3.0])
    return make_transform_matrix(rvec, tvec)


@pytest.fixture
def identity_quaternion():
    """Identity quaternion [x, y, z, w]."""
    return np.array([0.0, 0.0, 0.0, 1.0])


# =============================================================================

# DataPoint Fixtures

# =============================================================================

@pytest.fixture
def sample_data_point(random_image, identity_quaternion):
    """Single DataPoint with random image."""
    position = np.array([1.0, 2.0, 3.0])
    return DataPoint(
        gripper_position=position,
        gripper_quaternion=identity_quaternion,
        T_gripper2base=make_transform_matrix_from_quat(identity_quaternion, position),
        image=random_image,
        filename="test_image.png"
    )


@pytest.fixture
def data_point_with_board(board_image, identity_quaternion):
    """DataPoint with generated board image."""
    position = np.array([0.0, 0.0, 0.5])
    return DataPoint(
        gripper_position=position,
        gripper_quaternion=identity_quaternion,
        T_gripper2base=make_transform_matrix_from_quat(identity_quaternion, position),
        image=board_image,
        filename="board_image.png"
    )


@pytest.fixture
def multiple_data_points(identity_quaternion):
    """List of 5 DataPoints with varying poses."""
    data_points = []
    for i in range(5):
        position = np.array([i * 0.1, i * 0.05, 0.5 + i * 0.02])
        # Slight rotation variation
        angle = i * 0.05
        quaternion = np.array([0.0, 0.0, np.sin(angle/2), np.cos(angle/2)])
        transform = make_transform_matrix_from_quat(quaternion, position)
        image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        data_points.append(DataPoint(
            gripper_position=position,
            gripper_quaternion=quaternion,
            T_gripper2base=transform,
            image=image,
            filename=f"image_{i:04d}.png"
        ))
    return data_points


# =============================================================================

# Camera Parameter Fixtures

# =============================================================================

@pytest.fixture
def camera_matrix():
    """Typical camera intrinsic matrix."""
    return np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


@pytest.fixture
def dist_coeffs():
    """Typical distortion coefficients."""
    return np.array([0.1, -0.25, 0.0, 0.0, 0.1], dtype=np.float64)


@pytest.fixture
def camera_params(camera_matrix, dist_coeffs):
    """Tuple of (camera_matrix, dist_coeffs)."""
    return camera_matrix, dist_coeffs


# =============================================================================

# Mock Detection Fixtures

# =============================================================================

@pytest.fixture
def mock_detection_valid(sample_data_point):
    """Mock DetectionResult that is valid."""
    return DetectionResult(
        charuco_corners=np.random.rand(20, 1, 2).astype(np.float32) * 640,
        charuco_ids=np.arange(20).reshape(-1, 1).astype(np.int32),
        marker_corners=[np.random.rand(4, 1, 2).astype(np.float32) * 640 for _ in range(10)],
        marker_ids=np.arange(10).reshape(-1, 1).astype(np.int32),
        data_point=sample_data_point
    )


@pytest.fixture
def mock_detection_invalid():
    """Mock DetectionResult that is invalid (no corners)."""
    return DetectionResult(
        charuco_corners=None,
        charuco_ids=None,
        marker_corners=None,
        marker_ids=None,
        data_point=None
    )


# =============================================================================

# Mock Result Fixtures

# =============================================================================

@pytest.fixture
def mock_intrinsic_result(camera_matrix, dist_coeffs):
    """Mock IntrinsicCalibrationResult."""
    return IntrinsicCalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms_error=0.5,
        rvecs=[np.array([0.1, 0.2, 0.3]) for _ in range(5)],
        tvecs=[np.array([0.0, 0.0, 0.5]) for _ in range(5)],
        image_size=(640, 480),
        num_images_used=5,
        per_view_errors=np.array([0.4, 0.5, 0.6, 0.5, 0.4])
    )


@pytest.fixture
def mock_pose_result(sample_transform, camera_matrix, dist_coeffs, mock_detection_valid):
    """Mock PoseResult."""
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([0.0, 0.0, 0.5])
    return PoseResult(
        success=True,
        rvec=rvec,
        tvec=tvec,
        transform=make_transform_matrix(rvec, tvec),
        detection=mock_detection_valid,
        reprojection_error=0.5,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )


# =============================================================================

# File I/O Fixtures

# =============================================================================

@pytest.fixture
def temp_directory():
    """Temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_directory):
    """Temporary file path."""
    return temp_directory / "test_file.json"