# tests/test_data_classes.py

"""Tests for data_classes.py"""

import pytest
import numpy as np
from cv2 import aruco

from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.data_classes import (
    BoardConfig, DataPoint, DetectionResult, PoseResult,
    MatchedDataPoint, IntrinsicCalibrationResult, ExtrinsicCalibrationResult
)


class TestBoardConfig:
    """Tests for BoardConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BoardConfig()
        
        assert config.num_squares == (14, 9)
        assert config.square_length == 0.020
        assert config.marker_length == 0.015
        assert config.dictionary == aruco.DICT_5X5_250

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BoardConfig(
            num_squares=(10, 8),
            square_length=0.030,
            marker_length=0.025,
            dictionary=aruco.DICT_6X6_250
        )
        
        assert config.num_squares == (10, 8)
        assert config.square_length == 0.030
        assert config.marker_length == 0.025
        assert config.dictionary == aruco.DICT_6X6_250

    def test_to_dict(self, default_board_config):
        """Test conversion to dictionary."""
        result = default_board_config.to_dict()
        
        assert isinstance(result, dict)
        assert result["num_squares"] == (14, 9)
        assert result["square_length"] == 0.020
        assert result["marker_length"] == 0.015
        assert result["dictionary"] == aruco.DICT_5X5_250

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "num_squares": [10, 8],
            "square_length": 0.025,
            "marker_length": 0.020,
            "dictionary": aruco.DICT_4X4_50
        }
        
        config = BoardConfig.from_dict(data)
        
        assert config.num_squares == (10, 8)
        assert config.square_length == 0.025
        assert config.marker_length == 0.020
        assert config.dictionary == aruco.DICT_4X4_50

    def test_from_dict_default_dictionary(self):
        """Test from_dict uses default dictionary if not provided."""
        data = {
            "num_squares": [5, 4],
            "square_length": 0.020,
            "marker_length": 0.015
        }
        
        config = BoardConfig.from_dict(data)
        
        assert config.dictionary == aruco.DICT_5X5_250


class TestDataPoint:
    """Tests for DataPoint dataclass."""

    def test_creation(self, random_image, identity_quaternion):
        """Test DataPoint creation."""
        position = np.array([1.0, 2.0, 3.0])
        transform = np.eye(4)
        
        dp = DataPoint(
            gripper_position=position,
            gripper_quaternion=identity_quaternion,
            T_gripper2base=transform,
            image=random_image,
            filename="test.png"
        )
        
        np.testing.assert_array_equal(dp.gripper_position, position)
        np.testing.assert_array_equal(dp.gripper_quaternion, identity_quaternion)
        np.testing.assert_array_equal(dp.T_gripper2base, transform)
        assert dp.image.shape == (480, 640, 3)
        assert dp.filename == "test.png"

    def test_filename_optional(self, random_image, identity_quaternion):
        """Test that filename is optional."""
        dp = DataPoint(
            gripper_position=np.zeros(3),
            gripper_quaternion=identity_quaternion,
            T_gripper2base=np.eye(4),
            image=random_image
        )
        
        assert dp.filename is None


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_valid_detection(self, mock_detection_valid):
        """Test valid detection properties."""
        assert mock_detection_valid.valid is True
        assert mock_detection_valid.num_corners == 20
        assert mock_detection_valid.num_markers == 10

    def test_invalid_detection(self, mock_detection_invalid):
        """Test invalid detection properties."""
        assert mock_detection_invalid.valid is False
        assert mock_detection_invalid.num_corners == 0
        assert mock_detection_invalid.num_markers == 0

    def test_min_ids_success(self, mock_detection_valid):
        """Test min_ids returns True when threshold met."""
        assert mock_detection_valid.min_ids(5, 10) is True
        assert mock_detection_valid.min_ids(10, 20) is True

    def test_min_ids_failure(self, mock_detection_valid):
        """Test min_ids returns False when threshold not met."""
        assert mock_detection_valid.min_ids(15, 25) is False
        assert mock_detection_valid.min_ids(5, 30) is False

    def test_min_ids_invalid_detection(self, mock_detection_invalid):
        """Test min_ids returns False for invalid detection."""
        assert mock_detection_invalid.min_ids(0, 0) is False


class TestPoseResult:
    """Tests for PoseResult dataclass."""

    def test_successful_pose(self, mock_pose_result):
        """Test successful pose result properties."""
        assert mock_pose_result.success is True
        assert mock_pose_result.rvec is not None
        assert mock_pose_result.tvec is not None
        assert mock_pose_result.transform is not None

    def test_rotation_matrix_property(self, mock_pose_result):
        """Test rotation_matrix property."""
        R = mock_pose_result.rotation_matrix
        
        assert R is not None
        assert R.shape == (3, 3)
        # Check orthogonality: R @ R.T should be identity
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)

    def test_translation_property(self, mock_pose_result):
        """Test translation property."""
        t = mock_pose_result.translation
        
        assert t is not None
        assert t.shape == (3,)

    def test_euler_angles_property(self, mock_pose_result):
        """Test euler_angles_deg property."""
        euler = mock_pose_result.euler_angles_deg
        
        assert euler is not None
        assert euler.shape == (3,)

    def test_failed_pose(self):
        """Test failed pose result."""
        pose = PoseResult(success=False)
        
        assert pose.success is False
        assert pose.rotation_matrix is None
        assert pose.translation is None
        assert pose.euler_angles_deg is None


class TestIntrinsicCalibrationResult:
    """Tests for IntrinsicCalibrationResult dataclass."""

    def test_properties(self, mock_intrinsic_result):
        """Test intrinsic result properties."""
        assert mock_intrinsic_result.rms_error == 0.5
        assert mock_intrinsic_result.num_images_used == 5
        assert mock_intrinsic_result.image_size == (640, 480)

    def test_focal_length_property(self, mock_intrinsic_result):
        """Test focal_length property."""
        fx, fy = mock_intrinsic_result.focal_length
        
        assert fx == 800.0
        assert fy == 800.0

    def test_principal_point_property(self, mock_intrinsic_result):
        """Test principal_point property."""
        cx, cy = mock_intrinsic_result.principal_point
        
        assert cx == 320.0
        assert cy == 240.0

    def test_empty_result(self):
        """Test empty result properties."""
        result = IntrinsicCalibrationResult()
        
        assert result.focal_length is None
        assert result.principal_point is None


class TestExtrinsicCalibrationResult:
    """Tests for ExtrinsicCalibrationResult dataclass."""

    def test_successful_result(self, sample_transform):
        """Test successful extrinsic result."""
        result = ExtrinsicCalibrationResult(
            success=True,
            rotation=sample_transform[:3, :3],
            translation=sample_transform[:3, 3],
            transform=sample_transform,
            num_poses_used=10,
            method="AXXB_TSAI"
        )
        
        assert result.success is True
        assert result.num_poses_used == 10
        assert result.method == "AXXB_TSAI"

    def test_failed_result(self):
        """Test failed extrinsic result."""
        result = ExtrinsicCalibrationResult(success=False)
        
        assert result.success is False
        assert result.transform is None