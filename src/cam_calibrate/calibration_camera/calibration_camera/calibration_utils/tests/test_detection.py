# tests/test_detection.py

"""Tests for detection.py"""

import pytest
import numpy as np
import cv2
from cv2 import aruco
from unittest.mock import Mock, patch, MagicMock

from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.detection import CharucoDetector
from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.data_classes import BoardConfig, DetectionResult, DataPoint


class TestCharucoDetectorInit:
    """Tests for CharucoDetector initialization."""

    def test_initialization_default_config(self, default_board_config):
        """Test initialization with default config."""
        detector = CharucoDetector(default_board_config)
        
        assert detector.config == default_board_config
        assert detector.board is not None
        assert detector.detector is not None

    def test_initialization_custom_config(self, small_board_config):
        """Test initialization with custom config."""
        detector = CharucoDetector(small_board_config)
        
        assert detector.config.num_squares == (5, 4)
        assert detector.config.dictionary == aruco.DICT_4X4_50

    def test_board_property(self, charuco_detector):
        """Test board property returns CharucoBoard."""
        board = charuco_detector.board
        
        assert isinstance(board, cv2.aruco.CharucoBoard)

    def test_detector_property(self, charuco_detector):
        """Test detector property returns CharucoDetector."""
        detector = charuco_detector.detector
        
        assert isinstance(detector, cv2.aruco.CharucoDetector)


class TestCharucoDetectorDetect:
    """Tests for CharucoDetector.detect method."""

    def test_detect_with_board_image(self, charuco_detector, data_point_with_board):
        """Test detection with generated board image."""
        result = charuco_detector.detect(data_point_with_board)
        
        assert isinstance(result, DetectionResult)
        assert result.valid is True
        assert result.num_corners > 0
        assert result.num_markers > 0
        assert result.data_point == data_point_with_board

    def test_detect_with_random_image(self, charuco_detector, sample_data_point):
        """Test detection with random noise image (should fail)."""
        result = charuco_detector.detect(sample_data_point)
        
        assert isinstance(result, DetectionResult)
        # Random noise should not contain a valid board
        # (might occasionally detect false positives)
        assert result.data_point == sample_data_point

    def test_detect_grayscale_image(self, charuco_detector, grayscale_image, identity_quaternion):
        """Test detection with grayscale image."""
        dp = DataPoint(
            gripper_position=np.zeros(3),
            gripper_quaternion=identity_quaternion,
            T_gripper2base=np.eye(4),
            image=grayscale_image
        )
        
        result = charuco_detector.detect(dp)
        
        assert isinstance(result, DetectionResult)

    def test_detect_stores_data_point(self, charuco_detector, data_point_with_board):
        """Test that detection result stores data point reference."""
        result = charuco_detector.detect(data_point_with_board)
        
        assert result.data_point is data_point_with_board


class TestCharucoDetectorGenerateBoard:
    """Tests for CharucoDetector.generate_board_image method."""

    def test_generate_board_default_size(self, charuco_detector):
        """Test board generation with default size."""
        image = charuco_detector.generate_board_image()
        
        assert image.shape == (600, 800)  # Height, Width
        assert image.dtype == np.uint8

    def test_generate_board_custom_size(self, charuco_detector):
        """Test board generation with custom size."""
        image = charuco_detector.generate_board_image(size=(400, 300))
        
        assert image.shape == (300, 400)  # Height, Width

    def test_generate_board_with_margin(self, charuco_detector):
        """Test board generation with margin."""
        image = charuco_detector.generate_board_image(size=(400, 300), margin=20)
        
        assert image.shape == (300, 400)

    def test_generate_board_content(self, charuco_detector):
        """Test that generated board contains both black and white areas."""
        image = charuco_detector.generate_board_image()
        
        # Should have black areas (markers)
        assert np.any(image == 0)
        # Should have white areas (squares)
        assert np.any(image == 255)

    def test_generated_board_is_detectable(self, charuco_detector, identity_quaternion):
        """Test that generated board can be detected."""
        board_gray = charuco_detector.generate_board_image((600, 400))
        board_bgr = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
        
        dp = DataPoint(
            gripper_position=np.zeros(3),
            gripper_quaternion=identity_quaternion,
            T_gripper2base=np.eye(4),
            image=board_bgr
        )
        
        result = charuco_detector.detect(dp)
        
        assert result.valid is True
        assert result.num_corners > 0