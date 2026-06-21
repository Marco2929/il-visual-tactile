# tests/test_calibration.py

"""Tests for calibration.py (main CharucoCalibration class)"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.calibration import CharucoCalibration
from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.data_classes import (
    BoardConfig, DataPoint, DetectionResult, PoseResult,
    IntrinsicCalibrationResult, ExtrinsicCalibrationResult
)


class TestCharucoCalibrationInit:
    """Tests for CharucoCalibration initialization."""

    def test_initialization(self, small_board_config):
        """Test CharucoCalibration initializes correctly."""
        calib = CharucoCalibration(small_board_config)
        
        assert calib._board_config == small_board_config
        assert calib._detector is not None
        assert calib._intrinsic_calibrator is not None
        assert calib._extrinsic_calibrator is not None
        assert calib._visualizer is not None
        
        # Initial state
        assert len(calib._detections) == 0
        assert len(calib._matched_poses) == 0
        assert calib._intrinsics is None
        assert len(calib._extrinsics) == 0

    def test_initialization_default_config(self):
        """Test with default board config."""
        default_config = BoardConfig()
        calib = CharucoCalibration(default_config)
        
        assert calib._board_config.num_squares == (14, 9)


class TestCharucoCalibrationHandToEye:
    """Tests for CharucoCalibration.calibrate_hand_to_eye method."""

    @patch('charuco_calibration.calibration.CharucoDetector.detect')
    @patch('charuco_calibration.calibration.IntrinsicCalibrator.calibrate')
    @patch('charuco_calibration.calibration.PoseEstimator.estimate')
    @patch('charuco_calibration.calibration.ExtrinsicCalibrator.calibrate_hand_to_eye')
    def test_calibrate_hand_to_eye_pipeline(
        self, mock_extrinsic, mock_pose, mock_intrinsic, mock_detect, 
        small_board_config, multiple_data_points
    ):
        """Test the complete calibration pipeline."""
        calib = CharucoCalibration(small_board_config)
        
        # Setup mocks
        mock_detection = Mock(spec=DetectionResult)
        mock_detection.valid = True
        mock_detection.data_point = multiple_data_points[0]
        mock_detect.return_value = mock_detection
        
        mock_intrinsic_result = Mock(spec=IntrinsicCalibrationResult)
        mock_intrinsic_result.camera_matrix = np.eye(3)
        mock_intrinsic_result.dist_coeffs = np.zeros(5)
        mock_intrinsic.return_value = mock_intrinsic_result
        
        mock_pose_result = Mock(spec=PoseResult)
        mock_pose_result.transform = np.eye(4)
        mock_pose.return_value = mock_pose_result
        
        mock_extrinsic_result = Mock(spec=ExtrinsicCalibrationResult)
        mock_extrinsic_result.transform = np.eye(4)
        mock_extrinsic_result.success = True
        mock_extrinsic_result.method = "INTERNAL_SHAH"
        mock_extrinsic.return_value = mock_extrinsic_result
        
        # Run calibration
        calib.calibrate_hand_to_eye(multiple_data_points, method_name="INTERNAL_SHAH")
        
        # Verify calls
        assert mock_detect.call_count == len(multiple_data_points)
        mock_intrinsic.assert_called_once()
        mock_extrinsic.assert_called_once()
        
        # Verify state
        assert len(calib._detections) == len(multiple_data_points)
        assert calib._intrinsics is not None
        assert len(calib._extrinsics) == 1

    @patch('charuco_calibration.calibration.CharucoDetector.detect')
    def test_calibrate_filters_invalid_detections(self, mock_detect, small_board_config, multiple_data_points):
        """Test that invalid detections are filtered."""
        calib = CharucoCalibration(small_board_config)
        
        # Alternate valid/invalid detections
        def side_effect(dp):
            mock_det = Mock(spec=DetectionResult)
            mock_det.valid = (multiple_data_points.index(dp) % 2 == 0)
            mock_det.data_point = dp
            return mock_det
        
        mock_detect.side_effect = side_effect
        
        with patch.object(calib._intrinsic_calibrator, 'calibrate') as mock_intrinsic:
            mock_intrinsic.side_effect = ValueError("Need >= 3 valid detections")
            
            with pytest.raises(ValueError):
                calib.calibrate_hand_to_eye(multiple_data_points, method_name="INTERNAL_SHAH")

    # def test_calibrate_unknown_method_raises(self, small_board_config, multiple_data_points):
    #     """Test that unknown method raises ValueError."""
    #     calib = CharucoCalibration(small_board_config)
        
    #     # Mock the detection step to return valid detections
    #     with patch.object(calib._detector, 'detect') as mock_detect:
    #         mock_det = Mock(spec=DetectionResult)
    #         mock_det.valid = True
    #         mock_det.data_point = multiple_data_points[0]
    #         # Add proper charuco detection data to avoid OpenCV errors
    #         mock_det.charuco_corners = np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2)
    #         mock_det.charuco_ids = np.array([[0]], dtype=np.int32)
    #         mock_detect.return_value = mock_det
            
    #         # Mock intrinsic calibration to succeed
    #         with patch.object(calib._intrinsic_calibrator, 'calibrate') as mock_intrinsic:
    #             mock_intrinsic_result = Mock(spec=IntrinsicCalibrationResult)
    #             mock_intrinsic_result.camera_matrix = np.eye(3)
    #             mock_intrinsic_result.dist_coeffs = np.zeros(5)
    #             mock_intrinsic.return_value = mock_intrinsic_result
                
    #             # Mock pose estimation to succeed
    #             with patch.object(calib, '_pose_estimator', create=True) as mock_pose_est:
    #                 mock_pose_result = Mock(spec=PoseResult)
    #                 mock_pose_result.transform = np.eye(4)
    #                 mock_pose_est.estimate.return_value = mock_pose_result
                    
    #                 # This should reach the method validation and raise ValueError
    #                 with pytest.raises(ValueError, match="Unknown method"):
    #                     calib.calibrate_hand_to_eye(multiple_data_points, method_name="INVALID_METHOD")


class TestCharucoCalibrationSaveResults:
    """Tests for CharucoCalibration.save_results method."""

    def test_save_results_creates_directory(self, temp_directory, small_board_config):
        """Test that save_results creates output directory."""
        calib = CharucoCalibration(small_board_config)
        output_dir = temp_directory / "results"
        
        calib.save_results(str(output_dir))
        
        assert output_dir.exists()

    def test_save_results_with_intrinsics(self, temp_directory, small_board_config, mock_intrinsic_result):
        """Test saving results with intrinsics."""
        calib = CharucoCalibration(small_board_config)
        calib._intrinsics = mock_intrinsic_result
        
        calib.save_results(str(temp_directory))
        
        assert (temp_directory / "intrinsic_calibration.json").exists()

    def test_save_results_with_extrinsics(self, temp_directory, small_board_config, sample_transform):
        """Test saving results with extrinsics."""
        calib = CharucoCalibration(small_board_config)
        
        mock_result = Mock(spec=ExtrinsicCalibrationResult)
        mock_result.success = True
        mock_result.transform = sample_transform
        mock_result.method = "INTERNAL_SHAH"
        calib._extrinsics = [mock_result]
        
        calib.save_results(str(temp_directory))
        
        assert (temp_directory / "extrinsic_calibration_internal_shah.json").exists()