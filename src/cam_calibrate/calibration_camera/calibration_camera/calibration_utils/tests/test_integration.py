# tests/test_integration.py

"""Integration tests for charuco_calibration package."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from calibration_camera.calibration_camera.calibration_utils.charuco_calibration import (
    CharucoCalibration, BoardConfig, DataPoint, CharucoDetector, CalibrationIO
)
from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.transforms import make_transform_matrix_from_quat


@pytest.mark.integration
class TestEndToEndCalibration:
    """End-to-end integration tests."""

    def test_detection_with_generated_board(self, small_board_config):
        """Test that a generated board can be detected."""
        detector = CharucoDetector(small_board_config)
        
        # Generate board
        board_gray = detector.generate_board_image((600, 400))
        board_bgr = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
        
        # Create data point
        dp = DataPoint(
            gripper_position=np.zeros(3),
            gripper_quaternion=np.array([0, 0, 0, 1]),
            T_gripper2base=np.eye(4),
            image=board_bgr
        )
        
        # Detect
        result = detector.detect(dp)
        
        assert result.valid is True
        assert result.num_corners > 0
        assert result.num_markers > 0

    def test_io_roundtrip(self, temp_directory):
        """Test saving and loading data points."""
        # Create CalibrationIO and add data
        io = CalibrationIO(num_squares=(5, 4))
        
        for i in range(3):
            position = np.array([i * 0.1, 0.0, 0.5])
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
            
            # Add directly to _data_points (bypassing detection)
            io._data_points.append(DataPoint(
                gripper_position=position,
                gripper_quaternion=quaternion,
                T_gripper2base=make_transform_matrix_from_quat(quaternion, position),
                image=image
            ))
        
        # Save
        io.save_data_points(temp_directory, "test_data")
        
        # Load
        loaded_points, loaded_config = CalibrationIO.load_data_points(
            temp_directory / "test_data"
        )
        
        assert len(loaded_points) == 3
        np.testing.assert_array_almost_equal(
            loaded_points[0].gripper_position,
            np.array([0.0, 0.0, 0.5])
        )

    def test_calibration_io_intrinsics_extrinsics(self, temp_directory, camera_matrix, dist_coeffs, sample_transform):
        """Test saving and loading calibration results."""
        # Save intrinsics
        CalibrationIO.save_intrinsics(
            temp_directory / "intrinsics.json",
            camera_matrix,
            dist_coeffs
        )
        
        # Save extrinsics
        CalibrationIO.save_extrinsics(
            temp_directory / "extrinsics.json",
            sample_transform,
            name="T_camera_base"
        )
        
        # Load and verify
        loaded_cam, loaded_dist = CalibrationIO.load_intrinsics(
            temp_directory / "intrinsics.json"
        )
        loaded_ext = CalibrationIO.load_extrinsics(
            temp_directory / "extrinsics.json",
            name="T_camera_base"
        )
        
        np.testing.assert_array_almost_equal(loaded_cam, camera_matrix)
        np.testing.assert_array_almost_equal(loaded_dist, dist_coeffs)
        np.testing.assert_array_almost_equal(loaded_ext, sample_transform)


@pytest.mark.integration
class TestCharucoCalibrationWithRealDetection:
    """Integration tests using actual ChArUco detection."""

    def test_detect_generated_boards(self, small_board_config):
        """Test detection on multiple generated board images."""
        calib = CharucoCalibration(small_board_config)
        detector = CharucoDetector(small_board_config)
        
        # Generate board
        board_gray = detector.generate_board_image((500, 400))
        board_bgr = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
        
        # Create multiple data points with the same image
        # (In reality, you'd have different views)
        data_points = []
        for i in range(5):
            position = np.array([i * 0.05, 0.0, 0.4 + i * 0.02])
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            
            data_points.append(DataPoint(
                gripper_position=position,
                gripper_quaternion=quaternion,
                T_gripper2base=make_transform_matrix_from_quat(quaternion, position),
                image=board_bgr.copy()
            ))
        
        # Detect all boards
        detections = [detector.detect(dp) for dp in data_points]
        
        # All should be valid
        assert all(d.valid for d in detections)
        assert all(d.num_corners > 0 for d in detections)