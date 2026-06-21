# tests/test_io_utils.py

"""Tests for io_utils.py"""

import pytest
import numpy as np
import cv2
from cv2 import aruco
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.io_utils import CalibrationIO
from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.data_classes import BoardConfig, DataPoint


class TestCalibrationIOInit:
    """Tests for CalibrationIO initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        io = CalibrationIO()
        
        assert io._num_marker_ids == 0
        assert io._num_charuco_ids == 0
        assert len(io._data_points) == 0
        assert io._board_config is not None

    def test_custom_initialization(self):
        """Test custom initialization."""
        io = CalibrationIO(
            num_squares=(10, 8),
            square_length=0.030,
            marker_length=0.025,
            dictionary=aruco.DICT_4X4_50,
            num_marker_ids=5,
            num_charuco_ids=10
        )
        
        assert io._num_marker_ids == 5
        assert io._num_charuco_ids == 10
        assert io._board_config.num_squares == (10, 8)


class TestCalibrationIOArucoDictNames:
    """Tests for ArUco dictionary name conversion."""

    def test_get_aruco_dict_name(self):
        """Test getting dictionary name from ID."""
        io = CalibrationIO()
        
        name = io.get_aruco_dict_name(aruco.DICT_5X5_250)
        assert name == "DICT_5X5_250"
        
        name = io.get_aruco_dict_name(aruco.DICT_4X4_50)
        assert name == "DICT_4X4_50"

    def test_get_aruco_dict_id(self):
        """Test getting dictionary ID from name."""
        io = CalibrationIO()
        
        dict_id = io.get_aruco_dict_id("DICT_5X5_250")
        assert dict_id == aruco.DICT_5X5_250
        
        dict_id = io.get_aruco_dict_id("DICT_4X4_50")
        assert dict_id == aruco.DICT_4X4_50

    def test_get_aruco_dict_name_unknown(self):
        """Test unknown dictionary returns string representation."""
        io = CalibrationIO()
        
        # Use an invalid ID
        name = io.get_aruco_dict_name(9999)
        assert name == "9999"


class TestCalibrationIOLogData:
    """Tests for CalibrationIO.log_data method."""

    def test_log_data_success(self):
        """Test successful data logging."""
        io = CalibrationIO(num_marker_ids=0, num_charuco_ids=0)
        
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock detector to always succeed
        with patch.object(io._detector, 'detect') as mock_detect:
            mock_result = Mock()
            mock_result.min_ids.return_value = True
            mock_result.marker_ids = np.arange(10).reshape(-1, 1)
            mock_result.charuco_ids = np.arange(20).reshape(-1, 1)
            mock_detect.return_value = mock_result
            
            success = io.log_data(position, quaternion, image)
        
        assert success is True
        assert len(io.data_points) == 1
        np.testing.assert_array_equal(io.data_points[0].gripper_position, position)

    def test_log_data_failure(self):
        """Test failed data logging (insufficient markers)."""
        io = CalibrationIO(num_marker_ids=10, num_charuco_ids=20)
        
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch.object(io._detector, 'detect') as mock_detect:
            mock_result = Mock()
            mock_result.min_ids.return_value = False
            mock_result.num_markers = 5
            mock_result.num_corners = 10
            mock_detect.return_value = mock_result
            
            success = io.log_data(position, quaternion, image)
        
        assert success is False
        assert len(io.data_points) == 0


class TestCalibrationIOSaveLoad:
    """Tests for CalibrationIO save/load methods."""

    def test_save_and_load_intrinsics(self, temp_directory, camera_matrix, dist_coeffs):
        """Test saving and loading intrinsics."""
        filepath = temp_directory / "intrinsics.json"
        
        # Save
        CalibrationIO.save_intrinsics(filepath, camera_matrix, dist_coeffs)
        
        # Verify file exists
        assert filepath.exists()
        
        # Load
        loaded_cam, loaded_dist = CalibrationIO.load_intrinsics(filepath)
        
        np.testing.assert_array_almost_equal(loaded_cam, camera_matrix)
        np.testing.assert_array_almost_equal(loaded_dist, dist_coeffs)

    def test_save_intrinsics_with_board_config(self, temp_directory, camera_matrix, dist_coeffs, small_board_config):
        """Test saving intrinsics with board config."""
        filepath = temp_directory / "intrinsics.json"
        
        CalibrationIO.save_intrinsics(filepath, camera_matrix, dist_coeffs, small_board_config)
        
        # Verify board config is in file
        with open(filepath) as f:
            data = json.load(f)
        
        assert "board_config" in data
        assert data["board_config"]["num_squares"] == [5, 4]

    def test_save_and_load_extrinsics(self, temp_directory, sample_transform):
        """Test saving and loading extrinsics."""
        filepath = temp_directory / "extrinsics.json"
        
        # Save
        CalibrationIO.save_extrinsics(filepath, sample_transform, name="T_test")
        
        # Verify file exists
        assert filepath.exists()
        
        # Load
        loaded = CalibrationIO.load_extrinsics(filepath, name="T_test")
        
        np.testing.assert_array_almost_equal(loaded, sample_transform)

    def test_save_extrinsics_contains_all_formats(self, temp_directory, sample_transform):
        """Test that saved extrinsics contain matrix, translation, quaternion, euler."""
        filepath = temp_directory / "extrinsics.json"
        
        CalibrationIO.save_extrinsics(filepath, sample_transform, name="T_cam")
        
        with open(filepath) as f:
            data = json.load(f)
        
        assert "T_cam" in data
        assert "T_cam_translation" in data
        assert "T_cam_quaternion_xyzw" in data
        assert "T_cam_euler_xyz_deg" in data

    def test_save_and_load_data_points(self, temp_directory, multiple_data_points):
        """Test saving and loading data points."""
        io = CalibrationIO()
        io._data_points = multiple_data_points
        
        # Save
        io.save_data_points(temp_directory, "test_calibration")
        
        # Verify structure
        save_path = temp_directory / "test_calibration"
        assert save_path.exists()
        assert (save_path / "calibration_data.yaml").exists()
        assert (save_path / "images").exists()
        
        # Load
        loaded_points, loaded_config = CalibrationIO.load_data_points(save_path)
        
        assert len(loaded_points) == len(multiple_data_points)
        np.testing.assert_array_almost_equal(
            loaded_points[0].gripper_position,
            multiple_data_points[0].gripper_position
        )

    def test_load_data_points_missing_file(self, temp_directory):
        """Test loading from non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            CalibrationIO.load_data_points(temp_directory / "nonexistent")