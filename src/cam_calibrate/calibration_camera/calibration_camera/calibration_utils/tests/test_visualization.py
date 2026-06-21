# tests/test_visualization.py

"""Tests for visualization.py"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.visualization import CalibrationVisualizer
from calibration_camera.calibration_camera.calibration_utils.charuco_calibration.data_classes import DetectionResult, PoseResult


class TestCalibrationVisualizerInit:
    """Tests for CalibrationVisualizer initialization."""

    def test_initialization(self):
        """Test visualizer initialization."""
        vis = CalibrationVisualizer()
        assert vis is not None


class TestCalibrationVisualizerDrawDetection:
    """Tests for CalibrationVisualizer.draw_detection method."""

    def test_draw_detection_returns_image(self, random_image, mock_detection_valid):
        """Test draw_detection returns an image."""
        vis = CalibrationVisualizer()
        
        result = vis.draw_detection(random_image, mock_detection_valid)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == random_image.shape

    def test_draw_detection_does_not_modify_original(self, random_image, mock_detection_valid):
        """Test that original image is not modified."""
        vis = CalibrationVisualizer()
        original = random_image.copy()
        
        vis.draw_detection(random_image, mock_detection_valid)
        
        np.testing.assert_array_equal(random_image, original)

    def test_draw_detection_with_no_markers(self, random_image, mock_detection_invalid):
        """Test drawing detection with no markers."""
        vis = CalibrationVisualizer()
        
        result = vis.draw_detection(random_image, mock_detection_invalid)
        
        assert result.shape == random_image.shape


class TestCalibrationVisualizerDrawPose:
    """Tests for CalibrationVisualizer.draw_pose method."""

    def test_draw_pose_returns_image(self, random_image, mock_pose_result):
        """Test draw_pose returns an image."""
        vis = CalibrationVisualizer()
        
        result = vis.draw_pose(random_image, mock_pose_result)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == random_image.shape

    def test_draw_pose_failed_returns_original(self, random_image):
        """Test draw_pose with failed pose returns original image."""
        vis = CalibrationVisualizer()
        failed_pose = PoseResult(success=False)
        
        result = vis.draw_pose(random_image, failed_pose)
        
        np.testing.assert_array_equal(result, random_image)

    def test_draw_pose_no_camera_matrix_returns_original(self, random_image):
        """Test draw_pose without camera matrix returns original."""
        vis = CalibrationVisualizer()
        pose = PoseResult(success=True, camera_matrix=None)
        
        result = vis.draw_pose(random_image, pose)
        
        np.testing.assert_array_equal(result, random_image)

    def test_draw_pose_axis_length(self, random_image, mock_pose_result):
        """Test draw_pose with different axis lengths."""
        vis = CalibrationVisualizer()
        
        result1 = vis.draw_pose(random_image, mock_pose_result, axis_length=0.05)
        result2 = vis.draw_pose(random_image, mock_pose_result, axis_length=0.2)
        
        # Both should return valid images (content will differ)
        assert result1.shape == random_image.shape
        assert result2.shape == random_image.shape


class TestCalibrationVisualizerDrawDetectionAndPose:
    """Tests for CalibrationVisualizer.draw_detection_and_pose method."""

    def test_draw_detection_and_pose(self, random_image, mock_pose_result):
        """Test combined detection and pose drawing."""
        vis = CalibrationVisualizer()
        
        result = vis.draw_detection_and_pose(random_image, mock_pose_result)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == random_image.shape


class TestCalibrationVisualizerVisualizeCoverage:
    """Tests for CalibrationVisualizer.visualize_coverage method."""

    def test_visualize_coverage_returns_heatmap(self, mock_detection_valid):
        """Test coverage visualization returns colored heatmap."""
        vis = CalibrationVisualizer()
        detections = [mock_detection_valid for _ in range(5)]
        image_size = (640, 480)
        
        result = vis.visualize_coverage(detections, image_size)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)  # BGR heatmap

    def test_visualize_coverage_empty_detections(self):
        """Test coverage with empty detections."""
        vis = CalibrationVisualizer()
        image_size = (640, 480)
        
        result = vis.visualize_coverage([], image_size)
        
        assert result.shape == (480, 640, 3)