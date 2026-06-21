"""Visualization utilities."""

import cv2
import numpy as np

from .data_classes import DetectionResult, PoseResult


class CalibrationVisualizer:
    """Visualization for calibration results."""
    
    def __init__(
        self
    ):
        pass
    
    def draw_detection(self, image: np.ndarray, detection: DetectionResult) -> np.ndarray:
        """Draw detected markers and corners."""
        output = image.copy()
        if detection.marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(output, detection.marker_corners, detection.marker_ids)
        if detection.charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(output, detection.charuco_corners, detection.charuco_ids)
        return output
    
    def draw_pose(
        self,
        image: np.ndarray,
        pose: PoseResult,
        axis_length: float = 0.1,
        thickness: int = 3,
    ) -> np.ndarray:
        """Draw coordinate axes for pose."""
        if not pose.success or pose.camera_matrix is None:
            return image
        output = image.copy()
        cv2.drawFrameAxes(
            output, pose.camera_matrix, pose.dist_coeffs,
            pose.rvec, pose.tvec, axis_length, thickness
        )
        return output
    
    def draw_detection_and_pose(
        self,
        image: np.ndarray,
        pose: PoseResult,
        axis_length: float = 0.1,
    ) -> np.ndarray:
        """Draw detection and pose together."""
        output = image.copy()
        if pose.detection:
            output = self.draw_detection(output, pose.detection)
        return self.draw_pose(output, pose, axis_length)
    
    def visualize_coverage(
        self,
        detections: list[DetectionResult],
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Create heatmap of detected corner positions."""
        heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        
        for det in detections:
            if det.charuco_corners is not None:
                for corner in det.charuco_corners:
                    x, y = int(corner[0, 0]), int(corner[0, 1])
                    if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                        cv2.circle(heatmap, (x, y), 20, 1.0, -1)
        
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)