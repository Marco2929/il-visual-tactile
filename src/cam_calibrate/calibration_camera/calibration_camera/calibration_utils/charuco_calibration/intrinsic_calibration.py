"""Intrinsic camera calibration."""

import cv2
import numpy as np
import logging

from .data_classes import DetectionResult, IntrinsicCalibrationResult
from .detection import CharucoDetector

logger = logging.getLogger(__name__)


class IntrinsicCalibrator:
    """Performs intrinsic camera calibration."""
    
    def __init__(self, detector: CharucoDetector):
        """
        Initialize calibrator.
        
        Args:
            detector: CharucoDetector instance
        """
        self._detector = detector
        self._board = detector.board
    
    def calibrate(
        self,
        detections: list[DetectionResult],
        flags: int = 0,
    ) -> IntrinsicCalibrationResult:
        """
        Calibrate camera from detections.
        
        Args:
            detections: List of DetectionResult (from multiple images)
            flags: OpenCV calibration flags
            
        Returns:
            IntrinsicCalibrationResult
        """
        valid = [d for d in detections if d.valid]
        
        if len(valid) < 3:
            raise ValueError(f"Need >= 3 valid detections, got {len(valid)}")
        
        obj_points_list = []
        img_points_list = []
        image_size = None
        
        for det in valid:
            obj_pts, img_pts = self._board.matchImagePoints(det.charuco_corners, det.charuco_ids)
            if obj_pts is not None and len(obj_pts) > 0:
                obj_points_list.append(obj_pts)
                img_points_list.append(img_pts)
                if image_size is None and det.data_point.image is not None:
                    h, w = det.data_point.image.shape[:2]
                    image_size = (w, h)
        
        if image_size is None:
            raise ValueError("Could not determine image size")
        
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list, img_points_list, image_size, None, None, flags=flags
        )
        
        # Per-view errors
        per_view_errors = []
        for obj_pts, img_pts, rvec, tvec in zip(obj_points_list, img_points_list, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
            error = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
            per_view_errors.append(error)
        
        logger.info(f"Calibration complete: RMS={rms:.4f}, images={len(obj_points_list)}")
        
        return IntrinsicCalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rms_error=rms,
            rvecs=rvecs,
            tvecs=tvecs,
            image_size=image_size,
            num_images_used=len(obj_points_list),
            per_view_errors=np.array(per_view_errors),
        )