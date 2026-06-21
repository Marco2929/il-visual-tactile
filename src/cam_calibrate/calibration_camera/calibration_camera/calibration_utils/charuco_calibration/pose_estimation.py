"""Pose estimation."""

import cv2
import numpy as np
import logging
import copy

from .data_classes import DetectionResult, PoseResult
from .detection import CharucoDetector
from .transforms import make_transform_matrix

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Estimates board pose from images."""
    
    def __init__(
        self,
        detector: CharucoDetector,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ):
        self._detector = detector
        self._board = detector.board
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def estimate(
        self,
        detection: DetectionResult,
        undistort: bool = True,
        pnp_flags: int = cv2.SOLVEPNP_ITERATIVE,
        refine: bool = True,
    ) -> PoseResult:
        """
        Estimate board pose from single image.
        
        Args:
            image: Input image
            undistort: Undistort image first
            pnp_flags: solvePnP flags
            refine: Refine with LM
            
        Returns:
            PoseResult
        """
        if undistort:
            proc_image, cam_mtx = self.undistort_image(detection.data_point.image, alpha=1.0)
            new_detection = copy.deepcopy(detection)
            new_detection.data_point.image = proc_image
            dist = np.zeros(5)
            detection = self._detector.detect(data_point=new_detection.data_point)
        else:
            proc_image = detection.data_point.image
            cam_mtx = self.camera_matrix
            dist = self.dist_coeffs
        
        if not detection.valid:
            return PoseResult(success=False, detection=detection)
        
        if detection.charuco_corners is None or detection.charuco_ids is None:
            return PoseResult(success=False, detection=detection)

        if len(detection.charuco_ids) < 4:
            logger.debug(
                "Insufficient charuco corners for solvePnP: %d",
                len(detection.charuco_ids),
            )
            return PoseResult(success=False, detection=detection)

        obj_pts, img_pts = self._board.matchImagePoints(
            detection.charuco_corners, detection.charuco_ids
        )

        if obj_pts is None or img_pts is None or len(obj_pts) < 4 or len(img_pts) < 4:
            logger.debug(
                "Insufficient matched points for solvePnP: obj=%s img=%s",
                0 if obj_pts is None else len(obj_pts),
                0 if img_pts is None else len(img_pts),
            )
            return PoseResult(success=False, detection=detection)
        
        # FIXME check how matrix is returned (cam_to_target or target_to_cam)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mtx, dist, flags=pnp_flags)
        
        if not success:
            return PoseResult(success=False, detection=detection)
        
        if refine:
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, cam_mtx, dist, rvec, tvec)
        
        # Reprojection error
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist)
        reproj_error = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
        
        return PoseResult(
            success=True,
            rvec=rvec,
            tvec=tvec,
            transform=make_transform_matrix(rvec, tvec),
            detection=detection,
            reprojection_error=reproj_error,
            camera_matrix=cam_mtx,
            dist_coeffs=dist,
        )
    
    def undistort_image(self, image: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Undistort image, return (undistorted, new_camera_matrix)."""
        h, w = image.shape[:2]
        new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha, (w, h)
        )
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_cam_mtx)
        return undistorted, new_cam_mtx