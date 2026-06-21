"""Data classes for calibration results and configuration."""

import numpy as np
import cv2
from cv2 import aruco
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

#FIXME Fix namings and add comments for the variables and methods

@dataclass
class DataPoint:
    """Container for a single calibration data point."""
    gripper_position: np.ndarray
    gripper_quaternion: np.ndarray # in (x, y, z, w) format
    T_gripper2base: np.ndarray # gripper relative to base
    image: np.ndarray
    filename: str | None = None
    
@dataclass
class BoardConfig:
    """Configuration for ChArUco board."""
    num_squares: tuple[int, int] = (14, 9)
    square_length: float = 0.020
    marker_length: float = 0.015
    dictionary: int = aruco.DICT_5X5_250
    
    def to_dict(self) -> dict:
        return {
            "num_squares": self.num_squares,
            "square_length": self.square_length,
            "marker_length": self.marker_length,
            "dictionary": self.dictionary,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BoardConfig":
        return cls(
            num_squares=tuple(data["num_squares"]),
            square_length=data["square_length"],
            marker_length=data["marker_length"],
            dictionary=data.get("dictionary", aruco.DICT_5X5_250),
        )


@dataclass
class DetectionResult:
    """Container for ChArUco board detection results."""
    charuco_corners: np.ndarray | None = None
    charuco_ids: np.ndarray | None = None
    marker_corners: list[np.ndarray] | None = None
    marker_ids: np.ndarray | None = None
    data_point: DataPoint | None = None
    
    def min_ids(self, min_num_marker_ids: int, min_num_charuco_ids: int) -> bool:
        return (
            self.charuco_ids is not None
            and self.num_corners >= min_num_charuco_ids
            and self.marker_ids is not None
            and self.num_markers >= min_num_marker_ids
        )

    @property
    def valid(self) -> bool:
        return (
            self.charuco_ids is not None
            and self.num_corners >= 0
            and self.marker_ids is not None
            and self.num_markers >= 0
        )
    
    @property
    def num_corners(self) -> int:
        return len(self.charuco_ids) if self.charuco_ids is not None else 0
    
    @property
    def num_markers(self) -> int:
        return len(self.marker_ids) if self.marker_ids is not None else 0

# FIXME add another dataclass for rvec,tvec and transform
# include target2gripper in CalibrationResult
@dataclass
class PoseResult:
    """Container for pose estimation results."""
    success: bool = False
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    transform: np.ndarray | None = None
    detection: DetectionResult | None = None
    reprojection_error: float | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    
    @property
    def rotation_matrix(self) -> np.ndarray | None:
        if self.rvec is not None:
            return cv2.Rodrigues(self.rvec)[0]
        return None
    
    @property
    def translation(self) -> np.ndarray | None:
        return self.tvec.flatten() if self.tvec is not None else None
    
    @property
    def euler_angles_deg(self) -> np.ndarray | None:
        if self.rvec is not None:
            return np.degrees(R.from_rotvec(self.rvec.flatten()).as_euler("xyz"))
        return None

@dataclass
class MatchedDataPoint:
    detection: DetectionResult | None = None
    estimated_pose: PoseResult | None = None

@dataclass
class IntrinsicCalibrationResult:
    """Container for intrinsic calibration results."""
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    rms_error: float = 0.0
    rvecs: list[np.ndarray] | None = None
    tvecs: list[np.ndarray] | None = None
    image_size: tuple[int, int] | None = None
    num_images_used: int = 0
    per_view_errors: np.ndarray | None = None
    
    @property
    def focal_length(self) -> tuple[float, float] | None:
        if self.camera_matrix is not None:
            return (self.camera_matrix[0, 0], self.camera_matrix[1, 1])
        return None
    
    @property
    def principal_point(self) -> tuple[float, float] | None:
        if self.camera_matrix is not None:
            return (self.camera_matrix[0, 2], self.camera_matrix[1, 2])
        return None


@dataclass
class ExtrinsicCalibrationResult:
    """Container for hand-eye calibration results."""
    success: bool = False
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    transform: np.ndarray | None = None
    residual_error: float | None = None
    num_poses_used: int = 0
    method: str = ""