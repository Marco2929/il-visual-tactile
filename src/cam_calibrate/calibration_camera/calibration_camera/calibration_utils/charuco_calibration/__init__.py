"""
CharucoCalibration - ChArUco board calibration for OpenCV 4.7+ and Python 3.12+

Basic usage:
    from charuco_calibration import CharucoCalibrationHelper
    
    helper = CharucoCalibrationHelper(num_squares=(14, 9))
    result = helper.calibrate_intrinsics_from_directory("./images")
    pose = helper.estimate_pose(image)

Advanced usage (individual components):
    from charuco_calibration import CharucoDetector, BoardConfig
    
    config = BoardConfig(num_squares=(14, 9))
    detector = CharucoDetector(config)
    detection = detector.detect(image)
"""

# Data classes

from .data_classes import (
    BoardConfig,
    DetectionResult,
    PoseResult,
    IntrinsicCalibrationResult,
    ExtrinsicCalibrationResult,
    DataPoint,
    MatchedDataPoint
)

# Component classes

from .detection import CharucoDetector
from .intrinsic_calibration import IntrinsicCalibrator
from .pose_estimation import PoseEstimator
from .extrinsic_calibration import ExtrinsicCalibrator
from .visualization import CalibrationVisualizer
from .io_utils import CalibrationIO

# Main facade

from .calibration import CharucoCalibration

# Utility functions

from .transforms import (
    make_transform_matrix,
    make_transform_matrix_from_quat,
    invert_transform,
    average_transformations,
    transform_to_pose_string,
)

__version__ = "1.0.0"

__all__ = [
    # Main class
    "CharucoCalibration",
    # Data classes
    "BoardConfig",
    "DetectionResult",
    "PoseResult",
    "IntrinsicCalibrationResult",
    "ExtrinsicCalibrationResult",
    "DataPoint",
    "MatchedDataPoint",
    # Components
    "CharucoDetector",
    "IntrinsicCalibrator",
    "PoseEstimator",
    "ExtrinsicCalibrator",
    "CalibrationVisualizer",
    "CalibrationIO",
    # Functions
    "make_transform_matrix",
    "invert_transform",
    "average_transformations",
    "transform_to_pose_string",
    "make_transform_matrix_from_quat",
]