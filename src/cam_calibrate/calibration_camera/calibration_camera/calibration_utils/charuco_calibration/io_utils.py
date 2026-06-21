"""File I/O utilities."""

import cv2
import numpy as np
from cv2 import aruco
from pathlib import Path
import json
import logging
from scipy.spatial.transform import Rotation as R
import yaml
from collections import OrderedDict

from .data_classes import BoardConfig, DataPoint
from .transforms import make_transform_matrix_from_quat
from .detection import CharucoDetector

logger = logging.getLogger(__name__)


class CalibrationIO:
    """Save/load calibration data."""

    def __init__(
        self,
        num_squares: tuple[int, int] = (14, 9),
        square_length: float = 0.020,
        marker_length: float = 0.015,
        dictionary: int = aruco.DICT_5X5_250,
        num_marker_ids: int = 0,
        num_charuco_ids: int = 0,
        ) -> None:
        """Initialize CalibrationIO."""
        self._num_marker_ids = num_marker_ids
        self._num_charuco_ids = num_charuco_ids
        self._data_points: list[DataPoint] = []
        self._board_config = BoardConfig(num_squares, square_length, marker_length, dictionary)
        self._detector = CharucoDetector(self._board_config)

    def get_aruco_dict_name(self, dict_id: int) -> str:
        """Get ArUco dictionary name from integer value."""
        import cv2.aruco as aruco
        
        # Suche alle DICT_ Attribute im aruco Modul
        for attr_name in dir(aruco):
            if attr_name.startswith('DICT_'):
                if getattr(aruco, attr_name) == dict_id:
                    return attr_name
        
        return str(dict_id)  # Fallback to integer if not found

    def get_aruco_dict_id(self, dict_name: str) -> int:
        """Get ArUco dictionary integer from name."""
        import cv2.aruco as aruco
        
        if hasattr(aruco, dict_name):
            return getattr(aruco, dict_name)
        
        return int(dict_name)  # Fallback to integer if not found

    def log_data(self, gripper_position: np.ndarray, gripper_quaternion: np.ndarray, image: np.ndarray) -> bool:
        """Log a single data point."""
        data_point = DataPoint(
            gripper_position=gripper_position,
            gripper_quaternion=gripper_quaternion,
            T_gripper2base=make_transform_matrix_from_quat(gripper_quaternion, gripper_position),
            image=image,
            filename=None
        )
        result = self._detector.detect(data_point=data_point)
        if result.min_ids(self._num_marker_ids, self._num_charuco_ids):
            logger.debug(f"Detected markers: {len(result.marker_ids)}, charuco corners: {len(result.charuco_ids)}")
            self._data_points.append(DataPoint(
            gripper_position=gripper_position,
            gripper_quaternion=gripper_quaternion,
            T_gripper2base=make_transform_matrix_from_quat(gripper_quaternion, gripper_position),
            image=image,
            filename=None
            ))
            logger.debug(f"Logged data point: pos={gripper_position}, quat={gripper_quaternion}")
            return True, result.num_markers, result.num_corners

        logger.debug(f"Insufficient marker ids or charuco ids: {result.num_markers}, {result.num_corners}")
        return False, result.num_markers, result.num_corners
    
    @property
    def data_points(self) -> list[DataPoint]:
        """Get logged data points."""
        return self._data_points
    
    def save_data_points(self, folderpath: str | Path, foldername: str) -> None:
        """Save data points to YAML file and images to PNG files in subfolder."""
        
        # Convert to Path object for easier handling
        base_path = Path(folderpath)
        folder_path = base_path / foldername
        images_path = folder_path / "images"
        
        # Create directories if they don't exist
        folder_path.mkdir(parents=True, exist_ok=True)
        images_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for YAML
        yaml_data = {
            'board_config': {
                'num_squares': list(self._board_config.num_squares),
                'square_length': self._board_config.square_length,
                'marker_length': self._board_config.marker_length,
                'dictionary': self.get_aruco_dict_name(self._board_config.dictionary)
            },
            'data_points': []
        }
        
        # Process each data point
        for i, data_point in enumerate(self._data_points):
            # Generate image filename
            image_filename = f"image_{i:04d}.png"
            image_path = images_path / image_filename
            
            # Save image
            cv2.imwrite(str(image_path), data_point.image)
            
            # Add data point to YAML structure
            point_data = {
                'gripper_position': {
                    'x': float(data_point.gripper_position[0]),
                    'y': float(data_point.gripper_position[1]),
                    'z': float(data_point.gripper_position[2])
                },
                'gripper_quaternion': {
                    'x': float(data_point.gripper_quaternion[0]),
                    'y': float(data_point.gripper_quaternion[1]),
                    'z': float(data_point.gripper_quaternion[2]),
                    'w': float(data_point.gripper_quaternion[3])
                },
                'T_gripper2base': data_point.T_gripper2base.tolist(),
                'image_filename': f"images/{image_filename}"
            }
            yaml_data['data_points'].append(point_data)
        
        # Save YAML file
        yaml_file_path = folder_path / "calibration_data.yaml"
        with open(yaml_file_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"Saved {len(self._data_points)} data points to {yaml_file_path}")
        logger.info(f"Saved {len(self._data_points)} images to {images_path}")
        
    @staticmethod
    def load_data_points(folderpath: str | Path) -> tuple[list[DataPoint], BoardConfig]:
        """Load data points from YAML file and images from subfolder."""
        
        # Convert to Path object - this points to main folder
        main_folder = Path(folderpath)
        yaml_file_path = main_folder / "calibration_data.yaml"
        images_folder = main_folder / "images"
        
        # Check if YAML file exists
        if not yaml_file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file_path}")
        
        # Load YAML data
        with open(yaml_file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        loaded_data_points = []

        board_cfg = BoardConfig(
            num_squares=tuple(yaml_data['board_config']['num_squares']),
            square_length=yaml_data['board_config']['square_length'],
            marker_length=yaml_data['board_config']['marker_length'],
            dictionary=CalibrationIO().get_aruco_dict_id(yaml_data['board_config']['dictionary'])
        )
        
        # Process each data point
        for point_data in yaml_data['data_points']:
            # Reconstruct gripper position from dict
            gripper_position = np.array([
                point_data['gripper_position']['x'],
                point_data['gripper_position']['y'],
                point_data['gripper_position']['z']
            ])
            
            # Reconstruct gripper quaternion from dict (x, y, z, w format)
            gripper_quaternion = np.array([
                point_data['gripper_quaternion']['x'],
                point_data['gripper_quaternion']['y'],
                point_data['gripper_quaternion']['z'],
                point_data['gripper_quaternion']['w']
            ])
            
            # Reconstruct transformation matrix
            T_gripper2base = np.array(point_data['T_gripper2base'])
            
            # Load image - remove "images/" prefix since we already have the images_folder path
            image_filename = point_data['image_filename'].replace("images/", "")
            image_path = images_folder / image_filename
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                continue
            
            # Create DataPoint object
            data_point = DataPoint(
                gripper_position=gripper_position,
                gripper_quaternion=gripper_quaternion,
                T_gripper2base=T_gripper2base,
                image=image,
                filename=str(image_path)
            )
            
            loaded_data_points.append(data_point)
        
        logger.info(f"Loaded {len(loaded_data_points)} data points from {yaml_file_path}")
        
        return loaded_data_points, board_cfg
        
    
    @staticmethod
    def save_intrinsics(
        filepath: str | Path,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        board_config: BoardConfig | None = None,
    ) -> None:
        """Save intrinsics to JSON."""
        data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.flatten().tolist(),
        }
        if board_config:
            data["board_config"] = board_config.to_dict()
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved intrinsics to {filepath}")
    
    @staticmethod
    def load_intrinsics(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load intrinsics from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded intrinsics from {filepath}")
        return np.array(data["camera_matrix"]), np.array(data["dist_coeffs"])
    
    @staticmethod
    def save_extrinsics(
        filepath: str | Path,
        transform: np.ndarray,
        name: str = "T_base_to_camera",
    ) -> None:
        """Save extrinsic transform to JSON."""
        t = transform[:3, 3]
        q = R.from_matrix(transform[:3, :3]).as_quat()
        euler = R.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
        
        data = {
            name: transform.tolist(),
            f"{name}_translation": t.tolist(),
            f"{name}_quaternion_xyzw": q.tolist(),
            f"{name}_euler_xyz_deg": euler.tolist(),
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved extrinsics to {filepath}")
    
    @staticmethod
    def load_extrinsics(filepath: str | Path, name: str = "T_base_to_camera") -> np.ndarray:
        """Load extrinsic transform from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded extrinsics from {filepath}")
        return np.array(data[name])