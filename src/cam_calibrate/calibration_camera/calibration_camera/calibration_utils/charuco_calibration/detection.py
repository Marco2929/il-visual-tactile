"""ChArUco board detection."""

import cv2
import numpy as np
from cv2 import aruco
import logging

from .data_classes import DetectionResult, BoardConfig, DataPoint

logger = logging.getLogger(__name__)


class CharucoDetector:
    """Detects ChArUco boards in single images."""
    
    def __init__(self, config: BoardConfig):
        """
        Initialize detector.
        
        Args:
            config: Board configuration
        """
        self.config = config
        
        self._dictionary = aruco.getPredefinedDictionary(config.dictionary)
        self._board = aruco.CharucoBoard(
            config.num_squares,
            config.square_length,
            config.marker_length,
            self._dictionary,
        )
        
        charuco_params = aruco.CharucoParameters()
        detector_params = aruco.DetectorParameters()
        # FIXME detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        
        self._detector = aruco.CharucoDetector(
            self._board, charuco_params, detector_params
        )
        
        logger.debug(f"Initialized CharucoDetector: {config.num_squares}")
    
    @property
    def board(self) -> aruco.CharucoBoard:
        return self._board
    
    @property
    def detector(self) -> aruco.CharucoDetector:
        return self._detector
    
    def detect(self, data_point: DataPoint) -> DetectionResult:
        """
        Detect ChArUco board in a single image.
        
        Args:
            image: Input image (BGR or grayscale)
            filename: Optional filename for reference
            
        Returns:
            DetectionResult
        """
        gray = cv2.cvtColor(data_point.image, cv2.COLOR_BGR2GRAY) if len(data_point.image.shape) == 3 else data_point.image
        
        charuco_corners, charuco_ids, marker_corners, marker_ids = self._detector.detectBoard(gray)
        
        return DetectionResult(
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            marker_corners=marker_corners,
            marker_ids=marker_ids,
            data_point=data_point,
        )
    
    def generate_board_image(
        self,
        size: tuple[int, int] = (800, 600),
        margin: int = 10,
    ) -> np.ndarray:
        """Generate printable board image."""
        return self._board.generateImage(size, marginSize=margin)