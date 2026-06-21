#!/usr/bin/env python3
"""
ROS2 Service Node for logging calibration data.
Subscribes to robot poses, calls RGBD capture service on demand,
provides services to log and save data.
"""

import sys
import os
import time

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from interfaces.srv import (
    LogCalibrationData,
    SaveCalibrationData,
    ClearCalibrationData,
    GetCalibrationStatus,
    RunCalibration,
)
from interfaces.srv import GetRGBD


class CalibrationNode(Node):
    """Node that provides services for logging calibration data."""

    def __init__(self):
        super().__init__('calibration')

        # Callback group to allow nested service calls
        self._callback_group = ReentrantCallbackGroup()

        # =====================
        # Declare Parameters
        # =====================
        self._declare_parameters()

        # =====================
        # Load Parameters
        # =====================
        self._load_parameters()

        # =====================
        # Log Configuration
        # =====================
        self._log_configuration()

        # =====================
        # Initialize CalibrationIO
        # =====================
        self._init_calibration_io()

        # =====================
        # State
        # =====================
        self._bridge = CvBridge()
        self._latest_pose = None
        self._latest_pose_stamp = None
        self._last_saved_path = None

        # =====================
        # Service Client (RGBD Capture)
        # =====================
        self._rgbd_client = self.create_client(
            GetRGBD,
            self._rgbd_service_name,
            callback_group=self._callback_group,
        )

        # =====================
        # Subscribers
        # =====================
        self._pose_sub = self.create_subscription(
            PoseStamped,
            self._pose_topic,
            self._pose_callback,
            10,
        )

        # =====================
        # Services
        # =====================
        self._log_srv = self.create_service(
            LogCalibrationData,
            '~/log_data',
            self._log_data_callback,
            callback_group=self._callback_group,
        )

        self._save_srv = self.create_service(
            SaveCalibrationData,
            '~/save_data',
            self._save_data_callback,
            callback_group=self._callback_group,
        )

        self._clear_srv = self.create_service(
            ClearCalibrationData,
            '~/clear_data',
            self._clear_data_callback,
            callback_group=self._callback_group,
        )

        self._status_srv = self.create_service(
            GetCalibrationStatus,
            '~/get_status',
            self._status_callback,
            callback_group=self._callback_group,
        )

        self._run_calibration_srv = self.create_service(
            RunCalibration,
            '~/run_calibration',
            self._run_calibration_callback,
            callback_group=self._callback_group,
        )

        self.get_logger().info('Calibration Node initialized successfully ✓')

    # =====================
    # Parameter Management
    # =====================

    def _declare_parameters(self):
        """Declare all parameters with default values."""
        self.declare_parameter('rgbd_service', '/get_rgbd')
        self.declare_parameter('rgbd_service_timeout', 20.0)
        self.declare_parameter('pose_topic', '/example_topic/robot/pose')

        # Output directory
        self.declare_parameter('output_base_dir', '')

        # Board configuration
        self.declare_parameter('board.num_squares_x', 14)
        self.declare_parameter('board.num_squares_y', 9)
        self.declare_parameter('board.square_length', 0.020)
        self.declare_parameter('board.marker_length', 0.015)
        self.declare_parameter('board.dictionary', 'DICT_5X5_250')

        # Detection requirements
        self.declare_parameter('min_marker_ids', 4)
        self.declare_parameter('min_charuco_ids', 6)

    def _load_parameters(self):
        """Load parameter values into instance variables."""
        self._rgbd_service_name = self.get_parameter('rgbd_service').value
        self._rgbd_timeout = self.get_parameter('rgbd_service_timeout').value
        self._pose_topic = self.get_parameter('pose_topic').value
        self.get_logger().info(f'Loaded parameters: RGBD service="{self._rgbd_service_name}", '
                               f'timeout={self._rgbd_timeout}s, pose topic="{self._pose_topic}"')

        # Output directory
        self._output_base_dir = self._resolve_output_dir()

        # Board configuration
        self._board_config = {
            'num_squares_x': self.get_parameter('board.num_squares_x').value,
            'num_squares_y': self.get_parameter('board.num_squares_y').value,
            'square_length': self.get_parameter('board.square_length').value,
            'marker_length': self.get_parameter('board.marker_length').value,
            'dictionary': self.get_parameter('board.dictionary').value,
        }

        # Detection requirements
        self._min_marker_ids = self.get_parameter('min_marker_ids').value
        self._min_charuco_ids = self.get_parameter('min_charuco_ids').value

    def _resolve_output_dir(self) -> str:
        """Resolve output directory from parameter or environment variable."""
        output_dir = self.get_parameter('output_base_dir').value

        if output_dir:
            return output_dir

        env_dir = os.environ.get('CALIBRATION_DATA_DIR')
        if env_dir:
            self.get_logger().info(f'Using CALIBRATION_DATA_DIR: {env_dir}')
            return env_dir

        # Fallback: fester Pfad
        default = '/ros2_ws/data/images'
        self.get_logger().warn(f'No output dir specified, using default: {default}')
        return default

    def _log_configuration(self):
        """Log the loaded configuration."""
        self.get_logger().info('=' * 60)
        self.get_logger().info('Calibration Data Logger Configuration')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  RGBD service:    {self._rgbd_service_name}')
        self.get_logger().info(f'  Service timeout: {self._rgbd_timeout} s')
        self.get_logger().info(f'  Pose topic:      {self._pose_topic}')
        self.get_logger().info(f'  Output dir:      {self._output_base_dir}')
        self.get_logger().info(f'  Board config:')
        self.get_logger().info(f'    - Squares:     {self._board_config["num_squares_x"]}x{self._board_config["num_squares_y"]}')
        self.get_logger().info(f'    - Square size: {self._board_config["square_length"]} m')
        self.get_logger().info(f'    - Marker size: {self._board_config["marker_length"]} m')
        self.get_logger().info(f'    - Dictionary:  {self._board_config["dictionary"]}')
        self.get_logger().info(f'  Min markers:     {self._min_marker_ids}')
        self.get_logger().info(f'  Min charuco:     {self._min_charuco_ids}')
        self.get_logger().info('=' * 60)

    def _init_calibration_io(self):
        """Initialize CalibrationIO with board config."""
        try:
            from calibration_camera.calibration_utils.charuco_calibration.io_utils import CalibrationIO
            from cv2 import aruco

            num_squares = (
                self._board_config['num_squares_x'],
                self._board_config['num_squares_y'],
            )
            square_length = self._board_config['square_length']
            marker_length = self._board_config['marker_length']
            dict_name = self._board_config['dictionary']

            dictionary = getattr(aruco, dict_name, aruco.DICT_5X5_250)

            self._calib_io = CalibrationIO(
                num_squares=num_squares,
                square_length=square_length,
                marker_length=marker_length,
                dictionary=dictionary,
                num_marker_ids=self._min_marker_ids,
                num_charuco_ids=self._min_charuco_ids,
            )

            self.get_logger().info(f'CalibrationIO initialized: {num_squares}, {dict_name}')

        except ImportError as e:
            self.get_logger().error('=' * 60)
            self.get_logger().error(f'Could not import calibration module: {e}')
            self.get_logger().error('=' * 60)
            raise

    # =====================
    # RGBD Capture (FIXED)
    # =====================

    def _capture_rgbd(self) -> tuple:
        """
        Call the RGBD capture service using busy-wait pattern.

        Returns:
            tuple: (rgb_image_cv2, depth_image_cv2, rgb_camera_info) or (None, None, None) on failure
        """
        # ── 1. Check service availability ───────────────────────────
        if not self._rgbd_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f'RGBD service "{self._rgbd_service_name}" not available. '
                f'Is the camera server running?'
            )
            return None, None, None

        # ── 2. Call service async ───────────────────────────────────
        self.get_logger().info('Calling RGBD capture service...')
        request = GetRGBD.Request()
        future = self._rgbd_client.call_async(request)

        # ── 3. Busy-wait until done or timeout ──────────────────────
        start_time = time.time()
        while not future.done():
            elapsed = time.time() - start_time
            if elapsed > self._rgbd_timeout:
                self.get_logger().error(
                    f'RGBD capture service call timed out after {self._rgbd_timeout}s. '
                    f'Increase "rgbd_service_timeout" param or check camera server.'
                )
                future.cancel()
                return None, None, None
            time.sleep(0.01)

        # ── 4. Check result ─────────────────────────────────────────
        result = future.result()

        if result is None:
            self.get_logger().error('RGBD service returned no result')
            return None, None, None

        if not result.success:
            self.get_logger().error(f'RGBD capture failed: {result.message}')
            return None, None, None

        self.get_logger().info(f'RGBD capture succeeded: {result.message}')
        rgbd = result.rgbd

        # ── 5. Validate data ───────────────────────────────────────
        if not rgbd.rgb.data:
            self.get_logger().error('RGB image data is empty from camera')
            return None, None, None

        if not rgbd.depth.data:
            self.get_logger().error('Depth image data is empty from camera')
            return None, None, None

        if rgbd.rgb.width == 0 or rgbd.rgb.height == 0:
            self.get_logger().error(
                f'RGB image has invalid dimensions: '
                f'{rgbd.rgb.width}x{rgbd.rgb.height}')
            return None, None, None

        if rgbd.depth.width == 0 or rgbd.depth.height == 0:
            self.get_logger().error(
                f'Depth image has invalid dimensions: '
                f'{rgbd.depth.width}x{rgbd.depth.height}')
            return None, None, None

        # ── 6. Handle encoding defaults ────────────────────────────
        if not rgbd.rgb.encoding:
            self.get_logger().warn(
                "RGB encoding empty, defaulting to 'bgra8'")
            rgbd.rgb.encoding = "bgra8"

        if not rgbd.depth.encoding:
            self.get_logger().warn(
                "Depth encoding empty, defaulting to '32FC1'")
            rgbd.depth.encoding = "32FC1"

        # ── 7. Convert RGB (handle RGBA/BGRA) ─────────────────────
        try:
            if rgbd.rgb.encoding in ("bgra8", "rgba8", "8UC4"):
                self.get_logger().info(
                    f'Converting {rgbd.rgb.encoding} → bgr8')
                rgba = self._bridge.imgmsg_to_cv2(
                    rgbd.rgb, desired_encoding="bgra8")

                if rgba is None or rgba.size == 0:
                    self.get_logger().error(
                        f'RGBA conversion failed: image is empty. '
                        f'Original: {rgbd.rgb.width}x{rgbd.rgb.height}, '
                        f'data_len={len(rgbd.rgb.data)}')
                    return None, None, None

                rgb_image = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
            else:
                rgb_image = self._bridge.imgmsg_to_cv2(rgbd.rgb, 'bgr8')

            if rgb_image is None or rgb_image.size == 0:
                self.get_logger().error('RGB image conversion resulted in empty array')
                return None, None, None

        except Exception as e:
            self.get_logger().error(f'Failed to convert RGB image: {e}')
            return None, None, None

        # ── 8. Convert depth ───────────────────────────────────────
        try:
            self.get_logger().info(
                f"Depth msg: encoding='{rgbd.depth.encoding}', "
                f"{rgbd.depth.width}x{rgbd.depth.height}, "
                f"step={rgbd.depth.step}, data_len={len(rgbd.depth.data)}")

            depth_image = self._bridge.imgmsg_to_cv2(
                rgbd.depth, desired_encoding='passthrough')

            self.get_logger().info(
                f"Depth array: dtype={depth_image.dtype}, "
                f"shape={depth_image.shape}, "
                f"min={np.nanmin(depth_image):.4f}, "
                f"max={np.nanmax(depth_image):.4f}")

            # Convert to float32 meters
            if depth_image.dtype == np.uint16:
                # uint16 assumed in mm → convert to meters
                depth_image = depth_image.astype(np.float32) / 1000.0
            elif depth_image.dtype == np.float32:
                # float32 from server: already in mm (depth_to_mm=True) → to meters
                depth_image = depth_image / 1000.0
            else:
                self.get_logger().warn(
                    f'Unexpected depth dtype: {depth_image.dtype}, '
                    f'assuming mm → converting to meters')
                depth_image = depth_image.astype(np.float32) / 1000.0

            # Log NaN stats
            nan_count = np.count_nonzero(np.isnan(depth_image))
            if nan_count > 0:
                self.get_logger().warn(
                    f'Depth has {nan_count} NaN values '
                    f'({nan_count / depth_image.size * 100:.1f}%)')

            self.get_logger().info(
                f"Final depth (m): min={np.nanmin(depth_image):.4f}, "
                f"max={np.nanmax(depth_image):.4f}, "
                f"mean={np.nanmean(depth_image):.4f}")

        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return None, None, None

        # ── 9. Return ──────────────────────────────────────────────
        rgb_camera_info = rgbd.rgb_camera_info
        self.get_logger().info(
            f'Capture complete: RGB {rgb_image.shape}, '
            f'Depth {depth_image.shape} ({depth_image.dtype})')

        return rgb_image, depth_image, rgb_camera_info

    # =====================
    # Subscriber Callbacks
    # =====================

    def _pose_callback(self, msg: PoseStamped):
        """Store latest robot pose."""
        self._latest_pose = msg.pose
        self._latest_pose_stamp = msg.header.stamp

    # =====================
    # Service Callbacks
    # =====================

    def _log_data_callback(
        self,
        request: LogCalibrationData.Request,
        response: LogCalibrationData.Response
    ) -> LogCalibrationData.Response:
        """Log current camera image and robot pose."""

        # Initialize response fields
        response.num_markers = 0
        response.num_charuco = 0

        # Check pose availability
        if self._latest_pose is None:
            response.success = False
            response.message = 'No robot pose available'
            response.num_logged_points = len(self._calib_io.data_points)
            return response

        # Capture RGBD image on demand
        self.get_logger().info('Capturing RGBD image...')
        rgb_image, depth_image, camera_info = self._capture_rgbd()

        if rgb_image is None:
            response.success = False
            response.message = 'Failed to capture RGBD image from service'
            response.num_logged_points = len(self._calib_io.data_points)
            return response

        # Extract pose data
        pose = self._latest_pose
        position = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z
        ])
        quaternion = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])

        # Log data
        try:
            success, num_markers, num_charuco = self._calib_io.log_data(
                position, quaternion, rgb_image
            )

            response.num_markers = num_markers
            response.num_charuco = num_charuco

            if success:
                num_points = len(self._calib_io.data_points)
                response.success = True
                response.message = f'Logged data point {num_points}'
                response.num_logged_points = num_points
                self.get_logger().info(
                    f'Logged data point {num_points} '
                    f'(markers: {num_markers}, charuco: {num_charuco})'
                )
            else:
                response.success = False
                response.message = (
                    f'ChArUco detection failed: '
                    f'{num_markers} markers (min {self._min_marker_ids}), '
                    f'{num_charuco} charuco (min {self._min_charuco_ids})'
                )
                response.num_logged_points = len(self._calib_io.data_points)
                self.get_logger().warn(response.message)

        except Exception as e:
            response.success = False
            response.message = f'Error logging data: {str(e)}'
            response.num_logged_points = len(self._calib_io.data_points)
            self.get_logger().error(f'Error logging data: {e}')

        return response

    def _save_data_callback(
        self,
        request: SaveCalibrationData.Request,
        response: SaveCalibrationData.Response
    ) -> SaveCalibrationData.Response:
        """Save all logged data to disk."""

        num_points = len(self._calib_io.data_points)

        if num_points == 0:
            response.success = False
            response.message = 'No data points logged'
            response.saved_path = ''
            response.num_saved_points = 0
            return response

        folder_name = request.folder_name
        if not folder_name:
            folder_name = datetime.now().strftime('calibration_%Y%m%d_%H%M%S')

        try:
            self._calib_io.save_data_points(self._output_base_dir, folder_name)

            saved_path = str(Path(self._output_base_dir) / folder_name)
            self._last_saved_path = saved_path

            response.success = True
            response.message = f'Saved {num_points} data points'
            response.saved_path = saved_path
            response.num_saved_points = num_points

            self.get_logger().info(f'Saved {num_points} data points to {saved_path}')

        except Exception as e:
            response.success = False
            response.message = f'Error saving data: {str(e)}'
            response.saved_path = ''
            response.num_saved_points = 0
            self.get_logger().error(f'Error saving data: {e}')

        return response

    def _clear_data_callback(
        self,
        request: ClearCalibrationData.Request,
        response: ClearCalibrationData.Response
    ) -> ClearCalibrationData.Response:
        """Clear all logged data."""

        num_points = len(self._calib_io.data_points)
        self._init_calibration_io()

        response.success = True
        response.message = f'Cleared {num_points} data points'
        response.num_cleared_points = num_points

        self.get_logger().info(f'Cleared {num_points} data points')
        return response

    def _status_callback(
        self,
        request: GetCalibrationStatus.Request,
        response: GetCalibrationStatus.Response
    ) -> GetCalibrationStatus.Response:
        """Get current calibration logger status."""

        response.num_logged_points = len(self._calib_io.data_points)
        response.camera_data_available = self._rgbd_client.service_is_ready()
        response.robot_pose_available = self._latest_pose is not None

        return response

    def _run_calibration_callback(
        self,
        request: RunCalibration.Request,
        response: RunCalibration.Response
    ) -> RunCalibration.Response:
        """Run hand-eye calibration on saved data."""

        from calibration_camera.calibration_utils.charuco_calibration.io_utils import CalibrationIO
        from calibration_camera.calibration_utils.charuco_calibration.calibration import CharucoCalibration

        data_dir = request.data_dir if request.data_dir else self._last_saved_path

        if not data_dir:
            response.success = False
            response.message = (
                'No data directory specified and no data was saved previously. '
                'Either provide data_dir or call ~/save_data first.'
            )
            response.results_path = ''
            return response

        if not Path(data_dir).exists():
            response.success = False
            response.message = f'Data directory does not exist: {data_dir}'
            response.results_path = ''
            return response

        self.get_logger().info(f'Using data directory: {data_dir}')

        try:
            data_points, board_config = CalibrationIO.load_data_points(data_dir)

            if len(data_points) < 3:
                response.success = False
                response.message = (
                    f'Not enough data points: {len(data_points)} (minimum 3 required). '
                    f'Recommend at least 8-12 diverse poses.'
                )
                response.results_path = ''
                return response

            if len(data_points) < 8:
                self.get_logger().warn(
                    f'Only {len(data_points)} data points. '
                    f'Recommend at least 8-12 for robust calibration.'
                )

            calibration = CharucoCalibration(board_config)
            method = request.method if request.method else None

            if request.hand_in_eye:
                self.get_logger().info(
                    f'Running hand-in-eye calibration with {len(data_points)} images...'
                )
                success = calibration.calibrate_hand_in_eye(data_points, method_name=method)
            else:
                self.get_logger().info(
                    f'Running hand-to-eye calibration with {len(data_points)} images...'
                )
                success = calibration.calibrate_hand_to_eye(data_points, method_name=method)

            if success:
                calibration.save_results(data_dir, hand_to_eye=not request.hand_in_eye)
                response.success = True
                response.message = f'Calibration successful using {len(data_points)} images'
                response.results_path = data_dir
                self.get_logger().info(f'Calibration completed successfully ✓ → {data_dir}')
            else:
                response.success = False
                response.message = (
                    'Calibration failed. Possible causes:\n'
                    '  - Not enough images (have {}, recommend ≥8)\n'
                    '  - Insufficient rotation diversity (need ≥15° between poses)\n'
                    '  - Poor ChArUco detections\n'
                    'Tip: Vary both position AND orientation significantly between captures.'
                ).format(len(data_points))
                response.results_path = ''

        except cv2.error as e:
            response.success = False
            response.message = (
                f'OpenCV calibration error: {str(e)}. '
                f'This usually means insufficient rotational diversity. '
                f'Capture more poses with larger rotation differences (≥15-20°).'
            )
            response.results_path = ''
            self.get_logger().error(f'OpenCV error: {e}')

        except Exception as e:
            response.success = False
            response.message = f'Calibration error: {str(e)}'
            response.results_path = ''
            self.get_logger().error(f'Calibration failed: {e}')

        return response


def main(args=None):
    rclpy.init(args=args)

    node = CalibrationNode()

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()