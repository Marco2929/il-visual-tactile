import cv2
import numpy as np
from cv2 import aruco
from pathlib import Path
import logging

from .data_classes import (
    BoardConfig, DetectionResult, PoseResult,
    IntrinsicCalibrationResult, ExtrinsicCalibrationResult, DataPoint, MatchedDataPoint
)
from .detection import CharucoDetector
from .intrinsic_calibration import IntrinsicCalibrator
from .pose_estimation import PoseEstimator
from .extrinsic_calibration import ExtrinsicCalibrator
from .visualization import CalibrationVisualizer
from .io_utils import CalibrationIO
from .transforms import make_transform_matrix, average_transformations, transform_to_pose_string

logger = logging.getLogger(__name__)


class CharucoCalibration:

    def __init__(self, board_config: BoardConfig):
        self._board_config = board_config
        self._detector = CharucoDetector(board_config)
        self._intrinsic_calibrator = IntrinsicCalibrator(self._detector)
        self._extrinsic_calibrator = ExtrinsicCalibrator()
        self._pose_estimator = None
        self._hand_eye_calibrator = ExtrinsicCalibrator()
        self._visualizer = CalibrationVisualizer()

        self._detections = []
        self._matched_poses = []
        self._intrinsics = None
        self._extrinsics = []

        self._valid_T_gripper2base = []
        self._valid_T_target2camera = []

    # ═══════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════

    def calibrate_hand_to_eye(
        self, data_points: list[DataPoint], method_name: str | None = None
    ) -> bool:
        """
        Run hand-to-eye calibration pipeline.

        Args:
            data_points: List of DataPoint objects with images and TCP poses.
            method_name: Specific method to use, or None to try all.

        Returns:
            True if at least one method succeeded.
        """
        # ── 1. Detect ChArUco in all images ────────────────────────
        if not self._run_detection(data_points):
            return False

        # ── 2. Intrinsic calibration ──────────────────────────────
        if not self._run_intrinsic_calibration():
            return False

        # ── 3. Pose estimation ────────────────────────────────────
        if not self._run_pose_estimation():
            return False

        # ── 4. Check rotation diversity ───────────────────────────
        if not self._check_rotation_diversity(self._valid_T_gripper2base):
            return False

        # ── 5. Extrinsic calibration (hand-to-eye) ───────────────
        return self._run_extrinsic_hand_to_eye(method_name)

    def calibrate_hand_in_eye(
        self, data_points: list[DataPoint], method_name: str | None = None
    ) -> bool:
        """
        Run hand-in-eye calibration pipeline.

        Args:
            data_points: List of DataPoint objects with images and TCP poses.
            method_name: Specific method to use, or None to try all.

        Returns:
            True if at least one method succeeded.
        """
        # ── 1. Detect ChArUco in all images ────────────────────────
        if not self._run_detection(data_points):
            return False

        # ── 2. Intrinsic calibration ──────────────────────────────
        if not self._run_intrinsic_calibration():
            return False

        # ── 3. Pose estimation ────────────────────────────────────
        if not self._run_pose_estimation():
            return False

        # ── 4. Check rotation diversity ───────────────────────────
        if not self._check_rotation_diversity(self._valid_T_gripper2base):
            return False

        # ── 5. Extrinsic calibration (hand-in-eye) ───────────────
        return self._run_extrinsic_hand_in_eye(method_name)

    def visualize(self):
        """Show detections and pose overlays."""
        for pose in self._matched_poses:
            if pose.detection and pose.detection.data_point.image is not None:
                vis = self._visualizer.draw_detection_and_pose(
                    pose.detection.data_point.image,
                    pose.estimated_pose,
                    axis_length=0.2,
                )
                cv2.imshow("Detection + Pose", vis)
                key = cv2.waitKey(50)
                if key == ord('q'):
                    break
        cv2.destroyAllWindows()

    def save_results(self, output_dir: str, hand_to_eye: bool = True):
        """Save intrinsic and extrinsic calibration results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save intrinsics
        if (
            self._intrinsics
            and self._intrinsics.camera_matrix is not None
            and self._intrinsics.dist_coeffs is not None
        ):
            CalibrationIO.save_intrinsics(
                output_path / "intrinsic_calibration.json",
                self._intrinsics.camera_matrix,
                self._intrinsics.dist_coeffs,
                self._board_config,
            )
            logger.info(f"Saved intrinsics to {output_path / 'intrinsic_calibration.json'}")

        # Save extrinsics
        if hand_to_eye:
            name = "T_camera2base"
        else:
            name = "T_camera2gripper"

        saved_count = 0
        for result in self._extrinsics:
            if result.success and result.transform is not None:
                method_name = result.method.replace(" ", "_").lower()
                filepath = output_path / f"extrinsic_calibration_{method_name}.json"
                CalibrationIO.save_extrinsics(filepath, result.transform, name=name)
                logger.info(f"Saved extrinsics ({method_name}) to {filepath}")
                saved_count += 1

        logger.info(f"Saved {saved_count} extrinsic calibration result(s) to {output_path}")

    # ═══════════════════════════════════════════════════════════════
    # Private Pipeline Steps
    # ═══════════════════════════════════════════════════════════════

    def _run_detection(self, data_points: list[DataPoint]) -> bool:
        """Detect ChArUco board in all images."""
        self._detections = []

        for i, data_point in enumerate(data_points):
            detection = self._detector.detect(data_point)
            if detection.valid:
                self._detections.append(detection)
                logger.debug(f"Image {i}: detection valid")
            else:
                logger.warning(f"Image {i}: detection failed, skipping")

        logger.info(
            f"Detection: {len(self._detections)}/{len(data_points)} images have valid detections"
        )

        if len(self._detections) < 3:
            logger.error(
                f"Only {len(self._detections)} valid detections (minimum 3 required). "
                f"Check board visibility and lighting."
            )
            return False

        return True

    def _run_intrinsic_calibration(self) -> bool:
        """Run intrinsic camera calibration from detections."""
        try:
            self._intrinsics = self._intrinsic_calibrator.calibrate(self._detections)

            if self._intrinsics is None or self._intrinsics.camera_matrix is None:
                logger.error("Intrinsic calibration returned no result")
                return False

            logger.info(
                f"Intrinsic calibration complete. "
                f"Reprojection error: {self._intrinsics.reprojection_error:.4f} px"
                if hasattr(self._intrinsics, 'reprojection_error')
                else "Intrinsic calibration complete."
            )
            return True

        except Exception as e:
            logger.error(f"Intrinsic calibration failed: {e}")
            return False

    def _run_pose_estimation(self) -> bool:
        """Estimate board poses from detections using intrinsics."""
        self._matched_poses = []
        self._valid_T_gripper2base = []
        self._valid_T_target2camera = []

        self._pose_estimator = PoseEstimator(
            self._detector,
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
        )

        for detection in self._detections:
            pose_result = self._pose_estimator.estimate(detection)
            matched_pose = MatchedDataPoint(detection=detection, estimated_pose=pose_result)
            self._matched_poses.append(matched_pose)

            if pose_result.success and pose_result.transform is not None:
                self._valid_T_gripper2base.append(detection.data_point.T_gripper2base)
                self._valid_T_target2camera.append(pose_result.transform)

        logger.info(
            f"Pose estimation: {len(self._valid_T_target2camera)}/{len(self._detections)} "
            f"poses estimated successfully"
        )

        if len(self._valid_T_target2camera) < 3:
            logger.error(
                f"Not enough valid poses for extrinsic calibration: "
                f"{len(self._valid_T_target2camera)} (minimum 3 required)"
            )
            return False

        return True

    def _run_extrinsic_hand_to_eye(self, method_name: str | None = None) -> bool:
        """Run hand-to-eye extrinsic calibration with one or all methods."""
        method_list = list(self._extrinsic_calibrator.METHOD_NAMES.keys())
        any_success = False

        if method_name is not None:
            # ── Single method ──────────────────────────────────────
            if method_name not in method_list:
                raise ValueError(
                    f"Unknown method '{method_name}'. "
                    f"Supported methods: {', '.join(sorted(method_list))}"
                )
            success = self._try_hand_to_eye_method(method_name)
            any_success = success
        else:
            # ── Try all methods ────────────────────────────────────
            for method in method_list:
                success = self._try_hand_to_eye_method(method)
                if success:
                    any_success = True

        if any_success:
            logger.info(
                f"Hand-to-eye calibration: "
                f"{len(self._extrinsics)} method(s) succeeded"
            )
        else:
            logger.error(
                "Hand-to-eye calibration: all methods failed. "
                "Capture more diverse poses with larger rotations."
            )

        return any_success

    def _run_extrinsic_hand_in_eye(self, method_name: str | None = None) -> bool:
        """Run hand-in-eye extrinsic calibration with one or all methods."""
        method_list = list(self._extrinsic_calibrator.METHOD_NAMES.keys())
        any_success = False

        if method_name is not None:
            # ── Single method ──────────────────────────────────────
            if method_name not in method_list:
                raise ValueError(
                    f"Unknown method '{method_name}'. "
                    f"Supported methods: {', '.join(sorted(method_list))}"
                )
            success = self._try_hand_in_eye_method(method_name)
            any_success = success
        else:
            # ── Try all methods ────────────────────────────────────
            for method in method_list:
                success = self._try_hand_in_eye_method(method)
                if success:
                    any_success = True

        if any_success:
            logger.info(
                f"Hand-in-eye calibration: "
                f"{len(self._extrinsics)} method(s) succeeded"
            )
        else:
            logger.error(
                "Hand-in-eye calibration: all methods failed. "
                "Capture more diverse poses with larger rotations."
            )

        return any_success

    # ═══════════════════════════════════════════════════════════════
    # Private Helper Methods
    # ═══════════════════════════════════════════════════════════════

    def _try_hand_to_eye_method(self, method_name: str) -> bool:
        """
        Try a single hand-to-eye calibration method.
        Catches OpenCV and other exceptions gracefully.

        Returns:
            True if method succeeded.
        """
        try:
            result = self._extrinsic_calibrator.calibrate_hand_to_eye(
                T_gripper2base=self._valid_T_gripper2base,
                T_target2camera=self._valid_T_target2camera,
                method_name=method_name,
            )

            if result.success and result.transform is not None:
                T_camera2base = result.transform

                # Validate the result
                if not self._validate_transform(T_camera2base, method_name):
                    return False

                logger.info(f"\n[{method_name}] Camera relative to Base Transform:")
                logger.info(f"\n{T_camera2base}")
                logger.info(f"\n{transform_to_pose_string(T_camera2base)}")
                self._extrinsics.append(result)
                return True
            else:
                logger.warning(f"Method '{method_name}' returned no valid result.")
                return False

        except cv2.error as e:
            logger.warning(f"OpenCV error with method '{method_name}': {e}")
            return False
        except Exception as e:
            logger.warning(f"Error with method '{method_name}': {e}")
            return False

    def _try_hand_in_eye_method(self, method_name: str) -> bool:
        """
        Try a single hand-in-eye calibration method.
        Catches OpenCV and other exceptions gracefully.

        Returns:
            True if method succeeded.
        """
        try:
            result = self._extrinsic_calibrator.calibrate_hand_in_eye(
                T_gripper2base=self._valid_T_gripper2base,
                T_target2camera=self._valid_T_target2camera,
                method_name=method_name,
            )

            if result.success and result.transform is not None:
                T_camera2gripper = result.transform

                # Validate the result
                if not self._validate_transform(T_camera2gripper, method_name):
                    return False

                logger.info(f"\n[{method_name}] Camera relative to Gripper Transform:")
                logger.info(f"\n{T_camera2gripper}")
                logger.info(f"\n{transform_to_pose_string(T_camera2gripper)}")
                self._extrinsics.append(result)
                return True
            else:
                logger.warning(f"Method '{method_name}' returned no valid result.")
                return False

        except cv2.error as e:
            logger.warning(f"OpenCV error with method '{method_name}': {e}")
            return False
        except Exception as e:
            logger.warning(f"Error with method '{method_name}': {e}")
            return False

    def _validate_transform(
        self, transform: np.ndarray, method_name: str
    ) -> bool:
        """
        Validate that a 4x4 transform matrix is physically plausible.

        Checks:
            - Rotation matrix has determinant ≈ 1
            - Translation is finite and within reasonable bounds
            - No NaN/Inf values
        """
        if transform.shape != (4, 4):
            logger.warning(f"[{method_name}] Invalid transform shape: {transform.shape}")
            return False

        if not np.all(np.isfinite(transform)):
            logger.warning(f"[{method_name}] Transform contains NaN/Inf values")
            return False

        R = transform[:3, :3]
        t = transform[:3, 3]

        # Check rotation determinant
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.01:
            logger.warning(
                f"[{method_name}] Invalid rotation matrix: det(R) = {det:.6f} (expected 1.0)"
            )
            return False

        # Check orthogonality: R^T * R should be identity
        orth_error = np.linalg.norm(R.T @ R - np.eye(3))
        if orth_error > 0.01:
            logger.warning(
                f"[{method_name}] Rotation not orthogonal: ||R^T*R - I|| = {orth_error:.6f}"
            )
            return False

        # Check translation magnitude (sanity: < 10m for typical robot setups)
        t_norm = np.linalg.norm(t)
        if t_norm > 10.0:
            logger.warning(
                f"[{method_name}] Unreasonable translation magnitude: {t_norm:.3f} m"
            )
            return False

        return True

    def _check_rotation_diversity(
        self,
        transforms: list[np.ndarray],
        min_angle_deg: float = 10.0,
        min_pairs_above_threshold: int = 1,
    ) -> bool:
        """
        Check if there is sufficient rotational diversity between poses.

        Args:
            transforms: List of 4x4 transform matrices (T_gripper2base).
            min_angle_deg: Minimum angular difference required.
            min_pairs_above_threshold: Minimum number of pairs that must exceed threshold.

        Returns:
            True if sufficient diversity exists.
        """
        if len(transforms) < 2:
            logger.error("Need at least 2 transforms to check rotation diversity")
            return False

        angles = []
        for i in range(len(transforms)):
            for j in range(i + 1, len(transforms)):
                R_i = transforms[i][:3, :3]
                R_j = transforms[j][:3, :3]

                # Relative rotation
                R_rel = R_i.T @ R_j

                # Angle from rotation matrix (handle numerical issues)
                trace_val = np.trace(R_rel)
                cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                angles.append(angle_deg)

        max_angle = max(angles) if angles else 0.0
        mean_angle = np.mean(angles) if angles else 0.0
        pairs_above = sum(1 for a in angles if a >= min_angle_deg)

        logger.info(
            f"Rotation diversity: max={max_angle:.1f}°, mean={mean_angle:.1f}°, "
            f"pairs above {min_angle_deg}°: {pairs_above}/{len(angles)}"
        )

        if max_angle < min_angle_deg:
            logger.error(
                f"Insufficient rotational diversity! "
                f"Maximum angle between poses: {max_angle:.1f}° "
                f"(minimum {min_angle_deg}° required). "
                f"Include poses with larger rotation differences."
            )
            return False

        if pairs_above < min_pairs_above_threshold:
            logger.error(
                f"Not enough diverse pose pairs: {pairs_above} "
                f"(need at least {min_pairs_above_threshold}). "
                f"Add more varied orientations."
            )
            return False

        return True