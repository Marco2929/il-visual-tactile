"""Hand-eye calibration."""

import cv2
import numpy as np
from scipy.optimize import least_squares
import logging

from .data_classes import PoseResult, ExtrinsicCalibrationResult
from .transforms import average_transformations
import os
from datetime import datetime
import yaml
import json

logger = logging.getLogger(__name__)


class ExtrinsicCalibrator:
    """Performs hand-eye calibration.
    Notation:
    T_a2b: a relative to b transform
    gripper: robot end-effector
    flange: robot end-effector (same as gripper here)
    base: robot base
    camera: camera frame
    board: calibration board (e.g. ChArUco board)
    target: calibration target (same as board here)
    """

    METHOD_NAMES = {
        "AXXB_TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "AXXB_PARK": cv2.CALIB_HAND_EYE_PARK,
        "AXXB_HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "AXXB_ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "AXXB_DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
        "AXZB_SHAH": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        "AXZB_LI": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
        "INTERNAL_LEAST_SQUARES": "INTERNAL_LEAST_SQUARES",
        "INTERNAL_SHAH": "INTERNAL_SHAH"
    }

    def get_method_names(self):
        """Returns all method names as a list of strings."""
        return list(self.METHOD_NAMES.values())


    def calibrate_hand_to_eye(
        self, 
        T_gripper2base: list[np.ndarray], 
        T_target2camera: list[np.ndarray], 
        method_name: int | str = "INTERNAL_SHAH",
        T_target2gripper: np.ndarray | None = None,
    ) -> ExtrinsicCalibrationResult:
        """
        This wrapper function calls the specified method for hand to eye calibration.
        Either self implemented methods or opencv methods using AX=XB or AX=ZB formulation.
        
        Args:
            T_gripper2base: List of gripper relative to base transforms (4x4)
            T_target2camera: List of target relative to camera transforms (4x4)
            method: Method to use for calibration
            T_target2gripper: Optional target relative to gripper transform (4x4)

        Returns:
            ExtrinsicCalibrationResult (base-to-camera transform)
        """

        logger.info(f"Starting hand-to-eye calibration")
        # get the method name
        method = self.METHOD_NAMES.get(method_name, "UNKNOWN")
        
        if method_name.startswith("INTERNAL"):
            logger.info(f"Using internal extrinsic calibration method: {method_name}")
            if method_name == "INTERNAL_LEAST_SQUARES":
                T_base2gripper = [np.linalg.inv(g2b) for g2b in T_gripper2base]
                T_camera2base = self._calibrate_least_squares(T_base2gripper, T_target2camera, T_target2gripper)
            elif method_name == "INTERNAL_SHAH":
                # Output X is T_target2gripper transform. Can be extracted if needed.
                _, T_camera2base = self._calibrate_SHAH(T_gripper2base, T_target2camera)
        elif method_name.startswith("AXXB"):
            # With a change of the inputs T_target2gripper can be calculated
            logger.info(f"Using OpenCV extrinsic calibration method: {method_name}")
            T_base2gripper = [np.linalg.inv(g2b) for g2b in T_gripper2base]
            T_camera2base = self._calibrate_AX_XB(T_base2gripper, T_target2camera, method=method)
        elif method_name.startswith("AXZB"):
            # Output X is T_target2gripper transform. Can be extracted if needed.
            logger.info(f"Using OpenCV extrinsic calibration method: {method_name}")
            _, T_camera2base = self._calibrate_AX_ZB(T_gripper2base, T_target2camera, method=method)
        else:
            logger.error(f"Extrinsic calibration method not implemented: {method_name}")
            return ExtrinsicCalibrationResult(success=False, num_poses_used=0, method=method_name)

        return ExtrinsicCalibrationResult(
            success=T_camera2base is not None,
            transform=T_camera2base,
            rotation=T_camera2base[:3, :3],
            translation=T_camera2base[:3, 3].reshape(3, 1),
            num_poses_used=len(T_target2camera),
            method=method_name,
        )
    
    def calibrate_hand_in_eye(
        self, 
        T_gripper2base: list[np.ndarray], 
        T_target2camera: list[np.ndarray], 
        method_name: int | str = "INTERNAL_SHAH"
    ) -> ExtrinsicCalibrationResult:
        """
        This wrapper function calls the specified method for hand in eye calibration.
        Either self implemented methods or opencv methods using AX=XB or AX=ZB formulation.
        
        Args:
            T_gripper2base: List of gripper relative to base transforms (4x4)
            T_target2camera: List of target relative to camera transforms (4x4)
            method: Method to use for calibration

        Returns:
            ExtrinsicCalibrationResult (base-to-camera transform)
        """

        logger.info(f"Starting hand-in-eye calibration")
        # get the method name
        method = self.METHOD_NAMES.get(method_name, "UNKNOWN")
        
        if method_name.startswith("INTERNAL"):
            logger.info(f"Using internal extrinsic calibration method: {method_name}")
            if method_name == "INTERNAL_SHAH":
                T_base2gripper = [np.linalg.inv(g2b) for g2b in T_gripper2base]
                _, T_camera2gripper = self._calibrate_SHAH(T_base2gripper, T_target2camera)
            else:
                logger.error(f"Internal method not implemented for hand-in-eye calibration: {method_name}")
                return ExtrinsicCalibrationResult(success=False, num_poses_used=0, method=method_name)
        elif method_name.startswith("AXXB"):
            logger.info(f"Using OpenCV extrinsic calibration method: {method_name}")
            T_camera2gripper = self._calibrate_AX_XB(T_gripper2base, T_target2camera, method=method)
        elif method_name.startswith("AXZB"):
            logger.info(f"Using OpenCV extrinsic calibration method: {method_name}")
            T_base2gripper = [np.linalg.inv(g2b) for g2b in T_gripper2base]
            _, T_camera2gripper = self._calibrate_AX_ZB(T_base2gripper, T_target2camera, method=method)
        else:
            logger.error(f"Extrinsic calibration method not implemented: {method_name}")
            return ExtrinsicCalibrationResult(success=False, num_poses_used=0, method=method_name)

        return ExtrinsicCalibrationResult(
            success=T_camera2gripper is not None,
            transform=T_camera2gripper,
            rotation=T_camera2gripper[:3, :3],
            translation=T_camera2gripper[:3, 3].reshape(3, 1),
            num_poses_used=len(T_target2camera),
            method=method_name,
        )

    def _calibrate_AX_XB(
        self, 
        A: list[np.ndarray], 
        B: list[np.ndarray], 
        method=cv2.CALIB_HAND_EYE_TSAI
    ) -> np.ndarray:
        """
        Extrinsic calibration using AX=XB formulation.
        For hand-to-eye calibration:
        A: T_base2gripper transforms: base relative to gripper
        B: T_target2camera transforms: target relative to camera
        X: T_camera2base transform: camera relative to base (← das wollen wir)

        For hand-in-eye calibration:
        A: T_gripper2base transforms: gripper relative to base
        B: T_target2camera transforms: target relative to camera
        X: T_camera2gripper transform: camera relative to gripper (← das wollen wir)

        The correct form is actually A_i X = X B_i
        where A_i = base-to-gripper_(i+1) ^-1 * base-to-gripper_i
        and B_i = target-to-camera_(i+1) ^-1 * target-to-camera_i

        Similarly for hand-in-eye calibration.

        The algorithm uses the relative motions between poses to solve for X.
        
        Args:
            A: List of transforms (4x4)
            B: List of transforms (4x4)
            
        Returns:
            X: 4x4 homogeneous transformation
        """
        R_A = [a[:3, :3] for a in A]
        t_A = [a[:3, 3].reshape(3, 1) for a in A]
        R_B = [b[:3, :3] for b in B]
        t_B = [b[:3, 3].reshape(3, 1) for b in B]
        R_X, t_X = cv2.calibrateHandEye(R_A, t_A, R_B, t_B, method=method)

        X = np.eye(4)
        X[:3, :3] = R_X
        X[:3, 3] = t_X.flatten()

        return X

    def _calibrate_AX_ZB(
        self, 
        A: list[np.ndarray], 
        B: list[np.ndarray], 
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrinsic calibration using AX=ZB formulation.

        For hand-to-eye calibration:
        A: T_gripper2base
        B: T_target2camera
        X: T_target2gripper transform (Board am Flansch)
        Z: T_camera2base transform (← das wollen wir)

        For hand-in-eye calibration:
        A: T_base2gripper
        B: T_target2camera
        X: T_target2base transform
        Z: T_camera2gripper transform (← das wollen wir)

        Args:
            A: List of transforms (4x4)
            B: List of transforms (4x4)

        Returns:
            X: 4x4 homogeneous transformation
            Z: 4x4 homogeneous transformation
        """
        R_A = [a[:3, :3] for a in A]
        t_A = [a[:3, 3].reshape(3, 1) for a in A]
        R_B = [b[:3, :3] for b in B]
        t_B = [b[:3, 3].reshape(3, 1) for b in B]
        R_X, t_X, R_Z, t_Z = cv2.calibrateRobotWorldHandEye(R_A, t_A, R_B, t_B, method=method)

        X = np.eye(4)
        X[:3, :3] = R_X
        X[:3, 3] = t_X.flatten()

        Z = np.eye(4)
        Z[:3, :3] = R_Z
        Z[:3, 3] = t_Z.flatten()

        return X, Z
    
    def _calibrate_least_squares(
        self,
        T_gripper_base: list[np.ndarray], 
        T_camera_target: list[np.ndarray], 
        T_gripper_target: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extrinsic calibration via least squares.
        For hand-to-eye calibration (hand in eye not yet implemented):

        Args:
            T_gripper_base: List of base-to-gripper transforms (4x4)
            T_camera_target: List of target-to-camera transforms (4x4)
            T_gripper_target: Optional target-to-gripper transform (4x4)

        Returns:
            X: 4x4 homogeneous transformation
        """
        if T_gripper_target is None:
            T_gripper_target = np.eye(4)
            logger.warning("T_gripper_target not provided, using identity matrix as initial estimate. Results may be inaccurate.")

        T_target_gripper = np.linalg.inv(T_gripper_target)
        
        T_base_camera = []
        for T_g_b, T_c_t in zip(T_gripper_base, T_camera_target):
            T_b_g = np.linalg.inv(T_g_b)
            T_t_c = np.linalg.inv(T_c_t)
            T_b_c = T_b_g @ T_gripper_target @ T_t_c
            T_base_camera.append(T_b_c)
        T_init_base_camera = average_transformations(T_base_camera)
        init_params = np.zeros(6)
        init_params[3:] = cv2.Rodrigues(T_init_base_camera[:3, :3])[0].flatten()
        init_params[:3] = T_init_base_camera[:3, 3]
        
        def objective(params):
            T_base_camera = np.eye(4)
            T_base_camera[:3, :3] = cv2.Rodrigues(params[:3])[0]
            T_base_camera[:3, 3] = params[3:]
            
            errors = []
            sum_errors = 0.0
            for T_c_t, T_g_b in zip(T_camera_target, T_gripper_base):
                result = T_c_t @ T_target_gripper @ T_g_b @ T_base_camera
                translation_error = np.linalg.norm(result[:3, 3])
                rotation_error = np.linalg.norm(cv2.Rodrigues(result[:3, :3])[0])
                errors.append(translation_error)
                errors.append(rotation_error)
                sum_errors += translation_error + rotation_error
            # return sum of residuals as a single-element array for least_squares
            return sum_errors
        
        opt = least_squares(objective, init_params) #, method="lm")
        
        T_result = np.eye(4)
        T_result[:3, :3] = cv2.Rodrigues(opt.x[:3])[0]
        T_result[:3, 3] = opt.x[3:]
        
        return T_result


    def _calibrate_SHAH(
            self, 
            A: list[np.ndarray],
            B: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrinsic calibration using AX=ZB formulation.

        For hand-to-eye calibration:
        A: T_gripper2base
        B: T_target2camera
        X: T_target2gripper transform (Board am Flansch)
        Z: T_camera2base transform (← das wollen wir)

        For hand-in-eye calibration:
        A: T_base2gripper
        B: T_target2camera
        X: T_target2base transform
        Z: T_camera2gripper transform (← das wollen wir)

        Args:
            A: List of transforms (4x4)
            B: List of transforms (4x4)

        Returns:
            X: 4x4 homogeneous transformation
            Z: 4x4 homogeneous transformation
        """

        r = range(len(A))

        K = np.sum([np.kron(B[i][:3,:3], A[i][:3,:3]) for i in r], axis=0)
        U, s, Vt = np.linalg.svd(K)

        # rotation (theorem 2.4)
        Vx = np.reshape(Vt[0,:], (3,3)).T
        Vz = np.reshape(U[:,0], (3,3)).T
        detVx, detVz = np.linalg.det(Vx), np.linalg.det(Vz)
        Rx = np.sign(detVx)*np.abs(detVx)**(-1/3) * Vx
        Rz = np.sign(detVz)*np.abs(detVz)**(-1/3) * Vz

        U, s, Vt = np.linalg.svd(Rx); Rx = np.dot(U, Vt)
        U, s, Vt = np.linalg.svd(Rz); Rz = np.dot(U, Vt)

        # translation
        I = np.eye(3)
        At = np.concatenate([np.concatenate([I,-A[i][:3,:3]], axis=1) for i in r], axis=0)
        bt = np.concatenate([A[i][:3,3] - np.dot(Rz, B[i][:3,3]) for i in r], axis=0)
        t = np.dot(np.linalg.pinv(At), bt)
        tx, tz = t[3:], t[:3]

        X = np.vstack([np.hstack([Rx, tx[:,None]]), [0,0,0,1]])
        Z = np.vstack([np.hstack([Rz, tz[:,None]]), [0,0,0,1]])

        return X, Z


