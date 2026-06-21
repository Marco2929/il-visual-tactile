"""Transformation matrix utilities."""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def make_transform_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec.flatten())[0]
    T[:3, 3] = tvec.flatten()
    return T

def make_transform_matrix_from_quat(quat: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transformation matrix from quaternion and position."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos.flatten()
    return T

def decompose_transform_matrix(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose 4x4 matrix into (rotation_matrix, rvec, tvec)."""
    R_mat = transform[:3, :3]
    rvec = cv2.Rodrigues(R_mat)[0]
    tvec = transform[:3, 3]
    return R_mat, rvec, tvec


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 transformation matrix."""
    return np.linalg.inv(transform)


def average_transformations(transforms: list[np.ndarray]) -> np.ndarray:
    """
    Average multiple transformation matrices using quaternion interpolation.
    
    Args:
        transforms: List of 4x4 transformation matrices
        
    Returns:
        4x4 averaged transformation matrix
    """
    if not transforms:
        return np.eye(4)
    
    if len(transforms) == 1:
        return transforms[0].copy()
    
    # Extract rotations and translations
    translations = []
    quaternions = []
    
    for T in transforms:
        # Extract translation (easy)
        t = T[:3, 3]
        translations.append(t)
        
        # Extract rotation as quaternion
        R_mat = T[:3, :3]
        rotation = R.from_matrix(R_mat)
        q = rotation.as_quat()  # [x, y, z, w]
        quaternions.append(q)
    
    # Average translations (simple arithmetic mean)
    t_avg = np.mean(translations, axis=0)
    
    # Average quaternions (more complex)
    quaternions = np.array(quaternions)
    
    # Ensure all quaternions are in same hemisphere (avoid 180° flips)
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[0], quaternions[i]) < 0:
            quaternions[i] *= -1
    
    # Compute average quaternion
    q_avg = np.mean(quaternions, axis=0)
    
    # Normalize quaternion to ensure it's valid
    q_norm = np.linalg.norm(q_avg)
    if q_norm > 1e-8:
        q_avg = q_avg / q_norm
    else:
        # Fallback to identity quaternion if norm is too small
        q_avg = np.array([0.0, 0.0, 0.0, 1.0])
    
    # Ensure quaternion is real (not complex)
    q_avg = np.real(q_avg).astype(np.float64)
    
    # Convert back to rotation matrix
    try:
        R_avg = R.from_quat(q_avg).as_matrix()
    except ValueError as e:
        # If quaternion is still invalid, use identity rotation
        print(f"Warning: Invalid quaternion {q_avg}, using identity rotation. Error: {e}")
        R_avg = np.eye(3)
    
    # Construct averaged transformation matrix
    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    
    return T_avg


def transform_to_pose_string(transform: np.ndarray, precision: int = 4) -> str:
    """Convert transformation matrix to readable string."""
    t = transform[:3, 3]
    # FIXME Check if possible
    euler = R.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    return (
        f"Position: [{t[0]:.{precision}f}, {t[1]:.{precision}f}, {t[2]:.{precision}f}] m\n"
        f"Rotation: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg (XYZ)"
    )