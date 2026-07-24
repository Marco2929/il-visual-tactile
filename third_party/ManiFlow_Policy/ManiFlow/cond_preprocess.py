import os

import open3d as o3d
import kinpy as kp
import pytorch3d.ops as torch3d_ops
import torch
import cv2
import numpy as np

import sys
sys.path.insert(0, "/ros2_ws/data/record_utils")

from camera_info import camera_intrinsics, camera_extrinsics, workspace

URDF_PATH = os.environ.get("URDF_PATH", "/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf")
ROBOT_TIP_FRAME = os.environ.get("ROBOT_TIP_FRAME", "right_inspire_base_link")    

class PointCloudProcessor:
    """Point Cloud Processor"""
    def __init__(self):
        # Camera intrinsics
        self.camera_intrinsics = camera_intrinsics
        
        # Extrinsic matrix
        self.extrinsic_matrix = camera_extrinsics
        
        # Workspace range
        self.WORK_SPACE = workspace

    def filter_zero_values(self, depth_image):
        # 1. Initial mask of strictly zero, negative, or invalid pixels
        base_invalid_mask = ((depth_image <= 0) | ~np.isfinite(depth_image)).astype(np.uint8)

        # 2. Dilate the mask to engulf surrounding low-confidence pixels
        # A 5x5 kernel expands the boundary by roughly 2 pixels in all directions.
        # Increase the kernel size (e.g., (7, 7) or (9, 9)) if the spikes are wider.
        kernel = np.ones((7, 7), np.uint8)
        expanded_invalid_mask = cv2.dilate(base_invalid_mask, kernel, iterations=1)

        # 3. Hard-replace the entire expanded region with 600
        depth_image[expanded_invalid_mask == 1] = 6000.0

        return cv2.bilateralFilter(depth_image, d=5, sigmaColor=50, sigmaSpace=50)
    
    def _remove_flying_pixels(self, depth_image, gradient_threshold=100.0, dilate_size=5):
        """Remove flying pixels at depth discontinuities using Sobel gradients."""
        depth_float = depth_image.astype(np.float32)

        grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)

        edge_mask = (gradient > gradient_threshold).astype(np.uint8)

        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

        result = depth_image.copy()
        result[edge_mask == 1] = 0.0

        return result

    
    def process_rgbd(self, color_img, depth_img, num_points=1024):
        """Process RGB-D image to generate point cloud"""
        depth_img = self._remove_flying_pixels(depth_img, gradient_threshold=100.0, dilate_size=5)
        depth_img = self.filter_zero_values(depth_img)

        # Convert to Open3D format
        o3d_color = o3d.geometry.Image(color_img)
        o3d_depth = o3d.geometry.Image(depth_img)
        
        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, 
            o3d_depth,
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            self.camera_intrinsics
        )
        
        # Apply extrinsic transformation
        pcd.transform(self.extrinsic_matrix)

        # 180° rotation around Z-axis
        R_fix = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
        pcd.rotate(R_fix, center=(0, 0, 0))
        
        points = np.asarray(pcd.points)

        # Workspace filtering
        mask = (
            (points[:, 0] >= self.WORK_SPACE['x'][0]) & 
            (points[:, 0] <= self.WORK_SPACE['x'][1]) &
            (points[:, 1] >= self.WORK_SPACE['y'][0]) & 
            (points[:, 1] <= self.WORK_SPACE['y'][1]) &
            (points[:, 2] >= self.WORK_SPACE['z'][0]) & 
            (points[:, 2] <= self.WORK_SPACE['z'][1])
        )
        filtered_points = points[mask]

        # Farthest point sampling
        if len(filtered_points) > 0:
            filtered_points = torch.from_numpy(filtered_points).cuda()

            # Cap FPS input to a fixed size → deterministic FPS runtime.
            # Random pre-subsample keeps spatial coverage while removing the
            # data-dependent latency spikes (100ms↔1000ms) from variable N.
            MAX_FPS_INPUT = 8192
            n = filtered_points.shape[0]
            if n > MAX_FPS_INPUT:
                idx = torch.randperm(n, device=filtered_points.device)[:MAX_FPS_INPUT]
                filtered_points = filtered_points[idx]

            sampled_points, _ = torch3d_ops.sample_farthest_points(
                points=filtered_points.unsqueeze(0),
                K=[num_points]
            )
            sampled_points = sampled_points.squeeze(0).cpu().numpy()
            return sampled_points
        return None
    
def setup_kinematics():
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found at {URDF_PATH}")
    with open(URDF_PATH, 'r') as f:
        chain = kp.build_serial_chain_from_urdf(f.read(), end_link_name=ROBOT_TIP_FRAME, root_link_name="base_link")
    return chain

def matrix_to_6d(matrix):
    """
    Converts a 3x3 rotation matrix to 6D representation.
    Takes the first two columns (r1, r2).
    Input: [3, 3]
    Output: [6]
    """
    return matrix[:, :2].flatten(order='F')

def compute_fk(chain, full_state):
    """ Converts 12-dim Joint State -> 13-dim Cartesian State and extract success state"""
    ur_joints = full_state[:6]
    hand_joints = full_state[6:]
    
    # Ensure these names match your URDF
    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    
    th = dict(zip(joint_names, ur_joints))
    ret = chain.forward_kinematics(th)
    
    # DP3 expects [x, y, z, qx, qy, qz, qw]
    # Kinpy gives [w, x, y, z], so we reorder to [x, y, z, w]
    rot_xyzw = np.array([ret.rot[1], ret.rot[2], ret.rot[3], ret.rot[0]])
    
    return np.concatenate([ret.pos, rot_xyzw, hand_joints])

# palm_transform.py — wird von DataLoader UND Inference genutzt

import numpy as np
import kinpy as kp


class PalmFrameTransform:
    """Reusable: works in DataLoader AND live inference."""

    def __init__(self, urdf_path):
        with open(urdf_path, 'r') as f:
            urdf_data = f.read()
        self.chain = kp.build_chain_from_urdf(urdf_data)
        self.joint_names = self.chain.get_joint_parameter_names()
        self.tip_links = [f'rl_dg_{i}_4' for i in range(1, 6)]
        self.palm_link = 'tool0'

    def __call__(self, joints_deg: np.ndarray, force_torque: np.ndarray) -> np.ndarray:
        """
        Args:
            joints_deg: (20,) joint angles in degrees
            force_torque: (30,) raw F/T values (5×6, tip-local)

        Returns:
            (5, 13) palm-frame observation
        """
        joints_rad = np.deg2rad(joints_deg)
        joint_dict = {name: joints_rad[i] for i, name in enumerate(self.joint_names)}

        fk_dict = self.chain.forward_kinematics(joint_dict)

        T_palm = fk_dict[self.palm_link]
        R_palm = self._quat_to_matrix(T_palm.rot)
        p_palm = np.array(T_palm.pos)

        ft = force_torque.reshape(5, 6)
        fingers = []

        for i, tip_link in enumerate(self.tip_links):
            T_tip = fk_dict[tip_link]
            R_tip = self._quat_to_matrix(T_tip.rot)
            p_tip = np.array(T_tip.pos)

            tip_in_palm = R_palm.T @ (p_tip - p_palm)
            f_palm = R_palm.T @ (R_tip @ ft[i, 0:3])
            tau_palm = R_palm.T @ (R_tip @ ft[i, 3:6]) + np.cross(tip_in_palm, f_palm)
            finger_joints = joints_rad[i*4:(i+1)*4]

            fingers.append(np.concatenate([tip_in_palm, f_palm, tau_palm, finger_joints]))

        return np.stack(fingers)  # (5, 13)

    def _quat_to_matrix(self, q):
        q = np.array(q)
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])