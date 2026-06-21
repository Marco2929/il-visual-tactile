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

URDF_PATH = os.environ.get("FOLDER_NAME", "/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf")
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
    
    def process_rgbd(self, color_img, depth_img):
        """Process RGB-D image to generate point cloud"""
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
        transformed_points = pcd.transform(self.extrinsic_matrix)
        
        points = np.asarray(transformed_points.points)
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
            sampled_points, _ = torch3d_ops.sample_farthest_points(
                points=filtered_points.unsqueeze(0), 
                K=[1024]
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

