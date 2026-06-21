import numpy as np
import cv2
import open3d as o3d
import os
from tqdm import tqdm
import torch
import pytorch3d.ops as torch3d_ops
from data.record_utils.camera_info import camera_intrinsics, camera_extrinsics, workspace

'''
This script is used to generate point clouds from the raw data.
''' 
from scipy.interpolate import griddata

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

def farthest_point_sampling(points, num_points=4096, use_cuda=True):
    """Perform farthest point sampling on point cloud"""
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def transform_point_cloud(pcd, extrinsic_matrix):
    """Transform point cloud using inverse of extrinsic matrix"""

    pcd.transform(extrinsic_matrix)

    R_fix = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi)) 
    pcd.rotate(R_fix, center=(0, 0, 0))

    return pcd

def fill_missing_values(depth_image):
    """Fill missing values in depth image using nearest neighbor interpolation"""
    h, w = depth_image.shape
    y, x = np.mgrid[0:h, 0:w]
    valid_mask = depth_image > 0
    points = np.array((x[valid_mask], y[valid_mask])).T
    values = depth_image[valid_mask]
    filled_depth = griddata(points, values, (x, y), method='nearest')
    return filled_depth

def repair_depth_image(depth_image):
    """Repair depth image using bilateral filtering for interpolation"""
    # Fill missing values or noise (assuming areas with depth value 0 need repair)
    mask = (depth_image == 0)
    if np.any(mask):
        # Use neighborhood interpolation smoothing
        depth_image[mask] = np.median(depth_image[~mask])  # First use median to fill missing values
    repaired_depth = cv2.bilateralFilter(depth_image.astype(np.float32), 5, 50, 50)

    return repaired_depth

def filter_zero_values(depth_image):
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

def remove_flying_pixels(depth_image, gradient_threshold=100.0, dilate_size=5):

    depth_float = depth_image.astype(np.float32)
    
    grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    edge_mask = (gradient > gradient_threshold).astype(np.uint8)
    
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
    
    result = depth_image.copy()
    result[edge_mask == 1] = 0.0
    
    return result


def process_episode(episode_dir, camera_intrinsics, extrinsic_matrix):
    """Process data for a single episode"""
    img_dir = os.path.join(episode_dir, "img")
    depth_dir = os.path.join(episode_dir, "depth")
    point_cloud_dir = os.path.join(episode_dir, "point_cloud")
    os.makedirs(point_cloud_dir, exist_ok=True)

    frames = sorted([f[6:-4] for f in os.listdir(img_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    for frame_idx in tqdm(frames, desc=f"Processing {os.path.basename(episode_dir)}"):
        color_path = os.path.join(img_dir, f"frame_{frame_idx}.png")
        depth_path = os.path.join(depth_dir, f"frame_{frame_idx}.npy")

        # Read image and depth data
        color_image = cv2.imread(color_path)
        depth_image = np.load(depth_path)

        # depth_image = fill_missing_values(depth_image)
        # depth_image = repair_depth_image(depth_image)
        depth_image = remove_flying_pixels(depth_image, gradient_threshold=100.0, dilate_size=5)
        depth_image = filter_zero_values(depth_image)


        # Convert to Open3D format
        o3d_color = o3d.geometry.Image(color_image)
        o3d_depth = o3d.geometry.Image(depth_image.astype(np.float32))

        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, 
            o3d_depth,
            depth_scale=1000.0, 
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            camera_intrinsics
        )
        
        # Apply extrinsic matrix transformation
        transformed_cloud = transform_point_cloud(point_cloud, extrinsic_matrix)


        # Filter point cloud
        points = np.asarray(transformed_cloud.points)
        mask = (
            (points[:, 0] >= workspace['x'][0]) & (points[:, 0] <= workspace['x'][1]) &
            (points[:, 1] >= workspace['y'][0]) & (points[:, 1] <= workspace['y'][1]) &
            (points[:, 2] >= workspace['z'][0]) & (points[:, 2] <= workspace['z'][1])
        )
        filtered_points = points[mask]

        # Add farthest point sampling
        if len(filtered_points) > 0:
            sampled_points, _ = farthest_point_sampling(filtered_points, num_points=1024)
            filtered_points = sampled_points
            
            # Save point cloud
            pcd_path = os.path.join(point_cloud_dir, f"frame_{frame_idx}")

            np.save(f"{pcd_path}.npy", filtered_points)

def process_all_data(data_root, camera_intrinsics, extrinsic_matrix):
    """Process data for all episodes"""
    episodes = sorted([d for d in os.listdir(data_root) if d.startswith('episode_')])
    total_episodes = len(episodes)
    print(f"\nFound {total_episodes} episodes")
    
    for episode_idx, episode in enumerate(episodes, 1):
        episode_dir = os.path.join(data_root, episode)
        if os.path.isdir(episode_dir):
            print(f"\nProcessing {episode} ({episode_idx}/{total_episodes})...")
            process_episode(episode_dir, camera_intrinsics, extrinsic_matrix)

def main():
    data_root = f"/ros2_ws/data/episodes/{FOLDER_NAME}"

    # Process all data
    process_all_data(data_root=data_root, 
                     camera_intrinsics=camera_intrinsics, 
                     extrinsic_matrix=camera_extrinsics)

if __name__ == "__main__":
    main()
