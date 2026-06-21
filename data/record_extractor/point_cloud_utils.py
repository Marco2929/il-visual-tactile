# scripts/point_cloud_utils.py
"""Point cloud generation from depth + color."""

import cv2
import torch
import numpy as np
import open3d as o3d
from pytorch3d import ops as torch3d_ops


def generate_point_cloud_from_depth(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    extrinsic_matrix: np.ndarray,
    workspace_bounds: dict | None,
    num_points: int = 4096,
    include_color: bool = True,
) -> np.ndarray:
    """
    Depth + Color → filtered, FPS-sampled point cloud.

    Args:
        color_image: (H, W, 3) uint8 BGR
        depth_image: (H, W) float32, in mm
        intrinsics: Open3D camera intrinsics
        extrinsic_matrix: (4, 4) camera-to-world transform
        workspace_bounds: {"x": [min, max], "y": [...], "z": [...]}
        num_points: number of output points
        include_color: If True, returns (num_points, 6) with RGB. If False, returns (num_points, 3).

    Returns: float32 numpy array
    """
    # Depth cleanup
    depth_image = _remove_flying_pixels(depth_image, gradient_threshold=100.0, dilate_size=5)
    depth_image = _filter_zero_values(depth_image)

    # BGR → RGB for Open3D
    color_rgb = color_image[:, :, ::-1].copy()

    # RGBD → Point Cloud
    o3d_color = o3d.geometry.Image(color_rgb.astype(np.uint8))
    o3d_depth = o3d.geometry.Image(depth_image.astype(np.float32))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=1000.0, depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    # Transform to world frame
    pcd.transform(extrinsic_matrix)

    R_fix = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
    pcd.rotate(R_fix, center=(0, 0, 0))

    # Extract XYZ and RGB
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # Values are in [0, 1]

    # Handle empty point cloud early
    if len(points) == 0:
        dim = 6 if include_color else 3
        return np.zeros((num_points, dim), dtype=np.float32)

    # Concatenate features based on the flag
    if include_color:
        features = np.concatenate([points, colors], axis=-1)
    else:
        features = points

    # Workspace filter (Always evaluates indices 0, 1, 2 for XYZ)
    if workspace_bounds:
        mask = (
            (features[:, 0] >= workspace_bounds['x'][0]) & (features[:, 0] <= workspace_bounds['x'][1]) &
            (features[:, 1] >= workspace_bounds['y'][0]) & (features[:, 1] <= workspace_bounds['y'][1]) &
            (features[:, 2] >= workspace_bounds['z'][0]) & (features[:, 2] <= workspace_bounds['z'][1])
        )
        features = features[mask]

    if len(features) == 0:
        dim = 6 if include_color else 3
        return np.zeros((num_points, dim), dtype=np.float32)

    # Pad if too few points
    if len(features) < num_points:
        random_indices = np.random.choice(len(features), num_points, replace=True)
        features = features[random_indices]

    # Farthest point sampling (Calculate distance ONLY on XYZ)
    spatial_points = features[:, :3]
    _, fps_indices = _farthest_point_sampling(spatial_points, num_points)

    # Slice the final features using the sampled indices
    sampled = features[fps_indices]

    return sampled.astype(np.float32)


def _remove_flying_pixels(depth_image, gradient_threshold=100.0, dilate_size=5):
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


def _filter_zero_values(depth_image):
    """Expand invalid regions and apply bilateral smoothing."""
    base_invalid_mask = ((depth_image <= 0) | ~np.isfinite(depth_image)).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    expanded_invalid_mask = cv2.dilate(base_invalid_mask, kernel, iterations=1)

    depth_image = depth_image.copy()
    depth_image[expanded_invalid_mask == 1] = 6000.0

    return cv2.bilateralFilter(depth_image, d=5, sigmaColor=50, sigmaSpace=50)


def _farthest_point_sampling(points, num_points=1024, use_cuda=True):
    """Farthest point sampling using PyTorch3D."""
    K = [num_points]
    if use_cuda and torch.cuda.is_available():
        pts = torch.from_numpy(points).float().cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=pts.unsqueeze(0), K=K
        )
        sampled_points = sampled_points.squeeze(0).cpu().numpy()
        indices = indices.squeeze(0).cpu().numpy()
    else:
        pts = torch.from_numpy(points).float()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=pts.unsqueeze(0), K=K
        )
        sampled_points = sampled_points.squeeze(0).numpy()
        indices = indices.squeeze(0).numpy()

    return sampled_points, indices