import os
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

from data.record_utils.camera_info import camera_intrinsics, camera_extrinsics, workspace


# Cache mit GRÖSSERER Punktzahl – downsampling kommt später
CACHE_NUM_POINTS = 4096


def generate_point_cloud_from_depth(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    extrinsic_matrix: np.ndarray,
    workspace_bounds: dict,
    num_points: int = CACHE_NUM_POINTS,
) -> np.ndarray:
    """
    Depth + Color → filtered, FPS-sampled point cloud.
    Returns: (num_points, 3) float32
    """
    from .point_cloud_utils import (
        remove_flying_pixels, filter_zero_values,
        transform_point_cloud, farthest_point_sampling,
    )

    # Depth cleanup
    depth_image = remove_flying_pixels(depth_image, gradient_threshold=100.0, dilate_size=5)
    depth_image = filter_zero_values(depth_image)

    # RGBD → Point Cloud
    o3d_color = o3d.geometry.Image(color_image)
    o3d_depth = o3d.geometry.Image(depth_image.astype(np.float32))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=1000.0, depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd = transform_point_cloud(pcd, extrinsic_matrix)

    # Workspace filter
    points = np.asarray(pcd.points)
    mask = (
        (points[:, 0] >= workspace_bounds['x'][0]) & (points[:, 0] <= workspace_bounds['x'][1]) &
        (points[:, 1] >= workspace_bounds['y'][0]) & (points[:, 1] <= workspace_bounds['y'][1]) &
        (points[:, 2] >= workspace_bounds['z'][0]) & (points[:, 2] <= workspace_bounds['z'][1])
    )
    filtered = points[mask]

    if len(filtered) < num_points:
        # Pad with duplicates if too few points
        indices = np.random.choice(len(filtered), num_points, replace=True)
        filtered = filtered[indices]

    sampled, _ = farthest_point_sampling(filtered, num_points=num_points)
    return sampled.astype(np.float32)


def generate_for_episode(episode_dir: Path, camera: str = "front"):
    """Generate and cache point clouds for one episode."""
    if camera == "front":
        img_dir = episode_dir / "img"
        depth_dir = episode_dir / "depth"
        output_dir = episode_dir / "point_cloud"
    else:
        img_dir = episode_dir / "wrist_img"
        depth_dir = episode_dir / "wrist_depth"
        output_dir = episode_dir / "wrist_point_cloud"

    output_dir.mkdir(exist_ok=True)

    frames = sorted(img_dir.glob("frame_*.png"))

    for frame_path in tqdm(frames, desc=f"{episode_dir.name}/{camera}"):
        frame_id = frame_path.stem  # "frame_0001"
        depth_path = depth_dir / f"{frame_id}.npy"
        out_path = output_dir / f"{frame_id}.npy"

        if out_path.exists():
            continue  # Skip already generated

        color = cv2.imread(str(frame_path))
        depth = np.load(str(depth_path))

        pc = generate_point_cloud_from_depth(
            color, depth, camera_intrinsics, camera_extrinsics, workspace,
            num_points=CACHE_NUM_POINTS,
        )
        np.save(str(out_path), pc)