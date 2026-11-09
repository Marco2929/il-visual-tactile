import open3d as o3d
import numpy as np
import os
import argparse
from termcolor import cprint

def load_and_visualize_point_cloud(file_path, visualize=True):
    """
    Load and visualize point cloud file
    Supports .ply, .pcd, .npy formats
    Output format:
    - With color: points is (N, 6) array containing [x, y, z, r, g, b]
    - Without color: points is (N, 3) array containing [x, y, z]
    """
    file_extension = os.path.splitext(file_path)[1]
    
    try:
        if file_extension in ['.ply', '.pcd']:
            point_cloud = o3d.io.read_point_cloud(file_path)
            points = np.asarray(point_cloud.points)
            # Get color information if available
            colors = np.asarray(point_cloud.colors)
            if len(colors) == 0:
                # If no color information, fill with white
                colors = np.zeros((len(points), 3))
        elif file_extension == '.npy':
            data = np.load(file_path)
            # Check data format
            if data.shape[1] == 6:  # If contains color information
                points = data[:, :3]
                raw_colors = data[:, 3:]
                
                # Check if colors are 0-255 (uint8 style) or 0.0-1.0 (float style)
                if raw_colors.max() > 1.0:
                    colors = raw_colors / 255.0
                    cprint("  [Info] Color values scaled down from 0-255 to 0.0-1.0", "yellow")
                else:
                    colors = raw_colors # Colors are already 0.0 - 1.0
                    
            else:  # Position information only
                points = data
                colors = np.zeros((len(points), 3))
            
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Print point cloud information
        cprint(f"Loaded point cloud from: {file_path}", "green")
        cprint(f"Number of points: {len(points)}", "green")
        cprint(f"Point cloud bounds:", "green")
        cprint(f"X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]", "cyan")
        cprint(f"Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]", "cyan")
        cprint(f"Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]", "cyan")

        # Determine output format based on color information
        if np.all(colors == 0):  # If all white (no color information)
            output_points = points
            cprint(f"Point cloud format: Array of shape {output_points.shape}", "green")
            cprint(f"Format: [x, y, z]", "green")
        else:
            # We keep the output format exactly as read for fidelity
            output_points = np.concatenate([points, colors], axis=1)
            cprint(f"Point cloud format: Array of shape {output_points.shape}", "green")
            cprint(f"Format: [x, y, z, r, g, b]", "green")

        if visualize:
            # Create coordinate system visualization
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0])
            
            # Change background to dark grey to better see black/dark colors
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Point Cloud Visualization", width=1024, height=768)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            opt.point_size = 5.0 # Make points slightly larger since we only have 1024
            
            vis.add_geometry(point_cloud)
            vis.add_geometry(coordinate_frame)
            vis.run()
            vis.destroy_window()
        
        return point_cloud, output_points

    except Exception as e:
        cprint(f"Error loading point cloud: {str(e)}", "red")
        return None, None

def _load_cloud(file_path):
    """Load one file -> (o3d cloud, info string). No visualization."""
    ext = os.path.splitext(file_path)[1]
    if ext in ['.ply', '.pcd']:
        pc = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pc.points)
        colors = np.asarray(pc.colors)
        if len(colors) == 0:
            colors = np.zeros((len(points), 3))
    elif ext == '.npy':
        data = np.load(file_path)
        if data.ndim == 2 and data.shape[1] == 6:
            points = data[:, :3]
            raw = data[:, 3:]
            colors = raw / 255.0 if raw.max() > 1.0 else raw
        else:
            points = data
            colors = np.zeros((len(points), 3))
    else:
        return None, f"Unsupported: {ext}"

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    info = (f"{os.path.basename(file_path)} | {len(points)} pts | "
            f"x=[{points[:,0].min():.2f},{points[:,0].max():.2f}] "
            f"y=[{points[:,1].min():.2f},{points[:,1].max():.2f}] "
            f"z=[{points[:,2].min():.2f},{points[:,2].max():.2f}]")
    return pc, info


def visualize_multiple_point_clouds(directory_path, file_pattern=None):
    """Interactive browser: step through all clouds with N / P keys."""
    import glob
    
    if file_pattern:
        files = glob.glob(os.path.join(directory_path, file_pattern))
    else:
        files = []
        for ext in ['.ply', '.pcd', '.npy']:
            files.extend(glob.glob(os.path.join(directory_path, f'*{ext}')))
    files.sort()

    if not files:
        cprint(f"No point cloud files found in {directory_path}", "red")
        return

    cprint(f"Found {len(files)} files. Controls: N/-> next | P/<- prev | Q quit",
           "green")

    state = {"cur": 0, "cache": {}}

    def get(idx):
        if idx not in state["cache"]:
            state["cache"][idx] = _load_cloud(files[idx])
        return state["cache"][idx]

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    def refresh(vis):
        pc, info = get(state["cur"])
        vis.clear_geometries()
        vis.add_geometry(frame, reset_bounding_box=False)
        if pc is not None:
            vis.add_geometry(pc, reset_bounding_box=False)
        cprint(f"[{state['cur']+1}/{len(files)}] {info}", "cyan")

    def nxt(vis):
        if state["cur"] < len(files) - 1:
            state["cur"] += 1
            refresh(vis)
        return False

    def prv(vis):
        if state["cur"] > 0:
            state["cur"] -= 1
            refresh(vis)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Browser", width=1024, height=768)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 5.0

    vis.register_key_callback(ord("N"), nxt)
    vis.register_key_callback(ord("P"), prv)
    vis.register_key_callback(262, nxt)   # Right arrow
    vis.register_key_callback(263, prv)   # Left arrow

    pc0, _ = get(0)
    vis.add_geometry(frame)
    if pc0 is not None:
        vis.add_geometry(pc0)
    refresh(vis)

    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Visualization Tool')
    parser.add_argument('--path', type=str, required=True,
                       help='Path to point cloud file or directory')
    parser.add_argument('--pattern', type=str, default=None,
                       help='File pattern to match (e.g., "cloud_*.ply")')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization (only print info)')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # Single file
        load_and_visualize_point_cloud(args.path, not args.no_vis)
    elif os.path.isdir(args.path):
        # Directory
        visualize_multiple_point_clouds(args.path, args.pattern)
    else:
        cprint(f"Invalid path: {args.path}", "red")

if __name__ == "__main__":
    main()