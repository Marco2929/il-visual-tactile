import os
import argparse
import numpy as np
import open3d as o3d
from termcolor import cprint
import sys

sys.path.insert(0, "third_party/ManiFlow_Policy/ManiFlow/")
from maniflow.dataset.realdex_dataset import RealDexDataset

class DualZarrPointCloudViewer:
    def __init__(self, zarr_path1: str, zarr_path2: str, start_idx: int = 0, offset_x: float = 1.0):
        cprint(f"Loading Dataset 1: {zarr_path1}", "yellow")
        self.dataset1 = RealDexDataset(zarr_path=zarr_path1, horizon=1, pad_before=0, pad_after=0, augment=False)
        
        cprint(f"Loading Dataset 2: {zarr_path2}", "yellow")
        self.dataset2 = RealDexDataset(zarr_path=zarr_path2, horizon=1, pad_before=0, pad_after=0, augment=False)
        
        # Determine the maximum index based on the shorter dataset
        self.max_idx = min(len(self.dataset1), len(self.dataset2)) - 1
        if self.max_idx < 0:
            raise ValueError("One or both datasets are empty.")
            
        self.current_idx = min(max(0, start_idx), self.max_idx)
        self.offset_x = offset_x

        # Setup geometries
        self.pcd1 = o3d.geometry.PointCloud()
        self.pcd2 = o3d.geometry.PointCloud()
        
        # Coordinate frames for reference
        self.coord_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[self.offset_x, 0, 0])

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Dual Zarr Viewer", width=1280, height=720)

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 5.0

        # Load data before adding geometry to auto-scale the camera correctly
        self.load_sample(self.current_idx, is_init=True)

        self.vis.add_geometry(self.pcd1)
        self.vis.add_geometry(self.pcd2)
        self.vis.add_geometry(self.coord_frame1)
        self.vis.add_geometry(self.coord_frame2)

        # Register keyboard callbacks
        self.vis.register_key_callback(262, self.next_sample) # Right Arrow
        self.vis.register_key_callback(68, self.next_sample)  # D
        self.vis.register_key_callback(263, self.prev_sample) # Left Arrow
        self.vis.register_key_callback(65, self.prev_sample)  # A
        self.vis.register_key_callback(82, self.reset_view)   # R

    def _extract_points_and_colors(self, data: np.ndarray):
        """Helper to process tensor data into points and colors."""
        if data.shape[1] >= 6:
            points = data[:, :3]
            raw_colors = data[:, 3:6]
            colors = raw_colors / 255.0 if raw_colors.max() > 1.0 else raw_colors
        else:
            points = data[:, :3]
            colors = np.zeros((len(points), 3))
        return points, colors

    def load_sample(self, idx: int, is_init: bool = False):
        """Extracts samples from both datasets and applies the spatial offset."""
        # --- Load Dataset 1 (Left) ---
        sample1 = self.dataset1[idx]
        data1 = sample1['obs']['point_cloud'][0].numpy()
        points1, colors1 = self._extract_points_and_colors(data1)
        
        self.pcd1.points = o3d.utility.Vector3dVector(points1)
        self.pcd1.colors = o3d.utility.Vector3dVector(colors1)

        # --- Load Dataset 2 (Right, with offset) ---
        sample2 = self.dataset2[idx]
        data2 = sample2['obs']['point_cloud'][0].numpy()
        points2, colors2 = self._extract_points_and_colors(data2)
        
        # Apply spatial offset to the X-axis
        points2[:, 0] += self.offset_x
        
        self.pcd2.points = o3d.utility.Vector3dVector(points2)
        self.pcd2.colors = o3d.utility.Vector3dVector(colors2)

        if not is_init:
            self.vis.update_geometry(self.pcd1)
            self.vis.update_geometry(self.pcd2)

        cprint(f"Displaying Sample {idx} / {self.max_idx} | P1: {len(points1)} points, P2: {len(points2)} points", "green")

    def reset_view(self, vis):
        self.vis.reset_view_point(True)
        cprint("Camera view reset to fit both point clouds.", "cyan")
        return False

    def next_sample(self, vis):
        if self.current_idx < self.max_idx:
            self.current_idx += 1
            self.load_sample(self.current_idx)
        return False

    def prev_sample(self, vis):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_sample(self.current_idx)
        return False

    def run(self):
        cprint("\n--- Controls ---", "cyan")
        cprint("[Right Arrow] or [D] : Next Sample", "cyan")
        cprint("[Left Arrow]  or [A] : Previous Sample", "cyan")
        cprint("[R]                  : Reset Camera View", "cyan")
        cprint("[Q] or [Esc]         : Quit\n", "cyan")
        
        self.vis.run()
        self.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Dual Zarr Point Cloud Viewer')
    parser.add_argument('--path1', type=str, required=True, help='Path to first .zarr dataset')
    parser.add_argument('--path2', type=str, required=True, help='Path to second .zarr dataset')
    parser.add_argument('--sample', type=int, default=0, help='Start index')
    parser.add_argument('--offset', type=float, default=1.0, help='Distance to separate the point clouds along the X-axis')

    args = parser.parse_args()

    if os.path.isdir(args.path1) and os.path.isdir(args.path2):
        try:
            viewer = DualZarrPointCloudViewer(args.path1, args.path2, args.sample, args.offset)
            viewer.run()
        except Exception as e:
            cprint(f"Error initializing viewer: {e}", "red")
    else:
        cprint("Invalid paths. Ensure both --path1 and --path2 point to valid .zarr directories.", "red")

if __name__ == "__main__":
    main()