import os

import open3d as o3d
import kinpy as kp
import pytorch3d.ops as torch3d_ops
import torch
import cv2
import numpy as np
from termcolor import cprint

import sys
sys.path.insert(0, "/ros2_ws/data/record_utils")

from camera_info import camera_intrinsics, camera_extrinsics, workspace

# Must match data/configs/foil_lifting_wrist.yaml's proprioception.urdf_path /
# end_link, which is what built the ee_pose in the training set. `tool0`, not
# `right_inspire_base_link`: with the Inspire tip frame every pose would carry
# a constant ~18.6 mm offset against the data the policy was trained on, and
# nothing would flag it - the poses stay perfectly plausible.
URDF_PATH = os.environ.get(
    "URDF_PATH",
    "/ros2_ws/src/moveit_ur_hand_config/urdf/tesollo_hand_ur10e_flange_workcell.urdf")
ROBOT_TIP_FRAME = os.environ.get("ROBOT_TIP_FRAME", "tool0")

# ---------------------------------------------------------------------------
# Wrist RGB preprocessing
# ---------------------------------------------------------------------------
# Mirrors dataset_builder/processing/rgb.py, because any divergence here is a
# train/serve skew that no error message would reveal - the policy would just
# perform worse. The contract, read off the training path end to end:
#
#   * uint8 in [0, 255], NOT scaled. DINOv3ObsEncoder does /255 and applies
#     its own timm-resolved mean/std.
#   * channels-LAST (T, H, W, C). maniflow_multimodal_policy._encode_wrist()
#     is the single HWC->CHW boundary and permutes there, so handing it CHW
#     would double-permute.
#   * RGB order, not BGR. cache/loaders.py:load_wrist_rgb reads the PNGs with
#     cv2.imread (BGR) and flips via [..., ::-1], so the zarr holds RGB.
#   * 192x256, from INTER_AREA. Source is the native 480x640 wrist frame;
#     both are 4:3, so the builder's center_crop is a no-op and a plain
#     resize reproduces it exactly.
WRIST_RGB_HW = (192, 256)


def preprocess_wrist_rgb(color_bgr: np.ndarray,
                         out_hw: tuple = WRIST_RGB_HW) -> np.ndarray:
    """Live wrist frame (BGR, as delivered by cv_bridge) -> (H, W, 3) uint8 RGB.

    Resize happens BEFORE the BGR->RGB flip purely for efficiency; the two
    commute. INTER_AREA matches the builder's default and is the correct
    kernel for downscaling colour.
    """
    out_h, out_w = out_hw
    if (color_bgr.shape[0], color_bgr.shape[1]) != (out_h, out_w):
        color_bgr = cv2.resize(color_bgr, (out_w, out_h),
                               interpolation=cv2.INTER_AREA)
    rgb = color_bgr[:, :, ::-1]
    return np.ascontiguousarray(rgb, dtype=np.uint8)


class PointCloudProcessor:
    """Point Cloud Processor"""
    def __init__(self):
        # Camera intrinsics
        self.camera_intrinsics = camera_intrinsics
        
        # Extrinsic matrix
        self.extrinsic_matrix = camera_extrinsics
        
        # Workspace range
        self.WORK_SPACE = workspace

        # Which unit the depth stream was last seen in, so the warning in
        # _check_depth_units() is emitted once per unit rather than per frame.
        self._depth_unit_reported = None

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

    
    # Depth unit normalisation to MILLIMETRES, mirroring the dataset builder
    # (record_extractor/point_cloud_utils.generate_point_cloud_from_depth).
    #
    # This is not cosmetic. Every filter constant in this class is a
    # millimetre-domain value inherited from the builder, so feeding metres in
    # silently rescales three thresholds by 1000x:
    #
    #   _remove_flying_pixels(gradient_threshold=100.0)
    #       mm: a 10 cm depth step.  m: 100 m -> the mask never triggers, so
    #       flying pixels at object edges survive at inference time although
    #       they were removed from the training clouds.
    #   filter_zero_values(sigmaColor=50)
    #       mm: 5 cm.  m: 50 m -> the bilateral filter stops being edge-aware
    #       and degenerates into a plain Gaussian that smears across depth
    #       discontinuities.
    #   filter_zero_values(invalid sentinel 6000.0)
    #       mm: 6 m, i.e. "far away".  m: 6000 m -> harmless only because
    #       depth_trunc discards it later.
    #
    # The live topic /zed/zed_node/depth/depth_registered is 32FC1 in metres,
    # so in practice this converts on every frame. Normalising here rather
    # than retuning the constants keeps exactly one definition of the filter
    # behaviour, shared with the builder.
    #
    # Threshold: a scene at 0.5-3 m reads 500-3000 in mm and 0.5-3 in metres,
    # so the median of the valid pixels separates the two unambiguously.
    _DEPTH_UNIT_MEDIAN_MM_CUTOFF = 20.0

    def _normalize_depth_to_mm(self, depth_img):
        """Return depth in millimetres, converting from metres if needed.

        Same median test the builder uses. Reports the detected unit once per
        change so a driver that switches encoding is visible in the log rather
        than inferred from a plot of the cloud.
        """
        valid = depth_img[np.isfinite(depth_img) & (depth_img > 0)]
        if valid.size == 0:
            return depth_img

        median = float(np.median(valid))
        unit = "metres" if median < self._DEPTH_UNIT_MEDIAN_MM_CUTOFF else "millimetres"

        if self._depth_unit_reported != unit:
            self._depth_unit_reported = unit
            if unit == "metres":
                cprint(
                    f"[PointCloudProcessor] incoming depth median "
                    f"{median:.3f} -> METRES; converting to mm so the "
                    f"millimetre-domain filter constants match the dataset "
                    f"builder.", "yellow", attrs=["bold"])
            else:
                cprint(
                    f"[PointCloudProcessor] incoming depth median "
                    f"{median:.1f} -> millimetres already; no conversion.",
                    "yellow", attrs=["bold"])

        # float32 unconditionally: cv2.bilateralFilter in
        # filter_zero_values() accepts only 8-bit or float32, and a float64
        # depth topic would otherwise raise there instead of here.
        if unit == "metres":
            return (depth_img * 1000.0).astype(np.float32)
        return depth_img.astype(np.float32, copy=False)

    def process_rgbd(self, color_img, depth_img, num_points=1024):
        """Process RGB-D image to generate an (num_points, 6) XYZRGB cloud.

        Returns XYZ **and** RGB because the trained PointNetEncoderXYZRGB was
        built with in_channels=6 (shape_meta point_cloud: [1024, 6]). Dropping
        the colour channels here raises
            mat1 and mat2 shapes cannot be multiplied (Nx3 and 6x64)
        in the encoder's first Linear. Mirrors
        record_extractor/point_cloud_utils.generate_point_cloud_from_depth()
        with include_color=True, including the [0, 1] colour range that
        Open3D's pcd.colors returns and that the zarr therefore holds.
        """
        # To mm FIRST: the two filters below are tuned in millimetres.
        depth_img = self._normalize_depth_to_mm(depth_img)

        depth_img = self._remove_flying_pixels(depth_img, gradient_threshold=100.0, dilate_size=5)
        depth_img = self.filter_zero_values(depth_img)

        # BGR -> RGB before handing the frame to Open3D. cv_bridge delivers
        # bgr8 and the builder flips with [:, :, ::-1] too; skipping it here
        # would swap R and B against the training colours with no error.
        color_rgb = np.ascontiguousarray(color_img[:, :, ::-1], dtype=np.uint8)

        # Convert to Open3D format
        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth_img)
        
        # Create RGBD image
        # depth_scale=1000.0 divides the mm depth back into METRES, matching
        # the builder and the metre-scale workspace bounds / zarr clouds.
        # depth_trunc stays 3.0: Open3D applies it AFTER the depth_scale
        # division, so it is in metres in both paths.
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, 
            o3d_depth,
            depth_scale=1000.0,
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
        colors = np.asarray(pcd.colors)  # [0, 1], same range as the zarr

        if len(points) == 0:
            return None

        features = np.concatenate([points, colors], axis=-1)  # (N, 6)

        # Workspace filtering (XYZ only, columns 0..2)
        mask = (
            (features[:, 0] >= self.WORK_SPACE['x'][0]) & 
            (features[:, 0] <= self.WORK_SPACE['x'][1]) &
            (features[:, 1] >= self.WORK_SPACE['y'][0]) & 
            (features[:, 1] <= self.WORK_SPACE['y'][1]) &
            (features[:, 2] >= self.WORK_SPACE['z'][0]) & 
            (features[:, 2] <= self.WORK_SPACE['z'][1])
        )
        features = features[mask]

        if len(features) == 0:
            return None

        # Pad by resampling when the workspace filter left too few points, so
        # the returned shape is always (num_points, 6) as the encoder expects.
        if len(features) < num_points:
            pad_idx = np.random.choice(len(features), num_points, replace=True)
            features = features[pad_idx]

        features_t = torch.from_numpy(features).float().cuda()

        # Cap FPS input to a fixed size → deterministic FPS runtime.
        # Random pre-subsample keeps spatial coverage while removing the
        # data-dependent latency spikes (100ms↔1000ms) from variable N.
        MAX_FPS_INPUT = 8192
        n = features_t.shape[0]
        if n > MAX_FPS_INPUT:
            idx = torch.randperm(n, device=features_t.device)[:MAX_FPS_INPUT]
            features_t = features_t[idx]

        # FPS distances on XYZ only, then gather the full 6D rows by index -
        # exactly what the builder does. Running FPS on all six channels would
        # let colour influence which points are kept.
        _, fps_idx = torch3d_ops.sample_farthest_points(
            points=features_t[:, :3].unsqueeze(0).contiguous(),
            K=[num_points]
        )
        sampled = features_t[fps_idx.squeeze(0).long()]
        return sampled.cpu().numpy().astype(np.float32)
    
# Arm joint names in the order kinpy's serial chain expects them. Same list as
# record.py's UR_JOINT_NAMES and deploy.py's ARM_JOINT_NAMES; kept local so
# compute_fk() has no import cycle with deploy.
ARM_JOINT_NAMES_FK = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


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

# Hand DoF kept in the observation, as hand-local indices into the 20-joint
# finger-major vector. Mirrors proprioception.pinch_hand_indices() with
# PINCH_FINGERS=(1, 2): thumb + index, i.e. rj_dg_1_1..1_4, rj_dg_2_1..2_4.
#
# This is the OBSERVATION side of the same reduction deploy.py reverses on the
# ACTION side (expand_hand_action). Set to range(20) for a full-hand
# checkpoint, together with POLICY_HAND_JOINTS there.
ARM_DOF = 6
JOINTS_PER_FINGER = 4
PINCH_FINGERS = (1, 2)
PINCH_HAND_INDICES = [(f - 1) * JOINTS_PER_FINGER + j
                      for f in PINCH_FINGERS
                      for j in range(JOINTS_PER_FINGER)]


def compute_fk(chain, full_state):
    """(26,) joint state -> (15,) agent_pos, matching the trained layout.

    Input is 6 arm joints (radians) + 20 hand joints (DEGREES, as recorded).
    Output is [x, y, z, qx, qy, qz, qw] of the tip frame plus the pinch hand
    joints, i.e. exactly what processing/proprioception.build_state_vector()
    produced for the zarr: ee_pose followed by the reduced hand block.

    Verified against the built dataset: state columns 3..6 have unit norm
    (mean 1.000000, std 0.0 over 2000 frames), confirming the 7/8 split rather
    than a 6/9 one.
    """
    ur_joints = full_state[:ARM_DOF]
    hand_joints = full_state[ARM_DOF:]

    if len(hand_joints) != 20:
        raise ValueError(
            f"expected 20 hand joints after the {ARM_DOF} arm joints, got "
            f"{len(hand_joints)}. The Tesollo DG-5F has 20 DoF; a 6-DoF "
            f"vector means the Inspire joint names are still configured.")

    # kinpy's serial chain was built with root_link_name='base_link' and
    # end_link_name=ROBOT_TIP_FRAME, so these names are the arm joints only.
    th = dict(zip(ARM_JOINT_NAMES_FK, ur_joints))
    ret = chain.forward_kinematics(th)

    # kinpy returns [w, x, y, z]; the dataset stores [x, y, z, w].
    rot_xyzw = np.array([ret.rot[1], ret.rot[2], ret.rot[3], ret.rot[0]])

    pinch = np.asarray(hand_joints, dtype=np.float32)[PINCH_HAND_INDICES]

    return np.concatenate([ret.pos, rot_xyzw, pinch])

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