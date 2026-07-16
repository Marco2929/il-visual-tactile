# dataset_builder/model/vision_3d/pointnet_encoder.py
"""
Combined DP3 Encoder — merges pointwise/downsample (Original) 
with Uni3D/FPS-randomness (Modified Fork).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

from pytorch3d.ops import sample_farthest_points
from typing import Optional, Dict, List, Type
from termcolor import cprint


# =============================================================================
# Utility: MLP builder
# =============================================================================

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


# =============================================================================
# PointNet Encoders
# =============================================================================

class PointNetEncoderXYZRGB(nn.Module):
    """PointNet encoder for XYZ+RGB (6-channel) point clouds."""

    def __init__(self, in_channels: int, out_channels: int = 1024,
                 use_layernorm: bool = False, final_norm: str = 'none',
                 use_projection: bool = True, pointwise: bool = False, **kwargs):
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint(f"[PointNetEncoderXYZRGB] use_layernorm: {use_layernorm}", 'cyan')
        cprint(f"[PointNetEncoderXYZRGB] use_final_norm: {final_norm}", 'cyan')

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.pointwise = pointwise

    def forward(self, x):
        x = self.mlp(x)
        if not self.pointwise:
            x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZ(nn.Module):
    """PointNet encoder for XYZ-only (3-channel) point clouds."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1024,
                 use_layernorm: bool = False, final_norm: str = 'none',
                 use_projection: bool = True, pointwise: bool = False, **kwargs):
        super().__init__()
        block_channel = [64, 128, 256]
        cprint(f"[PointNetEncoderXYZ] use_layernorm: {use_layernorm}", 'cyan')
        cprint(f"[PointNetEncoderXYZ] use_final_norm: {final_norm}", 'cyan')

        assert in_channels == 3, f"PointNetEncoderXYZ only supports 3 channels, got {in_channels}"

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not using projection", "yellow")

        self.pointwise = pointwise

        # Grad-CAM hooks (optional, disabled by default)
        self.gradient = None
        self.feature = None
        self.input_pointcloud = None
        self._grad_cam_enabled = False

    def enable_grad_cam(self):
        """Enable gradient-based visualization hooks."""
        self._grad_cam_enabled = True
        self.mlp[0].register_forward_hook(self._save_input)
        self.mlp[6].register_forward_hook(self._save_feature)
        self.mlp[6].register_backward_hook(self._save_gradient)

    def forward(self, x):
        x = self.mlp(x)
        if not self.pointwise:
            x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def _save_feature(self, module, input, output):
        self.feature = output.detach() if not isinstance(output, tuple) else output[0].detach()

    def _save_input(self, module, input, output):
        self.input_pointcloud = input[0].detach()


# =============================================================================
# FPS utilities (with optional randomness)
# =============================================================================

def fps(data, number, use_random=False, random_start=True,
        random_noise_scale=0, shuffle_output=True):
    """Farthest Point Sampling with optional randomness."""
    xyz_coordinates = data[:, :, :3]
    B, N, _ = xyz_coordinates.shape

    if not use_random:
        _, fps_idx = sample_farthest_points(xyz_coordinates, K=number)
    else:
        modified_xyz = xyz_coordinates.clone()

        if random_start:
            start_indices = torch.randint(0, N, (B,), device=data.device)
            for b in range(B):
                start_idx = start_indices[b]
                modified_xyz[b, [0, start_idx]] = modified_xyz[b, [start_idx, 0]]

        if random_noise_scale > 0:
            noise = torch.randn_like(modified_xyz) * random_noise_scale
            modified_xyz = modified_xyz + noise

        _, fps_idx = sample_farthest_points(modified_xyz, K=number)

        if random_start:
            for b in range(B):
                start_idx = start_indices[b]
                mask_0 = fps_idx[b] == 0
                mask_start = fps_idx[b] == start_idx
                fps_idx[b][mask_0] = start_idx
                fps_idx[b][mask_start] = 0

        if shuffle_output:
            for b in range(B):
                perm = torch.randperm(number, device=data.device)
                fps_idx[b] = fps_idx[b][perm]

    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


def fps_torch(points, num_points):
    """Simple FPS wrapper returning (sampled_points, indices)."""
    xyz = points[:, :, :3]
    _, idx = sample_farthest_points(xyz, K=num_points)
    sampled = torch.gather(points, 1, idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))
    return sampled, idx


# =============================================================================
# Uni3D support modules
# =============================================================================

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def knn_points(query, key, k, sorted=False, transpose=False):
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind


def batch_index_select(input, index, dim):
    if index.dim() == 1:
        index = index.unsqueeze(1)
        squeeze_dim = True
    else:
        squeeze_dim = False
    views = [1 for _ in range(input.dim())]
    views[0] = index.size(0)
    views[dim] = index.size(1)
    expand_shape = list(input.shape)
    expand_shape[dim] = -1
    index = index.view(views).expand(expand_shape)
    out = torch.gather(input, dim, index)
    if squeeze_dim:
        out = out.squeeze(1)
    return out


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    B, N, _ = batch_pc.shape
    result = torch.clone(batch_pc)
    for b in range(B):
        dropout_ratio = torch.rand(1).item() * max_dropout_ratio
        drop_idx = torch.where(torch.rand(N) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            result[b, drop_idx, :] = batch_pc[b, 0, :].unsqueeze(0)
    return result


class PatchDropout(nn.Module):
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = x[:, :1]

        batch, num_tokens = x.size(0), x.size(1)
        num_keep = max(1, int(num_tokens * (1 - self.prob)))
        rand = torch.randn(batch, num_tokens, device=x.device)
        keep_idx = rand.topk(num_keep, dim=-1).indices
        batch_indices = torch.arange(batch, device=x.device)[..., None]
        x = x[batch_indices, keep_idx]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x


class Group(nn.Module):
    """Point cloud grouping with configurable FPS randomness."""
    def __init__(self, num_group, group_size, fps_random_config=None):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.fps_random_config = fps_random_config or {}

    def forward(self, xyz, color):
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group, **self.fps_random_config)
        idx = knn_point(self.group_size, xyz, center)

        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood_color = color.view(batch_size * num_points, -1)[idx]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood = neighborhood - center.unsqueeze(2)
        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features


class KNNGrouper(nn.Module):
    def __init__(self, num_groups, group_size, radius=None,
                 centralize_features=False, fps_random_config=None):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.radius = radius
        self.fps_random_config = fps_random_config or {}

    def forward(self, xyz, features, use_fps=True):
        batch_size, num_points, _ = xyz.shape
        with torch.no_grad():
            centers = fps(xyz, self.num_groups, **self.fps_random_config)
            _, knn_idx = knn_points(centers, xyz, self.group_size)

        batch_offset = torch.arange(batch_size, device=xyz.device).reshape(-1, 1, 1) * num_points
        knn_idx_flat = (knn_idx + batch_offset).reshape(-1)

        nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
        nbr_xyz = nbr_xyz.reshape(batch_size, self.num_groups, self.group_size, 3)
        nbr_xyz = nbr_xyz - centers.unsqueeze(2)
        if self.radius is not None:
            nbr_xyz = nbr_xyz / self.radius

        nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
        nbr_feats = nbr_feats.reshape(batch_size, self.num_groups, self.group_size, features.shape[-1])

        group_feats = torch.cat([nbr_xyz, nbr_feats], dim=-1)
        return dict(features=group_feats, centers=centers, knn_idx=knn_idx)


class NNGrouper(nn.Module):
    def __init__(self, num_groups, fps_random_config=None):
        super().__init__()
        self.num_groups = num_groups
        self.fps_random_config = fps_random_config or {}

    def forward(self, xyz, features):
        with torch.no_grad():
            centers = fps(xyz, self.num_groups, **self.fps_random_config)
            _, nn_idx = knn_points(xyz, centers, 1)
        nn_idx = nn_idx.squeeze(-1)
        nbr_xyz = xyz - batch_index_select(centers, nn_idx, dim=1)
        dist = torch.linalg.norm(nbr_xyz, dim=-1, keepdim=True, ord=2)
        nbr_xyz = nbr_xyz / torch.clamp(dist, min=1e-8)
        group_feats = torch.cat([nbr_xyz, dist, features], dim=-1)
        return dict(features=group_feats, centers=centers, nn_idx=nn_idx)


class PatchEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims: list):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def forward(self, point_patches):
        x = self.conv1(point_patches)
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand_as(x), x], dim=-1)
        x = self.conv2(x)
        y = torch.max(x, dim=-2).values
        return y


class Block(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, num_patches, patch_size,
                 radius=None, centralize_features=False, fps_random_config=None):
        super().__init__()
        self.out_channels = out_channels
        self.grouper = KNNGrouper(num_patches, patch_size, radius=radius,
                                  centralize_features=centralize_features,
                                  fps_random_config=fps_random_config)
        self.patch_encoder = PatchEncoder(in_channels, out_channels, [128, 512])

    def forward(self, coords, features):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]
        x = self.patch_encoder(patch_features)
        patches["embeddings"] = x
        return patches


class PatchEmbedNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_patches, fps_random_config=None):
        super().__init__()
        self.out_channels = out_channels
        hidden_dim = hidden_dim or out_channels
        self.num_patches = num_patches
        self.grouper = NNGrouper(num_patches, fps_random_config)
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks1 = nn.Sequential(*[Block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)])
        self.blocks2 = nn.Sequential(*[Block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, coords, features):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]
        nn_idx = patches["nn_idx"]
        x = self.in_proj(patch_features)
        x = self.blocks1(x)
        y = x.new_zeros(x.shape[0], self.grouper.num_groups, x.shape[-1])
        y.scatter_reduce_(1, nn_idx.unsqueeze(-1).expand_as(x), x, "amax", include_self=False)
        x = self.blocks2(y)
        x = self.norm(x)
        x = self.out_proj(x)
        patches["embeddings"] = x
        return patches


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix",
                             scale * torch.randn((3, num_pos_feats)))

    def forward(self, coords):
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


# =============================================================================
# Uni3D Encoder
# =============================================================================

class Uni3DPointcloudEncoder(nn.Module):
    """Uni3D point cloud encoder with optional pretrained weights."""

    def __init__(self, pc_model='eva02_large_patch14_448', pc_feat_dim=1024,
                 embed_dim=1024, group_size=32, num_group=512,
                 patch_dropout=0.5, drop_path_rate=0.2, pretrained_pc=None,
                 pc_encoder_dim=512, use_pretrained_weights=False,
                 pretrained_weights_path=None, feature_mode="pointsam",
                 fps_random_config=None, **kwargs):
        super().__init__()

        self.transformer = timm.create_model(pc_model, checkpoint_path=pretrained_pc,
                                             drop_path_rate=drop_path_rate)
        self.transformer_dim = self.transformer.embed_dim
        self.embed_dim = embed_dim
        self.num_group = num_group
        self.feature_mode = feature_mode

        self.patch_embed = PatchEmbed(in_channels=6, out_channels=512,
                                      num_patches=num_group, patch_size=group_size,
                                      fps_random_config=fps_random_config)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.transformer_dim))

        self.extract_global_feature = feature_mode != 'pointsam'
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.patch_dropout = PatchDropout(patch_dropout, exclude_first_token=(feature_mode == "cls")) \
            if patch_dropout > 0. else nn.Identity()
        self.out_proj = nn.Linear(self.transformer_dim, self.embed_dim)
        self.patch_proj = nn.Linear(self.patch_embed.out_channels, self.transformer_dim)

        if self.feature_mode == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.transformer_dim))
            cprint("[Uni3DPointcloudEncoder] using CLS token", "red")
        elif self.feature_mode == "max_pooling":
            cprint("[Uni3DPointcloudEncoder] using max pooling", "red")
        else:
            cprint("[Uni3DPointcloudEncoder] using pointsam (no global feature)", "red")

        if use_pretrained_weights:
            self._load_pretrained_weights(pretrained_weights_path)
        else:
            cprint("[Uni3DPointcloudEncoder] training from scratch", "red")

    def _load_pretrained_weights(self, path):
        if not os.path.exists(path):
            cprint(f"[Uni3DPointcloudEncoder] Weights not found: {path}", "red")
            return
        from safetensors.torch import load_file
        checkpoint = load_file(os.path.join(path, "model.safetensors"))
        processed = {k.replace('pc_encoder.', ''): v for k, v in checkpoint.items()
                     if k.startswith('pc_encoder.')}
        missing, unexpected = self.load_state_dict(processed, strict=False)
        cprint(f"[Uni3DPointcloudEncoder] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}", "yellow")

    def forward(self, pcd, eval=False):
        if not eval and self.training:
            pcd = random_point_dropout(pcd, max_dropout_ratio=0.8)

        pts = pcd[..., :3].contiguous()
        colors = pcd[..., 3:].contiguous()

        patches = self.patch_embed(pts, colors)
        patch_embed = patches["embeddings"]
        centers = patches["centers"]
        patch_embed = self.patch_proj(patch_embed)
        pos_embed = self.pos_embed(centers)

        if self.feature_mode == "cls":
            cls_tokens = self.cls_token.expand(patch_embed.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(pos_embed.size(0), -1, -1)
            patch_embed = torch.cat((cls_tokens, patch_embed), dim=1)
            pos_embed = torch.cat((cls_pos, pos_embed), dim=1)

        x = patch_embed + pos_embed

        if not eval and self.training:
            x = self.patch_dropout(x)
            x = self.transformer.pos_drop(x)

        for block in self.transformer.blocks:
            x = block(x)

        if self.extract_global_feature:
            if self.feature_mode == "cls":
                x = self.transformer.norm(x[:, 0, :])
            elif self.feature_mode == "max_pooling":
                x = self.transformer.norm(torch.max(x, dim=1)[0])
        else:
            x = self.transformer.norm(x)

        x = self.transformer.fc_norm(x)
        x = self.out_proj(x)

        if not self.extract_global_feature:
            pc_pe = self.pe_layer(centers)
            return x, pc_pe
        return x


# =============================================================================
# Main DP3 Encoder (combined)
# =============================================================================

class DP3Encoder(nn.Module):
    """
    Combined DP3 Encoder supporting:
    - pointnet (XYZ / XYZRGB) with pointwise option
    - uni3d / uni3d_pretrained
    - FPS downsampling
    - FPS randomness
    - cat_on_token mode
    """

    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 downsample_points=False,
                 fps_random_config=None,
                 cat_on_token=True,
                 separate_state=False,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.n_output_channels = out_channel
        self.separate_state = separate_state

        # Allow config override for state_mlp_size
        if pointcloud_encoder_cfg is not None and hasattr(pointcloud_encoder_cfg, 'get'):
            cfg_mlp = pointcloud_encoder_cfg.get('state_mlp_size', None)
            if cfg_mlp is not None:
                state_mlp_size = tuple(cfg_mlp)

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        self.imagination_shape = observation_space.get(self.imagination_key, None)

        # Downsampling
        self.downsample_points = downsample_points
        if self.downsample_points:
            self.num_points = pointcloud_encoder_cfg.get('num_points', self.point_cloud_shape[0])
        else:
            self.num_points = self.point_cloud_shape[0]  # FIX: index [0] = num points

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.cat_on_token = cat_on_token
        self.fps_random_config = fps_random_config or {}

        cprint(f"[DP3Encoder] State MLP size: {state_mlp_size}", "yellow")
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        if self.downsample_points:
            cprint(f"[DP3Encoder] Downsampling to {self.num_points} points", "yellow")

        # --- Build point cloud encoder ---
        if pointnet_type == "pointnet":
            pc_cfg = dict(pointcloud_encoder_cfg) if pointcloud_encoder_cfg else {}
            if use_pc_color:
                pc_cfg['in_channels'] = 6
                self.extractor = PointNetEncoderXYZRGB(**pc_cfg)
            else:
                pc_cfg['in_channels'] = 3
                self.extractor = PointNetEncoderXYZ(**pc_cfg)
            self.pc_encoder_extract_global_feature = True

        elif pointnet_type in ("uni3d", "uni3d_pretrained"):
            cprint(f"[DP3Encoder] Using {pointnet_type} encoder", "yellow")
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': pointnet_type == "uni3d_pretrained",
                'pretrained_weights_path': None,
                'fps_random_config': self.fps_random_config,
            }
            if pointcloud_encoder_cfg:
                uni3d_config.update(dict(pointcloud_encoder_cfg))
            uni3d_config['fps_random_config'] = self.fps_random_config

            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            self.n_output_channels = uni3d_config['embed_dim']

            feature_mode = uni3d_config.get('feature_mode', 'pointsam')
            self.pc_encoder_extract_global_feature = feature_mode != 'pointsam'
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        # --- Build state MLP ---
        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = list(state_mlp_size[:-1])
        output_dim = state_mlp_size[-1]

        self.state_feat_dim = output_dim

        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn),
            nn.Dropout(p=0.3),
        )

        # --- Compute final output dim ---
        if self.separate_state:
            # PC output dim stays as-is; state is returned separately and fed
            # into the DiTX input via state_cond (not concatenated to context).
            cprint(f"[DP3Encoder] SEPARATE STATE MODE: pc_dim={self.n_output_channels}, state_dim={self.state_feat_dim}", "green")
        elif self.cat_on_token:
            self.state_proj = nn.Linear(output_dim, self.n_output_channels)
        else:
            self.n_output_channels += output_dim

        self.pointwise = pointcloud_encoder_cfg.get('pointwise', False) if pointcloud_encoder_cfg else False

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")
        cprint(f"[DP3Encoder] pointwise: {self.pointwise}", "red")
        n_pts = self.num_points if self.pointwise else 1
        cprint(f"[DP3Encoder] output points num: {n_pts}", "red")
        cprint(f"[Encoder_type] {pointnet_type}", "red")

    def forward(self, observations: Dict, eval=False) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, f"point cloud shape: {points.shape}, expected 3D"

        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]]
            points = torch.cat([points, img_points], dim=1)

        # Downsampling
        if self.downsample_points and points.shape[1] > self.num_points:
            points, _ = fps_torch(points, self.num_points)

        # Handle Uni3D 6-channel requirement
        if self.pointnet_type in ("uni3d", "uni3d_pretrained"):
            if points.shape[-1] == 3:
                points = torch.cat([points, torch.zeros_like(points)], dim=-1)
            elif points.shape[-1] > 6:
                points = points[..., :6]

        # Encode point cloud
        if not self.pc_encoder_extract_global_feature:
            pn_feat, pc_pe = self.extractor(points, eval)
        else:
            if self.pointnet_type == "pointnet":
                pn_feat = self.extractor(points)
            else:
                pn_feat = self.extractor(points, eval)

        # Encode state
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)

        # ====== SEPARATE STATE MODE ======
        # Return PC features and state features separately. The policy feeds
        # state into the DiTX input (concatenation), while PC goes to context.
        if self.separate_state:
            if not self.pc_encoder_extract_global_feature:
                return pn_feat, state_feat, pc_pe
            return pn_feat, state_feat
        # ==================================

        # Combine features (legacy cat modes)
        if self.cat_on_token:
            state_feat = self.state_proj(state_feat)
            state_feat = state_feat.unsqueeze(1)
            final_feat = torch.cat([pn_feat, state_feat], dim=-2)
        else:
            if not self.pc_encoder_extract_global_feature:
                state_feat = state_feat.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)
            elif len(pn_feat.shape) == 3:
                state_feat = state_feat.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)
            final_feat = torch.cat([pn_feat, state_feat], dim=-1)

        if not self.pc_encoder_extract_global_feature:
            return final_feat, pc_pe
        return final_feat

    def output_shape(self):
        return self.n_output_channels