from typing import Dict
import torch
import numpy as np
import copy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset
from termcolor import cprint


class RealDexDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            # Spatial augmentation
            augment=False,
            # Additional modalitites
            use_force=False,
            use_progress=False,
            aux_action=False,
            aux_force=False,
            augment_shift=0.05,
            # Noise augmentation 
            use_noise_augmentation=False,
            pc_xyz_noise_std=0.002,
            pc_rgb_noise_std=0.01,
            agent_pos_noise_std=0.0002,
            # Color jitter
            use_color_jitter=False,
            brightness_range=(-0.125, 0.125),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            ):
        super().__init__()
        self.task_name = task_name
        self.augment = augment
        self.augment_shift = augment_shift
        self.use_noise_augmentation = use_noise_augmentation
        self.pc_xyz_noise_std = pc_xyz_noise_std
        self.pc_rgb_noise_std = pc_rgb_noise_std
        self.agent_pos_noise_std = agent_pos_noise_std
        self.use_color_jitter = use_color_jitter
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.aux_action = aux_action
        self.aux_force = aux_force
        self.use_force = use_force
        self.use_progress = use_progress

        # 1. Base keys that must always be loaded
        keys_to_load = ['state', 'action', 'point_cloud']

        # 2. Dynamically add conditioning (Inputs)
        if use_force:
            keys_to_load.append('hand_obs')

        # 3. Dynamically add progress
        if use_progress:
            keys_to_load.append('progress')

        # 4. Dynamically add Auxiliary Tasks (Outputs)
        if aux_action:
            keys_to_load.append('aux_action/ee_pose')
        if aux_force:
            keys_to_load.append('aux_action/force')

        # 5. Load the ReplayBuffer exactly ONCE
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys_to_load
        )

        # 6. Dynamic logging for clarity
        cprint("--------------------------", "cyan")
        cprint(f"Dataset loaded with keys: {keys_to_load}", "cyan")
        
        if use_force:
            cprint("Progress pediction enabled", "cyan")

        if use_force:
            cprint("Conditioning: hand_obs (Force/Palm Features) enabled", "cyan")
            
        if aux_action or aux_force:
            cprint(f"Auxiliary Tasks enabled: ee_pose={aux_action}, force={aux_force}", "cyan")
        else:
            cprint("Auxiliary Tasks disabled: action dim = joint only", "yellow")
            
        cprint("--------------------------", "cyan")
            
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

        # Determine if point cloud has color (6 channels)
        pc_shape = self.replay_buffer['point_cloud'].shape
        self.has_pc_color = pc_shape[-1] == 6

        self.proprio_mask_prob = 0.0

        # Log augmentation settings
        cprint("--------------------------", "green")
        cprint(f"[RealDexDataset] Augmentation settings:", "green")
        cprint(f"  Spatial shift: {augment} (±{augment_shift}m)", "green")
        cprint(f"  Noise: {use_noise_augmentation} (xyz={pc_xyz_noise_std}, rgb={pc_rgb_noise_std}, pos={agent_pos_noise_std})", "green")
        cprint(f"  Color jitter: {use_color_jitter} (has_color={self.has_pc_color})", "green")
        cprint("--------------------------", "green")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        # Disable ALL augmentation for validation
        val_set.augment = False
        val_set.use_noise_augmentation = False
        val_set.use_color_jitter = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        action_components = [self.replay_buffer['action']]
        
        if 'aux_action/ee_pose' in self.replay_buffer:
            action_components.append(self.replay_buffer['aux_action/ee_pose'])
            
        if 'aux_action/force' in self.replay_buffer:
            action_components.append(self.replay_buffer['aux_action/force'])
            
        combined_action = np.concatenate(action_components, axis=-1)
        
        data = {
            'action': combined_action,
            'agent_pos': self.replay_buffer['state'],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        
        if 'hand_obs' in self.replay_buffer:
            data['hand_obs'] = self.replay_buffer['hand_obs']

        if 'progress' in self.replay_buffer:
            progress_data = self.replay_buffer['progress']
            if progress_data.ndim == 1:
                progress_data = progress_data[:, np.newaxis]
            data['progress'] = progress_data
            
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # 1. Base inputs
        agent_pos = sample['state'][:].astype(np.float32)
        point_cloud = sample['point_cloud'][:].astype(np.float32)

        # 2. Dynamically build the action array based on available keys
        action_components = [sample['action'][:].astype(np.float32)]

        if 'aux_action/ee_pose' in sample:
            action_components.append(sample['aux_action/ee_pose'][:].astype(np.float32))
            
        if 'aux_action/force' in sample:
            action_components.append(sample['aux_action/force'][:].astype(np.float32))

        action = np.concatenate(action_components, axis=-1)

        if self.augment:
            if self.aux_action:
                cprint("Aux action is not using spatial shift", "red")
                pass
            else: 
                point_cloud, action, agent_pos = self._apply_spatial_shift(point_cloud, action, agent_pos)

        # 4. Sensor-Noise (XYZ + RGB + Proprioception)
        if self.use_noise_augmentation:
            point_cloud, agent_pos = self._apply_noise(point_cloud, agent_pos)

        # 5. Color Jitter (only if PC has color)
        if self.use_color_jitter and self.has_pc_color:
            point_cloud = self._apply_color_jitter(point_cloud)

        # 6. Progressive Proprioception Masking
        if self.augment and np.random.random() < self.proprio_mask_prob:
                agent_pos = np.random.uniform(-1.0, 1.0, size=agent_pos.shape).astype(np.float32)

        # 7. Dynamically build the observation dictionary
        obs_dict = {
            'point_cloud': point_cloud,
            'agent_pos': agent_pos,
        }
        
        # Inject hand_obs (Force/Palm features) if they were loaded by the dataset
        if 'hand_obs' in sample:
            obs_dict['hand_obs'] = sample['hand_obs'][:].astype(np.float32)

        data = {
            'obs': obs_dict,
            'action': action
        }

        if 'progress' in sample:
            progress = sample['progress'][:].astype(np.float32)
            if progress.ndim == 1:
                progress = progress[:, np.newaxis]  # (T,) → (T, 1)
            data['progress'] = progress

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    # =========================================================================
    # Augmentation methods
    # =========================================================================

    def _apply_spatial_shift(self, point_cloud, action, agent_pos):
        """Geometrisch konsistenter Shift — simuliert andere Objektposition."""
        shift = np.random.uniform(
            [-self.augment_shift, -self.augment_shift, -0.01],
            [ self.augment_shift,  self.augment_shift,  0.01],
            size=(3,)
        ).astype(np.float32)

        point_cloud[:, :, :3] += shift.reshape(1, 1, 3)
        action[:, :3] += shift.reshape(1, 3)
        agent_pos[:, :3] += shift.reshape(1, 3)
        return point_cloud, action, agent_pos

    def _apply_noise(self, point_cloud, agent_pos):
        """Unabhängiges Sensor-Noise — Robustheit gegen Messrauschen."""
        # XYZ noise auf Punktwolke
        xyz_noise = np.random.normal(
            0, self.pc_xyz_noise_std, point_cloud[:, :, :3].shape
        ).astype(np.float32)
        xyz_noise = np.clip(xyz_noise, -2*self.pc_xyz_noise_std, 2*self.pc_xyz_noise_std)
        point_cloud[:, :, :3] += xyz_noise

        # RGB noise (nur wenn Farbe vorhanden)
        if self.has_pc_color and self.pc_rgb_noise_std > 0:
            rgb_noise = np.random.normal(
                0, self.pc_rgb_noise_std, point_cloud[:, :, 3:].shape
            ).astype(np.float32)
            rgb_noise = np.clip(rgb_noise, -2*self.pc_rgb_noise_std, 2*self.pc_rgb_noise_std)
            point_cloud[:, :, 3:] += rgb_noise
            # Clip RGB to valid range [0, 1]
            point_cloud[:, :, 3:] = np.clip(point_cloud[:, :, 3:], 0.0, 1.0)

        # Leichtes Noise auf Propriozeption
        pos_noise = np.random.normal(
            0, self.agent_pos_noise_std, agent_pos.shape
        ).astype(np.float32)
        pos_noise = np.clip(pos_noise, -2*self.agent_pos_noise_std, 2*self.agent_pos_noise_std)
        agent_pos += pos_noise

        return point_cloud, agent_pos

    def _apply_color_jitter(self, point_cloud):
        """Color Jitter auf RGB-Kanäle der Punktwolke."""
        # Nur auf RGB (Kanäle 3:6)
        rgb = point_cloud[:, :, 3:]  # (T, N, 3)
        original_shape = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)

        # 1. Brightness
        brightness_delta = np.random.uniform(
            self.brightness_range[0], self.brightness_range[1])
        rgb_flat = rgb_flat + brightness_delta

        # 2. Contrast
        contrast_factor = np.random.uniform(
            self.contrast_range[0], self.contrast_range[1])
        rgb_mean = rgb_flat.mean()
        rgb_flat = (rgb_flat - rgb_mean) * contrast_factor + rgb_mean

        # 3. Saturation
        saturation_factor = np.random.uniform(
            self.saturation_range[0], self.saturation_range[1])
        gray = (0.299 * rgb_flat[:, 0:1] +
                0.587 * rgb_flat[:, 1:2] +
                0.114 * rgb_flat[:, 2:3])
        rgb_flat = gray + (rgb_flat - gray) * saturation_factor

        # Clip to [0, 1]
        rgb_flat = np.clip(rgb_flat, 0.0, 1.0)

        point_cloud[:, :, 3:] = rgb_flat.reshape(original_shape)
        return point_cloud