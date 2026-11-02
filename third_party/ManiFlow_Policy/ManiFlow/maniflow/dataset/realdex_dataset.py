from typing import Dict
import torch
import numpy as np
import copy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer
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
            use_wrist_rgb=False,
            # Keep only the first N wrist frames per sample (None = full horizon).
            wrist_obs_steps=None,
            use_front_rgb=False,
            # Same for the front camera. Independent of wrist_obs_steps so the
            # two cameras can carry different history lengths if needed.
            front_obs_steps=None,
            # Noise augmentation 
            use_noise_augmentation=False,
            pc_xyz_noise_std=0.002,
            agent_pos_noise_std=0.0002,
            use_progress=False,
            **kwargs,
            ):
        super().__init__()
        self.task_name = task_name
        self.use_progress = use_progress
        self.use_noise_augmentation = use_noise_augmentation
        self.pc_xyz_noise_std = pc_xyz_noise_std
        self.agent_pos_noise_std = agent_pos_noise_std
        self.use_wrist_rgb = use_wrist_rgb
        self.wrist_obs_steps = wrist_obs_steps
        self.use_front_rgb = use_front_rgb
        self.front_obs_steps = front_obs_steps

        # 1. Base keys that must always be loaded
        keys_to_load = ['state', 'action', 'point_cloud']

        # 2. Dynamically add RGB conditioning (Inputs).
        # Order is SORTED, not insertion order: the encoder concatenates its
        # per-key token groups in sorted key order, and the two must agree or
        # the conditioning tokens land in the wrong group after a reload.
        # ('front_rgb' < 'wrist_rgb')
        self.rgb_keys = sorted(
            (["wrist_rgb"] if use_wrist_rgb else [])
            + (["front_rgb"] if use_front_rgb else [])
        )
        # Per-key history length, consumed in _sample_to_data().
        self.rgb_obs_steps = {
            "wrist_rgb": wrist_obs_steps,
            "front_rgb": front_obs_steps,
        }
        keys_to_load.extend(self.rgb_keys)

        # 3. Load the ReplayBuffer exactly ONCE
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys_to_load
        )

        # 4. Dynamic logging for clarity
        cprint("--------------------------", "cyan")
        cprint(f"Dataset loaded with keys: {keys_to_load}", "cyan")

        for key in self.rgb_keys:
            cprint(f"Conditioning: {key} enabled "
                   f"{self.replay_buffer[key].shape[1:]}", "cyan")
            steps = self.rgb_obs_steps.get(key)
            if steps is not None:
                cprint(f"  {key} sliced to first {steps} frames "
                       f"of horizon {horizon}", "cyan")
            
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

        # Log augmentation settings
        cprint("--------------------------", "green")
        cprint(f"[RealDexDataset] Augmentation settings:", "green")
        cprint(f"  Noise: {use_noise_augmentation} (xyz={pc_xyz_noise_std}, pos={agent_pos_noise_std})", "green")
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
        val_set.use_noise_augmentation = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'point_cloud': self.replay_buffer['point_cloud'],
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # 1. Base inputs
        agent_pos = sample['state'][:].astype(np.float32)
        point_cloud = sample['point_cloud'][:].astype(np.float32)

        # 2. Action
        action = sample['action'][:].astype(np.float32)

        # 3. Sensor noise (PC xyz + proprioception)
        if self.use_noise_augmentation:
            point_cloud, agent_pos = self._apply_noise(point_cloud, agent_pos)

        # 4. Build the observation dictionary
        obs_dict = {
            'point_cloud': point_cloud,
            'agent_pos': agent_pos,
        }

        # Inject every loaded RGB stream.
        # NOTE: NOT normalized here - the image encoder does /255 itself.
        # RGB is stored as (T, H, W, C) uint8 in the zarr.
        # Keep it uint8! Casting to float32 here is a 4x memory blowup in every
        # dataloader worker (horizon=64 -> 9.4MB vs 37.7MB per sample) and is
        # pure waste, since the encoder divides by 255 on the GPU anyway.
        # Only the first n_obs_steps frames are ever consumed by the encoder,
        # so keeping the full horizon wastes ~30x memory per worker.
        for key in self.rgb_keys:
            if key not in sample:
                continue
            steps = self.rgb_obs_steps.get(key)
            obs_dict[key] = (sample[key][:steps] if steps is not None
                             else sample[key][:])

        return {
            'obs': obs_dict,
            'action': action
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    # =========================================================================
    # Augmentation methods
    # =========================================================================

    def _apply_noise(self, point_cloud, agent_pos):
        """Unabhängiges Sensor-Noise — Robustheit gegen Messrauschen."""
        # XYZ noise auf Punktwolke
        xyz_noise = np.random.normal(
            0, self.pc_xyz_noise_std, point_cloud[:, :, :3].shape
        ).astype(np.float32)
        xyz_noise = np.clip(xyz_noise, -2*self.pc_xyz_noise_std, 2*self.pc_xyz_noise_std)
        point_cloud[:, :, :3] += xyz_noise

        # Leichtes Noise auf Propriozeption
        pos_noise = np.random.normal(
            0, self.agent_pos_noise_std, agent_pos.shape
        ).astype(np.float32)
        pos_noise = np.clip(pos_noise, -2*self.agent_pos_noise_std, 2*self.agent_pos_noise_std)
        agent_pos += pos_noise

        return point_cloud, agent_pos
