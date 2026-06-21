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
from maniflow.common.palm_transform import PalmFrameTransform

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
            urdf_path="src/moveit_ur_hand_config/urdf/tesollo_hand_ur10e_flange_workcell.urdf",
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'hand_joints', 'hand_ft'])
        
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

        self.transform = PalmFrameTransform(urdf_path=urdf_path)
        self._precompute_all_hand_obs()

    def _precompute_hand_obs(self):
        """Pre-compute transformed hand obs for normalizer fitting."""
        all_joints = self.replay_buffer['hand_joints']    # (N, 20)
        all_ft = self.replay_buffer['hand_ft']            # (N, 30)
        
        # Transform all samples (vectorized per-sample)
        N = all_joints.shape[0]
        self._hand_obs_cache = np.zeros((N, 5, 13), dtype=np.float32)
        
        for i in range(N):
            self._hand_obs_cache[i] = self.transform(all_joints[i], all_ft[i])
        
        # Flatten to (N, 65) for normalizer
        self._hand_obs_flat = self._hand_obs_cache.reshape(N, -1)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
            'hand_obs': self._hand_obs_flat,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    
    def _precompute_all_hand_obs(self):
        """Einmal alles berechnen, dann im RAM halten."""
        all_joints = self.replay_buffer['hand_joints']  # (N, 20)
        all_ft = self.replay_buffer['hand_ft']          # (N, 30)
        
        N = all_joints.shape[0]
        print(f"[RealDexDataset] Pre-computing hand observations for {N} frames...")
        
        hand_obs_cache = np.zeros((N, 65), dtype=np.float32)  # direkt flat
        for i in range(N):
            hand_obs_cache[i] = self.transform(all_joints[i], all_ft[i]).reshape(-1)
        
        self._hand_obs_cache = hand_obs_cache
        
        self._hand_obs_flat = hand_obs_cache
        
        print(f"[RealDexDataset] Done. Shape: {hand_obs_cache.shape}")

    
    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, :].astype(np.float32)
        point_cloud = sample['point_cloud'][:, :].astype(np.float32)
        
        hand_obs = sample['hand_obs']  # (T, 65)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
                'hand_obs': hand_obs,         # (T, 65)
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

