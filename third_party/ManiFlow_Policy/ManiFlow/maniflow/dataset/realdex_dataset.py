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
            augment=False,
            augment_shift=0.05,
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
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
        self.augment = augment
        self.augment_shift = augment_shift

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

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
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, ].astype(np.float32)
        point_cloud = sample['point_cloud'][:, ].astype(np.float32)
        action = sample['action'][:, ].astype(np.float32)

        if self.augment:
            point_cloud, action, agent_pos = self.apply_augmentation(
                point_cloud, action, agent_pos
            )

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': action
        }
        return data
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


    def apply_augmentation(self, point_cloud, action, agent_pos):
        """
        Simulating augmentation of the data.
        Augmentations are applied to all samples in a batch.
        """
        shift = np.random.uniform(
            [-self.augment_shift, -self.augment_shift, -0.01],  # X, Y frei; Z minimal
            [ self.augment_shift,  self.augment_shift,  0.01],
            size=(3,)
        ).astype(np.float32)

        # Point Cloud: nur XYZ (erste 3 Kanäle)
        point_cloud[:, :, :3] += shift.reshape(1, 1, 3)

        # Action: nur EE-Position (Index 0:3), Quat + Hand bleiben
        action[:, :3] += shift.reshape(1, 3)

        # Agent Pos: nur EE-Position (Index 0:3)
        agent_pos[:, :3] += shift.reshape(1, 3)

        return point_cloud, action, agent_pos
