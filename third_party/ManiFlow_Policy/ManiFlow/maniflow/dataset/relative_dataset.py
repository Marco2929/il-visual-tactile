"""Dataset variant that encodes EE poses relative to the last observed pose.

EXPERIMENT ONLY. A subclass rather than a flag inside RealDexDataset, so the
baseline file stays untouched and the two cannot interfere; switch between
them with `_target_` in the task config.

WHAT CHANGES
------------
The builder writes ABSOLUTE poses (quaternion) per frame, which is the only
thing it can do: the relativisation depends on which window the sampler
draws, and windows do not exist until training time. A frame near the middle
of an episode appears in ~horizon different windows, each with a different
base pose, so there is no single "relative action" to precompute.

This class therefore relativises per sample, AFTER the window is drawn:

    base            = state[n_obs_steps - 1]      last OBSERVED pose
    state[k]        -> R_base^T (T_k - T_base),  R_base^T R_k
    action[j]       -> same transform, same base
    hand joints     -> untouched, absolute
    rotations       -> 6D (continuous, no q/-q ambiguity)

Vector layout grows 15 -> 17 (xyz 3 + rot6d 6 + hand 8), which the task
shape_meta must match.

WHY THE LAST OBSERVED POSE AND NOT THE FIRST ACTION TARGET
----------------------------------------------------------
At inference the policy conditions on observations only, so the base has to
be something it can actually measure: the newest pose in the observation
horizon. Encoding the chunk against the first commanded target instead makes
training and inference disagree about the origin of the frame, which is
reported to cost most of the task success rate. `assert_invariants=True`
checks the base really is the identity pose, which is what catches this.

HANDS STAY ABSOLUTE
-------------------
A relative finger angle says "close a bit more" and throws away the grasp
configuration - the thing a pinch is defined by. Joint angles are already
pallet-independent, so there is nothing to gain.

ACTIONS ARE MEASURED, NOT COMMANDED
-----------------------------------
The recorder does not log the teleop command stream, so actions are the
FUTURE MEASURED poses (builder `type: next_state`). The impedance
controller lags under contact, so a measured pose is the achieved one rather
than the requested one and the dataset slightly understates how hard the
operator pushed. Not quantified here; it applies to the baseline equally, so
the A/B stays fair.
"""

from typing import Dict

import numpy as np
import torch
from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.dataset.realdex_dataset import RealDexDataset
from maniflow.dataset.relative_pose import (
    HAND_SLICE, POS_SLICE, QUAT_SLICE, RELATIVE_DIM, assert_identity_pose,
    assert_valid_rotation, matrix_to_rot6d, quat_to_matrix, to_relative,
)
from maniflow.model.common.normalizer import LinearNormalizer


class RelativeRealDexDataset(RealDexDataset):
    """RealDexDataset with base-relative EE poses and 6D rotations.

    Extra args:
        n_obs_steps: length of the observation horizon. The base pose is
            state[n_obs_steps - 1]; MUST match the policy's n_obs_steps or
            training and inference disagree about the frame origin.
        assert_invariants: run the pose invariant checks on every sample.
            Cheap (a few hundred flops) and worth leaving on - the failure
            it catches is otherwise silent.
        normalizer_samples: how many windows to draw when fitting the
            normalizer. The statistics must come from relativised data, and
            running all ~40k windows would decompress every point cloud for
            nothing.
    """

    def __init__(self, *args, n_obs_steps=2, assert_invariants=True,
                 normalizer_samples=2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_obs_steps = int(n_obs_steps)
        self.assert_invariants = bool(assert_invariants)
        self.normalizer_samples = int(normalizer_samples)

        state_dim = self.replay_buffer['state'].shape[-1]
        if state_dim != 15:
            raise ValueError(
                f"relative encoding assumes the 15-dim pinch state "
                f"(xyz 3 + quat 4 + hand 8), got {state_dim}. The slices in "
                f"relative_pose.py would point at the wrong columns."
            )
        if self.horizon < self.n_obs_steps:
            raise ValueError(
                f"horizon {self.horizon} < n_obs_steps {self.n_obs_steps}: "
                f"the base pose would sit outside the sampled window."
            )

        cprint("--------------------------", "magenta")
        cprint("[RelativeRealDexDataset] base-relative EE poses", "magenta")
        cprint(f"  base pose      : state[{self.n_obs_steps - 1}] "
               f"(last observed)", "magenta")
        cprint(f"  vector layout  : 15 -> {RELATIVE_DIM} "
               f"(xyz 3 + rot6d 6 + hand 8)", "magenta")
        cprint(f"  asserts        : {self.assert_invariants}", "magenta")
        cprint("--------------------------", "magenta")

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------

    def _split(self, vec):
        """(T, 15) -> positions, rotation matrices, hand joints."""
        return (
            vec[:, POS_SLICE].astype(np.float64),
            quat_to_matrix(vec[:, QUAT_SLICE]),
            vec[:, HAND_SLICE],
        )

    def _encode(self, T, R, hand, T_base, R_base):
        """Relativise + 6D + reattach the absolute hand joints."""
        T_rel, R_rel = to_relative(T, R, T_base, R_base)
        return np.concatenate(
            [T_rel, matrix_to_rot6d(R_rel), hand], axis=-1
        ).astype(np.float32), T_rel, R_rel

    def _relativise(self, state, action):
        """(T, 15) state + (T, 15) action -> (T, 17) each, shared base."""
        T_s, R_s, hand_s = self._split(state)
        T_a, R_a, hand_a = self._split(action)

        # The base pose. Index n_obs_steps-1 is the NEWEST frame the policy
        # is allowed to see; anything later would be using the future.
        base = self.n_obs_steps - 1
        T_base = T_s[base].copy()
        R_base = R_s[base].copy()

        state_rel, T_rel_s, R_rel_s = self._encode(
            T_s, R_s, hand_s, T_base, R_base)
        # Same base object for the actions - not a recomputed copy.
        action_rel, _, R_rel_a = self._encode(
            T_a, R_a, hand_a, T_base, R_base)

        if self.assert_invariants:
            # 1. the base pose in its own frame is the identity
            assert_identity_pose(
                T_rel_s[base], R_rel_s[base],
                what=f"state[{base}] (base pose)")
            # 2. relativisation produced genuine rotations
            assert_valid_rotation(R_rel_s, what="relative state rotation")
            assert_valid_rotation(R_rel_a, what="relative action rotation")
            # 3. no NaNs leaking in from a padded or malformed frame
            if not (np.isfinite(state_rel).all()
                    and np.isfinite(action_rel).all()):
                raise AssertionError(
                    "relative encoding produced non-finite values")

        return state_rel, action_rel

    def _sample_to_data(self, sample):
        # The parent handles point clouds and RGB; this class owns the pose
        # encoding.
        data = super()._sample_to_data(sample)

        # Noise has to be applied to the ABSOLUTE state before relativising,
        # not to the parent's agent_pos: that one is overwritten below, so
        # noising it would be a no-op and the augmentation would silently do
        # nothing. Noising afterwards is also wrong - it would break the
        # P_rel[t] == (0, I) invariant and perturb rotation columns that are
        # unit vectors, not metres.
        state_abs = np.asarray(sample['state'], dtype=np.float32)
        if self.use_noise_augmentation and self.agent_pos_noise_std > 0:
            state_abs = state_abs.copy()
            s = self.agent_pos_noise_std
            noise = np.random.normal(0.0, s, state_abs[:, POS_SLICE].shape)
            state_abs[:, POS_SLICE] += np.clip(
                noise, -2 * s, 2 * s).astype(np.float32)

        state_rel, action_rel = self._relativise(
            state_abs,
            np.asarray(sample['action'], dtype=np.float32),
        )
        data['obs']['agent_pos'] = state_rel
        data['action'] = action_rel
        return data

    # ------------------------------------------------------------------
    # normalizer
    # ------------------------------------------------------------------

    def get_normalizer(self, mode='limits', **kwargs):
        """Fit on RELATIVISED samples, not on the absolute zarr arrays.

        The parent fits straight off replay_buffer['action'], which here
        would compute statistics for a representation the policy never sees
        - absolute metres and quaternions instead of base-relative metres
        and 6D - and the normalised inputs would be wrong by a large factor.

        Sampling a subset rather than every window: the statistics are
        aggregate min/max, which converge long before 40k windows, and each
        window otherwise drags a decompressed point cloud with it.
        """
        n = min(self.normalizer_samples, len(self.sampler))
        # Deterministic: a normalizer that changes between runs makes two
        # training runs incomparable for reasons unrelated to the policy.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(self.sampler), size=n, replace=False) \
            if n < len(self.sampler) else np.arange(len(self.sampler))

        states, actions = [], []
        for i in idx:
            sample = self.sampler.sample_sequence(int(i))
            s, a = self._relativise(
                np.asarray(sample['state'], dtype=np.float32),
                np.asarray(sample['action'], dtype=np.float32),
            )
            states.append(s)
            actions.append(a)

        cprint(f"[RelativeRealDexDataset] normalizer fitted on {n} "
               f"relativised windows", "magenta")

        data = {
            'action': np.concatenate(actions, axis=0),
            'agent_pos': np.concatenate(states, axis=0),
            'point_cloud': self.replay_buffer['point_cloud'],
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        self._check_normalizer(normalizer, data)
        return normalizer

    def _check_normalizer(self, normalizer, data, eps=1e-6):
        """Assert the fitted statistics are usable. No patching.

        There is an obvious-looking failure mode here that turns out NOT to
        happen, and it is worth writing down so nobody 'fixes' it later:

        agent_pos[n_obs_steps-1] is exactly (0, I) in every window, so one
        might expect those 9 channels to have zero range and blow up the
        (max - min) division. They do not. The normalizer is fitted on all
        `horizon` rows of every window flattened together, and only ONE row
        per window is the base - the other 63 vary freely. Measured on
        foil_lifting_wrist_tesollo.zarr: zero constant channels, scale in
        0.029..6.02.

        Should that ever change (horizon == n_obs_steps, say), LinearNormalizer
        ._fit already covers it: range_eps=1e-4 substitutes a unit range and
        centres the offset, giving scale=1, offset=0 (verified in
        tests/_peek_normalizer.py). Re-implementing that here would be
        duplicated logic that silently diverges from upstream, so this method
        only CHECKS the outcome.
        """
        for key in ('agent_pos', 'action', 'point_cloud'):
            params = normalizer.params_dict[key]
            for name in ('scale', 'offset'):
                t = params[name]
                if not bool(torch.isfinite(t).all()):
                    raise AssertionError(
                        f"normalizer {key}.{name} is non-finite - a "
                        f"zero-range channel was not neutralised")
            if bool((params['scale'] == 0).any()):
                raise AssertionError(
                    f"normalizer {key}.scale contains zeros, which would "
                    f"make unnormalize() lose the channel entirely")

        # Expected to be empty (see docstring); logged rather than asserted
        # because a constant channel is handled correctly either way.
        arr = np.asarray(data['agent_pos'])
        constant = np.flatnonzero(
            (arr.max(axis=0) - arr.min(axis=0)) < eps).tolist()
        cprint(f"[RelativeRealDexDataset] constant agent_pos channels: "
               f"{constant} (expected: none)", "magenta")

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._sample_to_data(self.sampler.sample_sequence(idx))
        return dict_apply(data, torch.from_numpy)