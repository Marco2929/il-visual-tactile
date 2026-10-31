"""Standalone proprioception encoder for point-cloud-free configurations.

Why this exists: `DP3Encoder` houses the proprioception MLP alongside the
point-cloud extractor, but the two are not coupled -- with
`separate_state=True` the encoder returns `state_feat` as an independent
tensor and the policy routes it into the DiTX *input* (concatenated onto the
noisy action), never into the cross-attention context. See
`DP3Encoder.forward`:

    if self.separate_state:
        return pn_feat, state_feat, pc_pe   # state NOT merged into pn_feat

So the only thing tying proprioception to the point cloud is that the MLP is
physically declared inside `DP3Encoder`, which hard-requires a `point_cloud`
key in shape_meta. For an RGB-only ablation that requirement is the sole
blocker.

This module reproduces that MLP EXACTLY -- same `create_mlp` helper, same
`state_mlp_size` default, same trailing `Dropout(p=0.3)`, same
`state_feat_dim` attribute name -- so the proprioception path is bit-identical
whether or not a point cloud is configured. That is a hard requirement for the
modality ablation: if the state path differed between conditions, a measured
difference could not be attributed to the visual modality.

The 0.3 dropout is NOT the same mechanism as the policy's
`proprio_dropout_prob`:
  * this one drops individual FEATURES inside the MLP, always, as
    regularisation;
  * `proprio_dropout_prob` zeroes the ENTIRE state vector for a fraction of
    samples, to stop the policy solving the task from proprioception alone.
Both are kept, matching the point-cloud path.
"""

import torch.nn as nn
from termcolor import cprint

# Imported rather than re-implemented: a divergent MLP builder here would
# silently change layer count/ordering relative to the point-cloud path and
# break the equivalence this class exists to guarantee.
from maniflow.model.vision_3d.pointnet_extractor import create_mlp


class StateEncoder(nn.Module):
    """Encode `agent_pos` into the state features DiTX concatenates on input.

    Mirrors the `state_mlp` branch of `DP3Encoder` for configurations without
    a point cloud.

    Args:
        observation_space: dict of obs key -> shape, as built by the policy
            (`dict_apply(obs_shape_meta, lambda x: x['shape'])`).
        state_mlp_size: hidden/output widths. The last entry is the output
            dim; earlier entries are hidden layers. Default matches
            `DP3Encoder`.
        state_mlp_activation_fn: activation between layers.
        state_key: shape_meta key holding proprioception.
    """

    def __init__(self,
                 observation_space: dict,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 state_key='agent_pos',
                 ):
        super().__init__()

        self.state_key = state_key
        if state_key not in observation_space:
            raise KeyError(
                f"StateEncoder requires obs key '{state_key}' in shape_meta "
                f"(available: {sorted(observation_space)})")
        self.state_shape = observation_space[state_key]

        # Same net_arch derivation as DP3Encoder: last entry is the output
        # width, everything before it is hidden.
        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = list(state_mlp_size[:-1])
        output_dim = state_mlp_size[-1]

        self.state_feat_dim = output_dim

        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch,
                        state_mlp_activation_fn),
            nn.Dropout(p=0.3),
        )

        cprint(f"[StateEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[StateEncoder] State MLP size: {tuple(state_mlp_size)} "
               f"-> state_dim={output_dim}", "yellow")
        cprint("[StateEncoder] no point cloud configured; proprioception "
               "path matches the DP3Encoder state_mlp exactly", "yellow")

    def forward(self, observations: dict):
        """(B*T, state_dim_raw) -> (B*T, state_feat_dim).

        Returns only state features. Callers must treat `vis_cond` as None in
        this configuration; there is no visual context from this module.
        """
        return self.state_mlp(observations[self.state_key])

    def output_shape(self):
        """No cross-attention context is produced without a point cloud."""
        return 0