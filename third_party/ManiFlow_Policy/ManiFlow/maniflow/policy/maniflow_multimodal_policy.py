from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.model_util import print_params
from maniflow.model.vision_3d.pointnet_extractor import DP3Encoder
from maniflow.model.common.state_encoder import StateEncoder
from maniflow.model.vision_2d.transformer_obs_encoder import TransformerObsEncoder
from maniflow.model.vision_2d.dinov3_obs_encoder import DINOv3ObsEncoder
from maniflow.model.diffusion.ditx import DiTX
import torch.nn as nn
from maniflow.model.common.sample_util import *

class ManiFlowMultimodalPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=256,
            diffusion_target_t_embed_dim=256,
            visual_cond_len=1024,
            n_layer=3,
            n_head=4,
            n_emb=256,
            qkv_bias=False,
            qk_norm=False,
            block_type="DiTX",
            language_conditioned=False,
            # Top vision
            encoder_type="DP3Encoder",
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            downsample_points=False,
            # Whether the point cloud is used at all. False builds NO
            # DP3Encoder and leaves the cross-attention context to the RGB
            # token group alone (vis_cond=None); proprioception then runs
            # through a standalone StateEncoder that reproduces DP3Encoder's
            # state_mlp exactly, so the state path is identical either way.
            #
            # Default True: every pre-existing config omits this key and must
            # keep its current behaviour and checkpoint compatibility.
            pc_conditioned=True,
            # PC history length; defaults to n_obs_steps.
            pc_obs_steps=None,
            fps_random_config=None,
            pre_norm_modality=False,
            # RGB vision (wrist and/or front). The `wrist_*` names are kept
            # for backward compatibility with existing configs/checkpoints;
            # the encoder itself is multi-camera.
            wrist_conditioned=False,
            wrist_encoder_cfg: dict = None,
            # Which obs keys feed the RGB encoder. None = every key in
            # shape_meta with type 'rgb'. Order is always sorted internally.
            rgb_keys=None,
            # 'dinov3' -> DINOv3ObsEncoder (multi-camera, per-key resolution)
            # 'transformer' -> legacy TransformerObsEncoder (single resolution)
            rgb_encoder_type='dinov3',
            # consistency flow training parameters
            flow_batch_ratio=0.75,
            consistency_batch_ratio=0.25,
            denoise_timesteps=10,
            sample_t_mode_flow="beta", 
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform", 
            sample_target_t_mode="relative", # relative, absolute
            proprio_dropout_prob=0.1,
            **kwargs):
        super().__init__()

        self.wrist_conditioned = wrist_conditioned
        self.pc_obs_steps = pc_obs_steps if pc_obs_steps is not None else n_obs_steps
        self.proprio_dropout_prob = proprio_dropout_prob

        if self.proprio_dropout_prob:
            cprint(f"[Policy] Dropout enabled", "cyan")

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # create observation encoder
        self.encoder_type = encoder_type
        # Resolved rather than trusted: a config may ask for the point cloud
        # while shape_meta has no such key (or vice versa). Erroring here is
        # better than the KeyError deep inside DP3Encoder.__init__.
        pc_in_shape_meta = 'point_cloud' in obs_shape_meta
        if pc_conditioned and not pc_in_shape_meta:
            raise ValueError(
                "pc_conditioned=True but shape_meta['obs'] has no "
                "'point_cloud' key. Either add it to the task config or set "
                "policy.pc_conditioned=false for an RGB-only run.")
        self.pc_conditioned = pc_conditioned

        if encoder_type != "DP3Encoder":
            raise ValueError(f"Unsupported encoder type {encoder_type}")

        if pc_conditioned:
            obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                    img_crop_shape=crop_shape,
                                                    out_channel=encoder_output_dim,
                                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                    use_pc_color=use_pc_color,
                                                    pointnet_type=pointnet_type,
                                                    downsample_points=downsample_points,
                                                    fps_random_config=fps_random_config,
                                                    separate_state=True,
                                                    )
            cprint(f"[Encoder_type] {encoder_type}", "yellow")
        else:
            # No point cloud: proprioception only. state_mlp_size is read from
            # the same place DP3Encoder reads it, so the state path stays
            # identical across ablation conditions.
            state_mlp_size = (64, 64)
            if pointcloud_encoder_cfg is not None and hasattr(
                    pointcloud_encoder_cfg, 'get'):
                cfg_mlp = pointcloud_encoder_cfg.get('state_mlp_size', None)
                if cfg_mlp is not None:
                    state_mlp_size = tuple(cfg_mlp)
            obs_encoder = StateEncoder(observation_space=obs_dict,
                                       state_mlp_size=state_mlp_size)
            if not self.wrist_conditioned:
                raise ValueError(
                    "pc_conditioned=False and wrist_conditioned=False leaves "
                    "the policy with no visual input at all. Enable at least "
                    "one RGB camera.")
            cprint("[Encoder_type] StateEncoder (point cloud disabled)",
                   "yellow")

        # ====== RGB encoder (own token group in cross-attention) ======
        # One token group covering every configured camera. DiTX projects it
        # with Linear(wrist_dim, n_emb) and concatenates onto the context, so
        # the token COUNT is free - only the feature dim has to be declared.
        self.rgb_keys = []
        if self.wrist_conditioned:
            assert wrist_encoder_cfg is not None, "wrist_encoder_cfg required when wrist_conditioned=True"

            if rgb_keys is None:
                # Sorted so the token group order is deterministic; it is baked
                # into every checkpoint trained against it.
                resolved_keys = sorted(
                    k for k, a in obs_shape_meta.items()
                    if a.get('type', 'low_dim') == 'rgb'
                )
            else:
                resolved_keys = sorted(rgb_keys)
            missing = [k for k in resolved_keys if k not in obs_shape_meta]
            assert not missing, (
                f"rgb_keys {missing} are not in shape_meta['obs'] "
                f"(available: {sorted(obs_shape_meta)})")
            assert resolved_keys, (
                "wrist_conditioned=True but shape_meta['obs'] has no key with "
                "type: rgb")
            self.rgb_keys = resolved_keys

            rgb_shape_meta = {
                'obs': {k: dict(obs_shape_meta[k]) for k in resolved_keys}
            }

            if rgb_encoder_type == 'dinov3':
                self.wrist_encoder = DINOv3ObsEncoder(
                    shape_meta=rgb_shape_meta, **wrist_encoder_cfg)
            elif rgb_encoder_type == 'transformer':
                # Legacy path: asserts a single shared resolution across all
                # rgb keys, so it only works for one camera (or several at
                # identical resolution).
                self.wrist_encoder = TransformerObsEncoder(
                    shape_meta=rgb_shape_meta, **wrist_encoder_cfg)
            else:
                raise ValueError(
                    f"Unsupported rgb_encoder_type {rgb_encoder_type!r}; "
                    f"choose 'dinov3' or 'transformer'")

            # The encoder already returns (B, N_tokens, dim); tokens include
            # the time axis. Read the real shape instead of inferring it.
            _, self.wrist_num_tokens, self.wrist_dim = self.wrist_encoder.output_shape()
            cprint(f"[Policy] RGB conditioning {self.rgb_keys} "
                   f"({rgb_encoder_type}): {self.wrist_num_tokens} tokens x "
                   f"{self.wrist_dim} dim", "cyan")


        # ====== Dimensions: PC only for context, state goes into input ======
        # PC features are the cross-attention context; state is concatenated
        # onto the model input instead (see DiTX state_cond_dim).
        obs_feature_dim = obs_encoder.output_shape()
        state_feat_dim = obs_encoder.state_feat_dim
        self.state_feat_dim = state_feat_dim

        input_dim = action_dim
        # DiTX builds Linear(cond_dim, n_emb) for the PC group unconditionally,
        # and nn.Linear rejects in_features=0. Pass 1 as an inert placeholder
        # when there is no point cloud: vis_cond is None at runtime, so that
        # layer is never called (see the guard in DiTX.forward).
        global_cond_dim = obs_feature_dim if pc_conditioned else 1

        cprint(f"[Policy] PC dim (context): {obs_feature_dim}"
               f"{'' if pc_conditioned else ' (disabled)'}", "green")
        cprint(f"[Policy] State dim (input concat): {state_feat_dim}", "green")
        cprint(f"[Policy] Action dim: {action_dim}", "green")
        # ===================================================================

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[ManiFlowMultimodalPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[ManiFlowMultimodalPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        # Hint: ensure encoder_output_dim matches Uni3D output dimension
        if pointnet_type in ["uni3d", "uni3d_pretrained"]:
            cprint(f"[DP3] Uni3D encoder detected, ensure encoder_output_dim matches Uni3D output dim", "cyan")

        
        cprint(f"[ManiFlowMultimodalPolicy] Using DiTX model", "red")
        model = DiTX(
            input_dim=input_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=global_cond_dim,
            state_cond_dim=state_feat_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            pre_norm_modality=pre_norm_modality,
            language_conditioned=language_conditioned,
            wrist_conditioned=wrist_conditioned,
            wrist_dim=self.wrist_dim if wrist_conditioned else 0,
            pc_conditioned=pc_conditioned,
        )
        
        self.obs_encoder = obs_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert flow_batch_ratio + consistency_batch_ratio == 1.0, "Sum of batch ratios should be equal to 1.0"
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_t_mode_consistency = sample_t_mode_consistency
        self.sample_dt_mode_consistency = sample_dt_mode_consistency
        self.sample_target_t_mode = sample_target_t_mode
        assert self.sample_target_t_mode in ["absolute", "relative"], "sample_target_t_mode must be either 'absolute' or 'relative'"


        cprint(f"[ManiFlowMultimodalPolicy] Initialized with parameters:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        cprint(f"  - flow_batch_ratio: {self.flow_batch_ratio}", "yellow")
        cprint(f"  - consistency_batch_ratio: {self.consistency_batch_ratio}", "yellow")
        cprint(f"  - denoise_timesteps: {self.denoise_timesteps}", "yellow")
        cprint(f"  - sample_t_mode_flow: {self.sample_t_mode_flow}", "yellow")
        cprint(f"  - sample_t_mode_consistency: {self.sample_t_mode_consistency}", "yellow")
        cprint(f"  - sample_dt_mode_consistency: {self.sample_dt_mode_consistency}", "yellow")
        cprint(f"  - sample_target_t_mode: {self.sample_target_t_mode}", "yellow")

        print_params(self)

    def _encode_obs(self, nobs, batch_size):
        """Encode observations, returning separate PC, state, and (optional) spatial PE.

        Returns (vis_cond, state_cond, pc_pe) where pc_pe is None for the
        global-feature encoders (pointnet) and a spatial positional encoding
        for pointwise encoders (pointsam / Uni3D).
        """
        # PC and state share one encoder call, so both use the same slice.
        To_pc = self.pc_obs_steps
        this_nobs = dict_apply(nobs,

            lambda x: x[:, :To_pc, ...].reshape(-1, *x.shape[2:]).to(self.device))

        # ---- No point cloud: state only, no cross-attention context ----
        if not self.pc_conditioned:
            state_feat = self.obs_encoder(this_nobs)
            state_feat = state_feat.reshape(batch_size, To_pc, -1)
            # Same as the PC path below: take the last obs step and broadcast
            # across the action horizon.
            state_cond = state_feat[:, -1:, :].expand(-1, self.horizon, -1)
            return None, state_cond, None
        # ----------------------------------------------------------------

        encoder_out = self.obs_encoder(this_nobs)

        # Handle tuple returns
        pc_pe = None
        if isinstance(encoder_out, tuple) and len(encoder_out) == 3:
            # pointsam mode: PC features + state + spatial positional encoding
            pc_feat, state_feat, pc_pe = encoder_out
        elif isinstance(encoder_out, tuple) and len(encoder_out) == 2:
            pc_feat, state_feat = encoder_out
        else:
            raise ValueError(f"Expected tuple from encoder with separate_state=True, got {type(encoder_out)}")

        # pc_feat: (B*n_obs, L, pc_dim) or (B*n_obs, pc_dim)
        # state_feat: (B*n_obs, state_dim)

        # Reshape PC features
        if len(pc_feat.shape) == 2:
            # Global feature: (B*n_obs, pc_dim) → (B, n_obs, pc_dim)

            vis_cond = pc_feat.reshape(batch_size, To_pc, -1)
        else:
            # Pointwise/patch: (B*n_obs, L, pc_dim) → (B, n_obs*L, pc_dim)
            vis_cond = pc_feat.reshape(batch_size, -1, self.obs_feature_dim)

        # Reshape state features: (B*n_obs, state_dim) → (B, n_obs, state_dim)

        state_feat = state_feat.reshape(batch_size, To_pc, -1)

        # Take last obs step and broadcast to horizon
        # (B, n_obs, state_dim) → (B, 1, state_dim) → (B, horizon, state_dim)
        state_cond = state_feat[:, -1:, :].expand(-1, self.horizon, -1)

        # Reshape pc_pe to match vis_cond token layout: (B*n_obs, L, pe_dim)
        # → (B, n_obs*L, pe_dim), mirroring the pointwise vis_cond reshape.
        if pc_pe is not None:
            pc_pe = pc_pe.reshape(batch_size, -1, pc_pe.shape[-1])

        return vis_cond, state_cond, pc_pe
        
    def _encode_wrist(self, nobs, batch_size):
        """Encode all RGB cameras into one token group (B, N, wrist_dim).

        The RGB encoder produces its own token group, appended to the DiTX
        cross-attention context alongside the point-cloud tokens. Inputs are
        NOT normalized here (the encoder does /255 + model-specific norm).

        Layout: the zarr stores images channels-LAST (T, H, W, C) to keep them
        uint8 and compact, while DINOv3ObsEncoder requires channels-first.
        This method is the single boundary between the two, so the permute
        happens here rather than being sniffed from the channel count inside
        the encoder.
        """
        obs = {}
        for key in self.rgb_keys:
            img = nobs[key][:, :self.n_obs_steps, ...].to(self.device)
            if img.ndim != 5:
                raise ValueError(
                    f"obs key '{key}': expected (B, T, H, W, C) or "
                    f"(B, T, C, H, W), got {tuple(img.shape)}")
            # (B, T, H, W, C) -> (B, T, C, H, W). Detected on the last axis
            # being the channel count rather than a hardcoded 3, so a
            # 2-channel input (e.g. depth) would route identically.
            if img.shape[-1] <= 4 and img.shape[2] > 4:
                img = img.permute(0, 1, 4, 2, 3)
            obs[key] = img
        return self.wrist_encoder(obs)

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            vis_cond=None,
            state_cond=None,
            lang_cond=None,
            wrist_cond=None,
            rtc_processor=None,
            prev_chunk_left_over=None,
            inference_delay=0,
            **kwargs
            ):
        
        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)
        
        ode_traj = self.sample_ode(
            x0 = noise, 
            N = self.num_inference_steps,
            rtc_processor=rtc_processor,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay,
            vis_cond=vis_cond,
            state_cond=state_cond,
            lang_cond=lang_cond,
            wrist_cond=wrist_cond,
           **kwargs)
        
        return ode_traj[-1] # sample ode returns the whole traj, return the last one
    


    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       rtc_processor=None, prev_chunk_left_over=None,
                       inference_delay=0) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key

        RTC args (optional):
            rtc_processor: RTCProcessor instance for prefix guidance.
            prev_chunk_left_over: (B, Tp, Da) unexecuted tail of previous chunk in
                                  NORMALIZED action space (full horizon actions).
            inference_delay: int, steps consumed during inference.
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # Guarded on pc_conditioned: with the point cloud disabled there is no
        # 'point_cloud' key to slice, and an unguarded index would KeyError on
        # every step of the RGB-only conditions.
        if self.pc_conditioned and not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        lang_cond = None
        wrist_cond = None

        if self.language_conditioned:
            # assume nobs has 'task_name' key for language condition
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        vis_cond, state_cond, pc_pe = self._encode_obs(nobs, B)

        # Add wrist-RGB conditioning (own token group, no spatial fusion)
        if self.wrist_conditioned:
            wrist_cond = self._encode_wrist(nobs, B)

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # Normalize prev_chunk_left_over for RTC guidance (runs in normalized space)
        prev_chunk_norm = None
        if rtc_processor is not None and prev_chunk_left_over is not None:
            # prev_chunk_left_over: (B, Tp, Da) in UNNORMALIZED action space
            prev_chunk_norm = self.normalizer['action'].normalize(prev_chunk_left_over).to(device)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            vis_cond=vis_cond,
            state_cond=state_cond,
            lang_cond=lang_cond,
            wrist_cond=wrist_cond,
            pc_pe=pc_pe,
            rtc_processor=rtc_processor,
            prev_chunk_left_over=prev_chunk_norm,
            inference_delay=inference_delay,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action_full = action_pred[:,start:end]

        result = {
            'action': action_full,
            'action_pred': action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float = None,
            obs_encoder_weight_decay: float = None,
            wrist_encoder_lr: float = None,
            betas: Tuple[float, float] = (0.9, 0.95)
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        # The obs_encoder MUST always be added, otherwise the point-cloud
        # encoder silently never trains. obs_encoder_lr only changes which
        # lr it gets, not whether it is optimized at all.
        if obs_encoder_lr is not None:
            cprint(f"[Policy] Separate lr for obs_encoder backbone: {obs_encoder_lr}", "yellow")
            backbone_params = list()
            other_obs_params = list()
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay if obs_encoder_weight_decay is not None else weight_decay,
                "lr": obs_encoder_lr # for fine tuning
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay if obs_encoder_weight_decay is not None else weight_decay,
            })
        else:
            optim_groups.append({
                "params": list(self.obs_encoder.parameters()),
                "weight_decay": weight_decay,
            })

        if self.wrist_conditioned:
            # Only trainable params. With a frozen DINOv3 trunk the vast
            # majority have requires_grad=False; handing those to AdamW would
            # allocate optimizer state (exp_avg + exp_avg_sq, i.e. 2x the
            # parameter memory) for tensors that never receive a gradient.
            rgb_params = [p for p in self.wrist_encoder.parameters()
                          if p.requires_grad]
            if rgb_params:
                # A pretrained backbone degrades quickly at the policy's base
                # lr; give it its own (lower) lr when configured.
                wrist_group = {
                    "params": rgb_params,
                    "weight_decay": weight_decay,
                }
                if wrist_encoder_lr is not None:
                    wrist_group["lr"] = wrist_encoder_lr
                    cprint(f"[Policy] RGB encoder lr: {wrist_encoder_lr}", "yellow")
                optim_groups.append(wrist_group)

            n_train, n_frozen = (
                self.wrist_encoder.count_parameters()
                if hasattr(self.wrist_encoder, "count_parameters")
                else (sum(p.numel() for p in rgb_params), -1))
            cprint(f"[Policy] RGB encoder params: {n_train:,} trainable / "
                   f"{n_frozen:,} frozen", "yellow")

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

        # Guard against a group being dropped: every trainable parameter must
        # be in exactly one group, or it silently never updates. Compare by
        # identity so the error names the offenders instead of just a count.
        # ModuleAttrMixin._dummy_variable is an empty Parameter that exists
        # only to expose .device/.dtype, so it is intentionally excluded.
        covered = {id(p) for g in optim_groups for p in g["params"]}
        missing = [
            name for name, p in self.named_parameters()
            if p.requires_grad
            and id(p) not in covered
            and not name.endswith("_dummy_variable")
        ]
        assert not missing, (
            f"these trainable params are in no optimizer group and would "
            f"never update: {missing}")
        cprint(f"[Policy] Optimizer: {len(optim_groups)} groups, "
               f"{len(covered)} params", "yellow")

        return optimizer
    
    def sample_t(self, batch_size, mode="uniform"):
        """
        Sample t for flow matching or consistency training.

        When `self.eval_deterministic` is set (validation), t/dt/noise are drawn
        from a fixed-seed generator so the reported loss is comparable across
        epochs. Otherwise the loss is dominated by (t, noise) sampling variance
        rather than by actual learning progress.
        """
        gen = getattr(self, '_eval_generator', None)
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device, generator=gen)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        elif mode == "discrete":
            if gen is not None:
                t = torch.randint(low=0, high=self.denoise_timesteps,
                                  size=(batch_size,), generator=gen,
                                  device=gen.device).float().to(self.device)
            else:
                t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,)).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(f" Unsupported sample_t_mode {mode}. Choose from 'uniform', 'cosmap', 'beta', 'discrete'.")
        return t

    def enable_deterministic_eval(self, seed=12345):
        """Pin (t, dt, noise) sampling to a fixed seed for comparable val loss."""
        self._eval_generator = torch.Generator(device=self.device)
        self._eval_generator.manual_seed(seed)

    def disable_deterministic_eval(self):
        self._eval_generator = None

    def _randn_like(self, ref, device):
        """randn_like that honours the deterministic-eval generator.

        The noise draw dominates flow-loss variance, so this must be pinned
        too — pinning only t/dt is not enough to make val loss comparable
        across epochs.
        """
        gen = getattr(self, '_eval_generator', None)
        if gen is None:
            return torch.randn_like(ref, device=device)
        return torch.randn(ref.shape, device=device, dtype=ref.dtype, generator=gen)

    def sample_dt(self, batch_size, sample_dt_mode="uniform"):
        """
        Sample dt for consistency training.
        """
        gen = getattr(self, '_eval_generator', None)
        if sample_dt_mode == "uniform":
            dt = torch.rand((batch_size,), device=self.device, generator=gen)
        else:
            raise ValueError(f"Unsupported sample_dt_mode {sample_dt_mode}")
        
        return dt
    
    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        """
        Linear interpolation between noise and target data with optional noise preservation.
        """
        # Calculate noise coefficient with epsilon adjustment
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        
        # Linear combination: preserved_noise + scaled_target
        interpolated_data_point = noise_coeff * noise + timestep * target
        
        return interpolated_data_point

    
    def get_flow_velocity(self, actions, **model_kwargs):
        """
        Get flow velocity targets for training.
        Flow training is used to train the model to predict instantaneous velocity given a timestep.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        state_cond = model_kwargs.get('state_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        wrist_cond = model_kwargs.get('wrist_cond', None)
        flow_batchsize = actions.shape[0]
        device = actions.device
        
        # sample t and dt for flow
        # dt is zero for flow, as we aim to predict the instantaneous velocity at t
        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((flow_batchsize,), device=device)
        
        # get target timestep
        if self.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        elif self.sample_target_t_mode == "relative":
            target_t_flow = dt_flow
        
        # compute interpolated data points at t and predict flow velocity
        x_0_flow = self._randn_like(actions, device)
        x_1_flow = actions.to(device) 
        x_t_flow = self.linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['v_target'] = v_t_flow
        target_dict['state_cond'] = state_cond 
        target_dict['vis_cond'] = vis_cond
        target_dict['lang_cond'] = lang_cond
        target_dict['wrist_cond'] = wrist_cond

        return target_dict
    
    def get_consistency_velocity(self, actions, **model_kwargs):
        """
        Get consistency velocity targets for training.
        Consistency training is used to train the model to be consistent across different timesteps.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        state_cond = model_kwargs.get('state_cond', None)
        wrist_cond = model_kwargs.get('wrist_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        pc_pe = model_kwargs.get('pc_pe', None)
        ema_model = model_kwargs.get('ema_model', None)
        consistency_batchsize = actions.shape[0]
        device = actions.device

        # sample t and dt for consistency training
        t_ct = self.sample_t(consistency_batchsize, mode=self.sample_t_mode_consistency).to(device)
        t_ct = t_ct.view(-1, 1, 1)
        delta_t1 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        delta_t2 = delta_t1.clone() # use the same delta_t or resample a new one

        # compute next timestep
        t_next = t_ct.squeeze() + delta_t1
        t_next = torch.clamp(t_next, max=1.0) # clip t to ensure it does not exceed 1.0
        t_next = t_next.view(-1, 1, 1)
        
        # compute target timestep
        if self.sample_target_t_mode == "absolute":
            target_t_next = t_next.squeeze() + delta_t2
        elif self.sample_target_t_mode == "relative":
            target_t_next = delta_t2

        # compute interpolated data points at timestep t and t_next
        x0_ct = self._randn_like(actions, device) 
        x1_ct = actions.to(device) 
        x_t_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        # predict the average velocity from t_next toward next target (t_next + delta_t2)
        with torch.no_grad():
            v_avg_to_next_target = ema_model.model(
                sample=x_t_next, 
                timestep=t_next.squeeze(),
                target_t=target_t_next.squeeze(), 
                vis_cond=vis_cond[-consistency_batchsize:] if vis_cond is not None else None,
                state_cond=state_cond[-consistency_batchsize:] if state_cond is not None else None, 
                lang_cond=lang_cond[-consistency_batchsize:] if lang_cond is not None else None,
                wrist_cond=wrist_cond[-consistency_batchsize:] if wrist_cond is not None else None,
                pc_pe=pc_pe[-consistency_batchsize:] if pc_pe is not None else None,
            ) 
        # predict the target data point using the average velocity
        pred_x1_ct = x_t_next + (1 - t_next) * v_avg_to_next_target
        # estimate the velocity at t by using the predicted endpoint
        v_ct = (pred_x1_ct - x_t_ct) / (1 - t_ct)

        # target_t_ct is the target timestep for the current timestep t
        target_t_ct = delta_t1 if self.sample_target_t_mode == "relative" else t_next.squeeze()
        
        target_dict['x_t'] = x_t_ct
        target_dict['t'] = t_ct
        target_dict['target_t'] = target_t_ct
        target_dict['v_target'] = v_ct

        return target_dict
    
    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, rtc_processor=None,
                   prev_chunk_left_over=None, inference_delay=0, **model_kwargs):
        ### NOTE: Use Euler method to sample from the learned flow
        # RTC guidance: if rtc_processor and prev_chunk_left_over are provided,
        # each denoising step is guided so the new chunk continues the previous one.
        if N is None:
            N = self.num_inference_steps
        dt = 1./N
        traj = [] # to store the trajectory
        x = x0.detach().clone()
        batchsize = x.shape[0]

        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N
        traj.append(x.detach().clone())

        use_rtc = (rtc_processor is not None) and (prev_chunk_left_over is not None)

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            if self.sample_target_t_mode == "absolute":
                target_t = ti + dt
            elif self.sample_target_t_mode == "relative":
                target_t = dt

            if use_rtc:
                def v_fn(x_in, _ti=ti, _tt=target_t):
                    return self.model(x_in, _ti, target_t=_tt, **model_kwargs)
                pred = rtc_processor.guide(
                    x, float(t[i]), v_fn, prev_chunk_left_over, inference_delay)
            else:
                pred = self.model(x, ti, target_t=target_t, **model_kwargs)

            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    def compute_loss(self, batch, ema_model=None, **kwargs):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        # Guarded on pc_conditioned: with the point cloud disabled there is no
        # 'point_cloud' key to slice, and an unguarded index would KeyError on
        # every step of the RGB-only conditions.
        if self.pc_conditioned and not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]

        lang_cond = None
        wrist_cond = None
        ema_model = ema_model

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None

        if self.wrist_conditioned:
            wrist_cond = self._encode_wrist(nobs, batch_size)

        # ====== Encode: PC and State separately ======
        vis_cond, state_cond, pc_pe = self._encode_obs(nobs, batch_size)
        # =============================================

        # === Proprio dropout: zero out state_cond ===
        if self.training and self.proprio_dropout_prob > 0:
            drop_mask = (torch.rand(batch_size, 1, 1, device=state_cond.device) < self.proprio_dropout_prob)
            state_cond = state_cond.clone()
            state_cond = state_cond.masked_fill(drop_mask, 0.0)
        # =============================================

        # Split batch
        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)

        # vis_cond is None when the point cloud is disabled; the slices must
        # stay None rather than raising, mirroring pc_pe / wrist_cond below.
        vis_cond_flow = vis_cond[:flow_batchsize] if vis_cond is not None else None
        vis_cond_ct = (vis_cond[flow_batchsize:flow_batchsize + consistency_batchsize]
                       if vis_cond is not None else None)
        state_cond_flow = state_cond[:flow_batchsize]
        state_cond_ct = state_cond[flow_batchsize:flow_batchsize + consistency_batchsize]
        pc_pe_flow = pc_pe[:flow_batchsize] if pc_pe is not None else None
        pc_pe_ct = pc_pe[flow_batchsize:flow_batchsize + consistency_batchsize] if pc_pe is not None else None
        wrist_cond_flow = wrist_cond[:flow_batchsize] if wrist_cond is not None else None
        wrist_cond_ct = wrist_cond[flow_batchsize:flow_batchsize + consistency_batchsize] if wrist_cond is not None else None
        lang_cond_flow = lang_cond[:flow_batchsize] if lang_cond is not None else None
        lang_cond_ct = lang_cond[flow_batchsize:flow_batchsize + consistency_batchsize] if lang_cond is not None else None

        # Flow
        flow_target_dict = self.get_flow_velocity(
            nactions[:flow_batchsize],  
            vis_cond=vis_cond_flow,
            state_cond=state_cond_flow,
            lang_cond=lang_cond_flow,
            wrist_cond=wrist_cond_flow,
            pc_pe=pc_pe_flow,
        )
        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'],
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond_flow,
            state_cond=state_cond_flow,
            wrist_cond=wrist_cond_flow,
            lang_cond=lang_cond_flow,
            pc_pe=pc_pe_flow,
        )
        v_flow_pred_magnitude = torch.sqrt(torch.mean(v_flow_pred ** 2)).item()

        # Consistency
        consistency_target_dict = self.get_consistency_velocity(
            nactions[flow_batchsize:flow_batchsize + consistency_batchsize],
            vis_cond=vis_cond_ct,
            state_cond=state_cond_ct,
            lang_cond=lang_cond_ct,
            wrist_cond=wrist_cond_ct,
            pc_pe=pc_pe_ct,
            ema_model=ema_model,
        )
        v_ct_pred = self.model(
            sample=consistency_target_dict['x_t'],
            timestep=consistency_target_dict['t'].squeeze(),
            target_t=consistency_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond_ct,
            state_cond=state_cond_ct,
            wrist_cond=wrist_cond_ct,
            lang_cond=lang_cond_ct,
            pc_pe=pc_pe_ct,
        )
        v_ct_pred_magnitude = torch.sqrt(torch.mean(v_ct_pred ** 2)).item()

        # Losses
        v_flow_target = flow_target_dict['v_target']
        loss_flow = reduce(
            F.mse_loss(v_flow_pred, v_flow_target, reduction='none'),
            'b ... -> b (...)', 'mean').mean()

        v_ct_target = consistency_target_dict['v_target']
        loss_ct = reduce(
            F.mse_loss(v_ct_pred, v_ct_target, reduction='none'),
            'b ... -> b (...)', 'mean').mean()

        loss = loss_flow + loss_ct

        loss_dict = {
            'total_loss': loss.item(),
            'loss_flow': loss_flow.item(),
            'loss_ct': loss_ct.item(),
            'v_flow_pred_magnitude': v_flow_pred_magnitude,
            'v_ct_pred_magnitude': v_ct_pred_magnitude,
        }

        # Track whether the wrist encoder is actively learning.
        if self.wrist_conditioned and wrist_cond is not None:
            loss_dict['wrist_cond_magnitude'] = torch.sqrt(torch.mean(wrist_cond ** 2)).item()

        return loss, loss_dict