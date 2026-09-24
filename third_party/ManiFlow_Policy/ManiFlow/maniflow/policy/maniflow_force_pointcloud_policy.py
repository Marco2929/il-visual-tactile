from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.model.progress.progress_head import ProgressHead
from maniflow.model.force.force_extractor import force_mlp, force_cnn
from maniflow.common.model_util import print_params
from maniflow.model.vision_3d.pointnet_extractor import DP3Encoder
from maniflow.model.diffusion.ditx import DiTX
import torch.nn as nn
from maniflow.model.common.sample_util import *

class ManiFlowTransformerPointcloudPolicy(BasePolicy):
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
            encoder_type="DP3Encoder",
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            downsample_points=False,
            fps_random_config=None,
            pre_norm_modality=False,
            language_conditioned=False,
            use_progress=False,
            force_conditioned=False,
            wrist_conditioned=False,
            wrist_encoder_cfg=None,
            # Force
            aux_action=False,
            aux_force=False,
            force_cond_len=65,
            lambda_aux_ee=0.5,
            lambda_aux_force=0.3,
            lambda_progress=0.05,
            # consistency flow training parameters
            flow_batch_ratio=0.75,
            consistency_batch_ratio=0.25,
            denoise_timesteps=10,
            sample_t_mode_flow="beta", 
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform", 
            sample_target_t_mode="relative", # relative, absolute
            use_relative=False,
            relative_action_dims=3, 
            proprio_dropout_prob=0.1,
            **kwargs):
        super().__init__()

        self.force_conditioned = force_conditioned
        self.wrist_conditioned = wrist_conditioned
        self.aux_action = aux_action
        self.aux_force = aux_force
        self.lambda_aux_ee = lambda_aux_ee
        self.lambda_progress = lambda_progress
        self.lambda_aux_force = lambda_aux_force
        
        self.use_relative = use_relative
        self.relative_action_dims = relative_action_dims
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
        if encoder_type == "DP3Encoder":
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
        else:
            raise ValueError(f"Unsupported encoder type {encoder_type}")
        cprint(f"[Encoder_type] {encoder_type}", "yellow")

        self.encoder_type = encoder_type
        self.use_force_cnn = kwargs.get('use_force_cnn', False)
        
        # TODO: Cross check in papers if this makes actual sense
        if self.force_conditioned and self.use_force_cnn:
            self.force_encoder = force_cnn(force_cond_len=force_cond_len,
                                           encoder_output_dim=encoder_output_dim)
            cprint(f"[Policy] Force CNN enabled. Outputting tokens matching history length.", "cyan")

        elif self.force_conditioned:
            self.force_encoder = force_mlp(n_obs_steps=n_obs_steps, 
                                           force_cond_len=force_cond_len,
                                           encoder_output_dim=encoder_output_dim)
            
            cprint(f"[Policy] Force enabled (MLP): {n_obs_steps * force_cond_len} flat -> {encoder_output_dim}", "cyan")
        else:
            pass

        self.use_progress = use_progress

        # ====== Wrist-RGB encoder (own token group in cross-attention) ======
        # Follows the exact same pattern as force_cond: a separate encoder
        # produces conditioning tokens that are appended to the DiTX context.
        # Wrist-RGB is used for the CONTACT phase, which the front camera
        # cannot see due to physical occlusion (arm blocks the grasp point).
        if self.wrist_conditioned:
            from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
            assert wrist_encoder_cfg is not None, "wrist_encoder_cfg required when wrist_conditioned=True"
            wrist_shape_meta = {'obs': {'wrist': dict(obs_shape_meta['wrist_rgb'])}}
            self.wrist_encoder = TimmObsEncoder(shape_meta=wrist_shape_meta, **wrist_encoder_cfg)
            # Determine token layout: encoder returns (1, F_flat).
            # We reshape to (B, N_tokens, wrist_dim). N_tokens = n_obs_steps
            # for option A (1 token/frame). For option B (all_tokens) the
            # encoder returns more tokens per frame; wrist_dim is inferred.
            wrist_out_flat = self.wrist_encoder.output_shape()[-1]
            self.wrist_num_tokens = n_obs_steps
            assert wrist_out_flat % n_obs_steps == 0, (
                f"Wrist encoder output {wrist_out_flat} not divisible by n_obs_steps={n_obs_steps}")
            self.wrist_dim = wrist_out_flat // n_obs_steps
            cprint(f"[Policy] Wrist-RGB enabled: {self.wrist_num_tokens} tokens x {self.wrist_dim} dim", "cyan")


        # ====== Dimensions: PC only for context, state goes into input ======
        pc_feature_dim = obs_encoder.output_shape()  # PC only (e.g. 384)
        state_feat_dim = obs_encoder.state_feat_dim   # State only (e.g. 64)
        self.state_feat_dim = state_feat_dim

        input_dim = action_dim  # action only (state is added via state_cond_dim)
        global_cond_dim = pc_feature_dim  # PC only in context

        cprint(f"[Policy] PC dim (context): {pc_feature_dim}", "green")
        cprint(f"[Policy] State dim (input concat): {state_feat_dim}", "green")
        cprint(f"[Policy] Action dim: {action_dim}", "green")
        # ===================================================================

        # create ManiFlow model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim
        

        if self.use_progress:
            self.progress_head = ProgressHead(visual_feat_dim=obs_feature_dim,
                                              use_force=self.force_conditioned)
            cprint(f"[Policy] Progress Head enabled (vision-only).", "cyan")

        # === Vision-only auxiliary action head (anti-proprio-shortcut) ===
        # Predicts the FULL action from vision ALONE (no proprioception),
        # forcing the point-cloud encoder to keep task-relevant spatial
        # information alive instead of collapsing to a proprio shortcut.
        self.use_vision_aux = kwargs.get('use_vision_aux', False)
        self.lambda_vision_aux = kwargs.get('lambda_vision_aux', 0.5)
        if self.use_vision_aux:
            from maniflow.model.progress.vision_action_head import VisionActionHead
            self.vision_action_head = VisionActionHead(
                visual_feat_dim=obs_feature_dim,
                action_dim=action_dim,
                horizon=horizon)
            cprint(f"[Policy] Vision-Aux action head enabled "
                   f"(lambda={self.lambda_vision_aux}) - forces vision to stay alive.", "magenta")


        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[ManiFlowTransformerPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[ManiFlowTransformerPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        # Hint: ensure encoder_output_dim matches Uni3D output dimension
        if pointnet_type in ["uni3d", "uni3d_pretrained"]:
            cprint(f"[DP3] Uni3D encoder detected, ensure encoder_output_dim matches Uni3D output dim", "cyan")

        
        cprint(f"[ManiFlowTransformerPointcloudPolicy] Using DiTX model", "red")
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
            force_conditioned=force_conditioned,
            wrist_conditioned=wrist_conditioned,
            wrist_dim=self.wrist_dim if wrist_conditioned else 0,
        )
        
        self.obs_encoder = obs_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = pc_feature_dim
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

        self.aux_ee_dim = 7 if aux_action else 0
        self.aux_force_dim = shape_meta.get('aux_force_dim', 0) if aux_force else 0
        self.primary_action_dim = action_dim - self.aux_ee_dim - self.aux_force_dim

        assert self.primary_action_dim > 0, (
            f"primary_action_dim={self.primary_action_dim} must be > 0. "
            f"action_dim={action_dim}, aux_ee={self.aux_ee_dim}, aux_force={self.aux_force_dim}"
        )
        
        if self.use_relative:
            cprint(f"[Policy] RELATIVE ACTIONS (mimic-one style): EE dims 0:{self.relative_action_dims} relative, "
                   f"hand dims {self.relative_action_dims}: absolute", "green", attrs=["bold"])
        cprint(f"- absolute/relative: {self.use_relative} "
            f"action splits: primary={self.primary_action_dim} "
               f"(joints+hand or ee+hand), ee_aux={self.aux_ee_dim}, "
               f"force_aux={self.aux_force_dim}, total={action_dim}", "cyan")

        cprint(f"[ManiFlowTransformerPointcloudPolicy] Initialized with parameters:", "yellow")
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
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]).to(self.device))

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
            vis_cond = pc_feat.reshape(batch_size, self.n_obs_steps, -1)
        else:
            # Pointwise/patch: (B*n_obs, L, pc_dim) → (B, n_obs*L, pc_dim)
            vis_cond = pc_feat.reshape(batch_size, -1, self.obs_feature_dim)

        # Reshape state features: (B*n_obs, state_dim) → (B, n_obs, state_dim)
        state_feat = state_feat.reshape(batch_size, self.n_obs_steps, -1)

        # Take last obs step and broadcast to horizon
        # (B, n_obs, state_dim) → (B, 1, state_dim) → (B, horizon, state_dim)
        state_cond = state_feat[:, -1:, :].expand(-1, self.horizon, -1)

        # Reshape pc_pe to match vis_cond token layout: (B*n_obs, L, pe_dim)
        # → (B, n_obs*L, pe_dim), mirroring the pointwise vis_cond reshape.
        if pc_pe is not None:
            pc_pe = pc_pe.reshape(batch_size, -1, pc_pe.shape[-1])

        return vis_cond, state_cond, pc_pe
        
    def _encode_wrist(self, nobs, batch_size):
        """Encode wrist-RGB into conditioning tokens (B, N, wrist_dim).

        Mirrors the force_cond pathway: the wrist encoder produces its own
        token group that is appended to the DiTX cross-attention context.
        Input nobs['wrist_rgb'] has shape (B, T, H, W, C) and is NOT
        normalized (the image encoder does /255 itself).
        """
        wrist_rgb = nobs['wrist_rgb'][:, :self.n_obs_steps, ...].to(self.device)
        # TimmObsEncoder expects obs_dict keyed by the rgb key ('wrist').
        wrist_flat = self.wrist_encoder({'wrist': wrist_rgb})  # (B, F_flat)
        # Reshape to token layout (B, N, wrist_dim).
        wrist_cond = wrist_flat.reshape(batch_size, self.wrist_num_tokens, self.wrist_dim)
        return wrist_cond

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            vis_cond=None,
            state_cond=None,
            lang_cond=None,
            force_cond=None,
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
            force_cond=force_cond,
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
        # ===== Make observation relative before normalizing =====
        # The normalizer was fit on RELATIVE data, so we must convert absolute
        # observations to relative space before normalization.
        if self.use_relative:
            d = self.relative_action_dims
            # Reference: last obs step's position (mirrors training: ref_idx = pad_before)
            current_pos_abs = obs_dict['agent_pos'][:, -1:, :d].clone()  # (B, 1, d)
            
            # Create relative obs_dict (don't modify original in-place)
            obs_dict_for_norm = {}
            for k, v in obs_dict.items():
                if k == 'agent_pos':
                    agent_pos_rel = v.clone()
                    agent_pos_rel[..., :d] = agent_pos_rel[..., :d] - current_pos_abs
                    obs_dict_for_norm[k] = agent_pos_rel
                else:
                    obs_dict_for_norm[k] = v
        else:
            obs_dict_for_norm = obs_dict
            current_pos_abs = None
        # =========================================================

        # normalize input
        nobs = self.normalizer.normalize(obs_dict_for_norm)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        lang_cond = None
        force_cond = None
        wrist_cond = None

        if self.language_conditioned:
            # assume nobs has 'task_name' key for language condition
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        vis_cond, state_cond, pc_pe = self._encode_obs(nobs, B)
        
        # TODO: Fix this that it works with both versions
        # Add force here
        if self.force_conditioned:
            # nobs['hand_obs'] hat durch die YAML jetzt die Form: (Batch, 16, 65)
            hand_obs = nobs['hand_obs']
            
            # Umordnen für das CNN: (Batch, Channels, Time) -> (Batch, 65, 16)
            hand_obs = hand_obs.permute(0, 2, 1)
            
            # CNN anwenden. Output bleibt (Batch, encoder_output_dim, 16)
            force_features = self.force_encoder(hand_obs)
            
            # Jetzt ordnen wir es für den Transformer um: (Batch, Time, Channels) -> (Batch, 16, 128)
            # Das bedeutet: Wir haben jetzt 16 echte Force-Tokens!
            force_cond = force_features.permute(0, 2, 1)

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
            force_cond=force_cond,
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

        # ===== Relative → Absolute =====
        if self.use_relative:
            d = self.relative_action_dims  # 3
            # current_pos_abs was saved before normalization (B, 1, d)
            ref = current_pos_abs.squeeze(1)  # (B, d)
            action_full = action_full.clone()
            action_full[..., :d] = action_full[..., :d] + ref.unsqueeze(1)
        # ================================
        
        result = {
            'action': action_full[..., :self.primary_action_dim],
            'action_pred': action_pred,
        }

        if self.aux_action:
            result['ee_pose_pred'] = action_full[..., self.primary_action_dim:self.primary_action_dim + self.aux_ee_dim]
        
        if self.aux_force:
            result['force_pred'] = action_full[..., self.primary_action_dim + self.aux_ee_dim:]

        if self.use_progress:
            with torch.no_grad():
                progress_score = self.progress_head(vis_cond, force_cond)
            result['progress_score'] = progress_score.squeeze(-1)  # (B,) Tensor

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
            betas: Tuple[float, float] = (0.9, 0.95)
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        if obs_encoder_lr is not None:
            cprint(f"[ManiFlowTransformerPointcloudPolicy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay,
                "lr": obs_encoder_lr # for fine tuning
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay
            })

        if self.force_conditioned:
            optim_groups.append({
                "params": list(self.force_encoder.parameters()),
                "weight_decay": weight_decay,
            })

        if self.wrist_conditioned:
            optim_groups.append({
                "params": list(self.wrist_encoder.parameters()),
                "weight_decay": weight_decay,
            })

        if self.use_progress:
            optim_groups.append({
                "params": list(self.progress_head.parameters()),
                "weight_decay": weight_decay,
            })

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

        return optimizer
    
    def sample_t(self, batch_size, mode="uniform"):
        """
        Sample t for flow matching or consistency training.
        """
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "lognorm":
            t = sample_logit_normal(batch_size, m=self.lognorm_m, s=self.lognorm_s, device=self.device)
        elif mode == "mode":
            t = sample_mode(batch_size, s=self.mode_s, device=self.device)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        elif mode == "discrete":
            t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,)).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(f" Unsupported sample_t_mode {mode}. Choose from 'uniform', 'lognorm', 'mode', 'cosmap', 'beta', 'discrete'.")
        return t

    def sample_dt(self, batch_size, sample_dt_mode="uniform"):
        """
        Sample dt for consistency training.
        """
        if sample_dt_mode == "uniform":
            dt = torch.rand((batch_size,), device=self.device)
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
        force_cond = model_kwargs.get('force_cond', None)
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
        x_0_flow = torch.randn_like(actions, device=device) 
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
        target_dict['force_cond'] = force_cond
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
        force_cond = model_kwargs.get('force_cond', None)
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
        x0_ct = torch.randn_like(actions, device=device) 
        x1_ct = actions.to(device) 
        x_t_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        # predict the average velocity from t_next toward next target (t_next + delta_t2)
        with torch.no_grad():
            v_avg_to_next_target = ema_model.model(
                sample=x_t_next, 
                timestep=t_next.squeeze(),
                target_t=target_t_next.squeeze(), 
                vis_cond=vis_cond[-consistency_batchsize:],
                state_cond=state_cond[-consistency_batchsize:] if state_cond is not None else None, 
                lang_cond=lang_cond[-consistency_batchsize:] if lang_cond is not None else None,
                force_cond=force_cond[-consistency_batchsize:] if force_cond is not None else None,
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

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]

        lang_cond = None
        force_cond = None
        wrist_cond = None
        ema_model = ema_model

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None

        if self.force_conditioned:
            hand_obs = nobs['hand_obs'][:, :self.n_obs_steps, ...].to(self.device)
            hand_obs = hand_obs.permute(0, 2, 1)
            force_features = self.force_encoder(hand_obs)
            force_cond = force_features.permute(0, 2, 1)

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

        vis_cond_flow = vis_cond[:flow_batchsize]
        vis_cond_ct = vis_cond[flow_batchsize:flow_batchsize + consistency_batchsize]
        state_cond_flow = state_cond[:flow_batchsize]
        state_cond_ct = state_cond[flow_batchsize:flow_batchsize + consistency_batchsize]
        pc_pe_flow = pc_pe[:flow_batchsize] if pc_pe is not None else None
        pc_pe_ct = pc_pe[flow_batchsize:flow_batchsize + consistency_batchsize] if pc_pe is not None else None
        force_cond_flow = force_cond[:flow_batchsize] if force_cond is not None else None
        force_cond_ct = force_cond[flow_batchsize:flow_batchsize + consistency_batchsize] if force_cond is not None else None
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
            force_cond=force_cond_flow,
            wrist_cond=wrist_cond_flow,
            pc_pe=pc_pe_flow,
        )
        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'],
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond_flow,
            state_cond=state_cond_flow,
            force_cond=force_cond_flow,
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
            force_cond=force_cond_ct,
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
            force_cond=force_cond_ct,
            wrist_cond=wrist_cond_ct,
            lang_cond=lang_cond_ct,
            pc_pe=pc_pe_ct,
        )
        v_ct_pred_magnitude = torch.sqrt(torch.mean(v_ct_pred ** 2)).item()

        # Losses
        loss = 0.
        p = self.primary_action_dim
        e = self.aux_ee_dim

        v_flow_target = flow_target_dict['v_target']
        loss_flow_raw = F.mse_loss(v_flow_pred, v_flow_target, reduction='none')
        loss_flow_primary = reduce(loss_flow_raw[..., :p], 'b ... -> b (...)', 'mean').mean()
        loss_flow_aux_ee = reduce(loss_flow_raw[..., p:p+e], 'b ... -> b (...)', 'mean').mean() if self.aux_action else torch.tensor(0.0, device=self.device)
        loss_flow_aux_force = reduce(loss_flow_raw[..., p+e:], 'b ... -> b (...)', 'mean').mean() if self.aux_force else torch.tensor(0.0, device=self.device)
        loss += loss_flow_primary + self.lambda_aux_ee * loss_flow_aux_ee + self.lambda_aux_force * loss_flow_aux_force

        v_ct_target = consistency_target_dict['v_target']
        loss_ct_raw = F.mse_loss(v_ct_pred, v_ct_target, reduction='none')
        loss_ct_primary = reduce(loss_ct_raw[..., :p], 'b ... -> b (...)', 'mean').mean()
        loss_ct_aux_ee = reduce(loss_ct_raw[..., p:p+e], 'b ... -> b (...)', 'mean').mean() if self.aux_action else torch.tensor(0.0, device=self.device)
        loss_ct_aux_force = reduce(loss_ct_raw[..., p+e:], 'b ... -> b (...)', 'mean').mean() if self.aux_force else torch.tensor(0.0, device=self.device)
        loss += loss_ct_primary + self.lambda_aux_ee * loss_ct_aux_ee + self.lambda_aux_force * loss_ct_aux_force

        loss_progress = 0.0
        if self.use_progress:
            progress_labels = batch['progress'][:, self.n_obs_steps - 1, :].to(self.device).float()
            progress_score = self.progress_head(vis_cond, force_cond)
            loss_progress = F.mse_loss(progress_score, progress_labels)
            loss = loss + self.lambda_progress * loss_progress
            loss_progress = loss_progress.item()

        # === Vision-only auxiliary action loss (anti-proprio-shortcut) ===
        # Predict the full normalized action from vision alone. This head
        # cannot use proprioception, so its gradient forces the encoder to
        # encode object location / geometry - exactly the info the main
        # (proprio-shortcutting) action path was ignoring.
        loss_vision_aux = 0.0
        if getattr(self, 'use_vision_aux', False):
            vision_action_pred = self.vision_action_head(vis_cond)
            loss_vision_aux_t = F.mse_loss(vision_action_pred, nactions)
            loss = loss + self.lambda_vision_aux * loss_vision_aux_t
            loss_vision_aux = loss_vision_aux_t.item()

        loss = loss.mean()

        loss_dict = {
            'total_loss': loss.item(),
            'loss_flow_primary': loss_flow_primary.item(),
            'loss_flow_ee': loss_flow_aux_ee.item(),
            'loss_flow_force': loss_flow_aux_force.item(),
            'loss_ct_primary': loss_ct_primary.item(),
            'loss_ct_ee': loss_ct_aux_ee.item(),
            'loss_ct_force': loss_ct_aux_force.item(),
            'loss_progress': loss_progress,
            'v_flow_pred_magnitude': v_flow_pred_magnitude,
            'v_ct_pred_magnitude': v_ct_pred_magnitude,
        }

        # LOGGING: track whether the wrist encoder is actively learning.
        if self.wrist_conditioned and wrist_cond is not None:
            loss_dict['wrist_cond_magnitude'] = torch.sqrt(torch.mean(wrist_cond ** 2)).item()

        return loss, loss_dict
