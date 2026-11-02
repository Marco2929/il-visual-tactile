"""Frozen DINOv3 multi-camera RGB encoder.

Why this is a separate class rather than a subclass of TransformerObsEncoder:
that class derives a single global `image_shape` from the shape_meta and
asserts every rgb key agrees with it. Supporting two cameras at different
resolutions means that assert must not run at all, which would require
skipping its __init__ entirely - at which point inheritance buys nothing and
only obscures which code is live. The old encoder stays untouched and in use
for single-camera configs.

Design:
  * One frozen backbone instance PER camera key (no weight sharing). A
    top-down view from 1.5 m and a wrist close-up share almost no image
    statistics; one shared trunk would have to cover both.
  * Per-key resolution, each validated independently.
  * ALL patch tokens are kept (no pooling) -> trainable
    Linear(embed_dim, out_dim) applied per token. The trainable parameters
    in this module are those projections plus one group embedding per
    camera key.
  * Token feature = projection + cam_emb[key] + grid_pe[patch]. Both extra
    terms are needed because the head supplies NEITHER: DiTX adds a single
    shared `modality_emb[1]` to the entire RGB group (so cameras are
    indistinguishable), and its `vis_cond_pos_embed` is added to the
    point-cloud group before the RGB group is concatenated (so patch
    position is lost).
      - cam_emb: learnable (n_keys, out_dim), ZERO-init -> no-op at step 0.
      - grid_pe: parameter-free factorised 2D sin-cos over the (h_p, w_p)
        patch raster, cached per (h_p, w_p, dtype, device). Resolution
        agnostic, so changing a camera resolution does not break a
        checkpoint.

Token group order is SORTED by key name and must stay that way: the
concatenation order is baked into every checkpoint that trains against it.
"""

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from termcolor import cprint

from maniflow.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

# DINOv3 ViT-S/16. Both target resolutions must be divisible by this.
# Pretrain resolution is 256x256; anything else relies on pos-embed
# interpolation, which is what dynamic_img_size=True enables.
#
# Current targets: front_rgb 192x320 (12x20 = 240 tokens),
#                  wrist_rgb 192x256 (12x16 = 192 tokens).
# FALLBACK: if the pretrained features measurably degrade at 192x320 because
# the pos-embed is interpolated down from 16x16, the next step up at the same
# 16:9 aspect is 256x448 (16x28 = 448 tokens). Not implemented - noted here so
# the reason for the current numbers is not lost.
PATCH_SIZE = 16


def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    """(M,) positions -> (M, dim) sin-cos embedding."""
    omega = torch.arange(dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000.0 ** (omega / (dim / 2.0)))
    out = pos.float()[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _sincos_2d(dim: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """Factorised 2D sin-cos PE over an (h, w) raster -> (h*w, dim).

    Half the channels encode the row index, half the column index. Token
    order is ROW-MAJOR to match the ViT's patch token order (indexing='ij'
    then flatten), so grid_pe[i] belongs to patch_tokens[:, i].
    """
    if dim % 4:
        raise ValueError(
            f"out_dim must be divisible by 4 for a factorised 2D sin-cos "
            f"positional encoding (half the channels per axis, each split "
            f"into sin/cos), got {dim}")
    grid_h, grid_w = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij')
    emb_h = _sincos_1d(dim // 2, grid_h.reshape(-1))
    emb_w = _sincos_1d(dim // 2, grid_w.reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1).to(dtype)


class DINOv3ObsEncoder(ModuleAttrMixin):
    """Encode one or more RGB camera streams into conditioning tokens.

    Assumes rgb input B,T,C,H,W (channels-first; see `forward`).
    """

    def __init__(self,
            shape_meta: dict,
            model_name: str = 'vit_small_patch16_dinov3.lvd1689m',
            pretrained: bool = True,
            frozen: bool = True,
            out_dim: int = 128,
            share_rgb_model: bool = False,
            dynamic_img_size: bool = True,
            color_jitter: dict = None,
            random_crop_ratio: float = None,
        ):
        super().__init__()

        obs_shape_meta = shape_meta['obs']

        # --- Collect + validate the rgb keys -----------------------------
        # Sorted so the token group order is deterministic and independent of
        # dict insertion order / YAML ordering.
        rgb_keys = sorted(
            key for key, attr in obs_shape_meta.items()
            if attr.get('type', 'low_dim') == 'rgb'
        )
        if not rgb_keys:
            raise ValueError(
                "DINOv3ObsEncoder requires at least one obs key with "
                "type: rgb in shape_meta"
            )

        key_shape_map = {}
        for key in rgb_keys:
            shape = tuple(obs_shape_meta[key]['shape'])
            # Every message names the key: with two towers an un-attributed
            # shape error is not localisable.
            if len(shape) != 3:
                raise ValueError(
                    f"obs key '{key}': expected shape (C, H, W), got {shape}"
                )
            c, h, w = shape
            if c != 3:
                raise ValueError(
                    f"obs key '{key}': expected 3 channels (CHW layout), got "
                    f"C={c} from shape {shape}. This encoder is RGB-only."
                )
            if h % PATCH_SIZE or w % PATCH_SIZE:
                raise ValueError(
                    f"obs key '{key}': resolution {h}x{w} is not divisible by "
                    f"the ViT patch size {PATCH_SIZE} "
                    f"({h}/{PATCH_SIZE}={h / PATCH_SIZE:.2f}, "
                    f"{w}/{PATCH_SIZE}={w / PATCH_SIZE:.2f}). A fractional "
                    f"patch grid silently changes the token count."
                )
            key_shape_map[key] = shape

        # --- Backbones ----------------------------------------------------
        timm_kwargs = dict(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,          # drop the classification head
        )
        if model_name.startswith('vit'):
            # Without this timm's PatchEmbed enforces strict_img_size and
            # rejects anything that is not the pretrain resolution (256x256).
            timm_kwargs['dynamic_img_size'] = dynamic_img_size

        key_model_map = nn.ModuleDict()
        shared = timm.create_model(**timm_kwargs) if share_rgb_model else None
        for key in rgb_keys:
            key_model_map[key] = (shared if shared is not None
                                  else timm.create_model(**timm_kwargs))

        reference = key_model_map[rgb_keys[0]]
        feature_dim = reference.num_features
        # Read the prefix token count off the model instead of assuming 1:
        # DINOv3 carries register tokens in addition to CLS, and dropping the
        # wrong number would mix non-spatial tokens into the spatial mean.
        self.num_prefix_tokens = getattr(reference, 'num_prefix_tokens', 1)

        patch_size = reference.patch_embed.patch_size
        patch_size = (patch_size if isinstance(patch_size, (tuple, list))
                      else (patch_size, patch_size))
        if tuple(patch_size) != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"model '{model_name}' has patch size {tuple(patch_size)}, but "
                f"this encoder validates resolutions against {PATCH_SIZE}. "
                f"Update PATCH_SIZE and the configured resolutions together."
            )

        if frozen:
            if not pretrained:
                raise ValueError(
                    "frozen=True with pretrained=False would freeze random "
                    "weights"
                )
            for model in key_model_map.values():
                for param in model.parameters():
                    param.requires_grad = False

        # --- Normalization statistics -------------------------------------
        # Resolved from the model's own pretrain config rather than hardcoded
        # ImageNet constants: DINOv3 need not use the same statistics, and a
        # silent mismatch degrades features without raising anything.
        data_cfg = timm.data.resolve_model_data_config(reference)
        mean = torch.tensor(data_cfg['mean']).view(-1, 1, 1)
        std = torch.tensor(data_cfg['std']).view(-1, 1, 1)
        # (C,1,1) broadcasts against both B,T,C,H,W and B*T,C,H,W.
        self.register_buffer('image_mean', mean, persistent=False)
        self.register_buffer('image_std', std, persistent=False)
        self._resolved_data_config = {
            'mean': tuple(float(v) for v in data_cfg['mean']),
            'std': tuple(float(v) for v in data_cfg['std']),
            'input_size': tuple(data_cfg.get('input_size', ())),
            'interpolation': data_cfg.get('interpolation'),
        }
        self._logged_data_config = False

        # --- Trainable projections ----------------------------------------
        key_projection_map = nn.ModuleDict({
            key: nn.Linear(feature_dim, out_dim) for key in rgb_keys
        })

        # --- Camera embedding (one vector per camera key) ------------------
        # Row i belongs to rgb_keys[i], so the order is tied to self.rgb_keys
        # rather than to hardcoded names: adding a camera to shape_meta shifts
        # the rows consistently with the concatenation order in forward().
        # Zero-init -> exact no-op at step 0, mirroring DiTX.modality_emb.
        cam_emb = nn.Parameter(torch.zeros(len(rgb_keys), out_dim))

        # Validate the sin-cos constraint once here rather than on the first
        # forward pass, so a bad out_dim fails at construction time.
        if out_dim % 4:
            raise ValueError(
                f"out_dim={out_dim} must be divisible by 4: the positional "
                f"encoding splits it into row/column halves, each into "
                f"sin/cos quarters")

        # --- Augmentation (train only) ------------------------------------
        self.color_jitter = None
        if color_jitter is not None:
            self.color_jitter = torchvision.transforms.ColorJitter(
                **dict(color_jitter))
            cprint(f"[DINOv3ObsEncoder] ColorJitter: {dict(color_jitter)} "
                   f"(train only)", 'cyan')
        self.random_crop_ratio = random_crop_ratio
        if random_crop_ratio is not None:
            if not 0 < random_crop_ratio <= 1:
                raise ValueError(
                    f"random_crop_ratio must be in (0, 1], got "
                    f"{random_crop_ratio}")
            cprint(f"[DINOv3ObsEncoder] RandomCrop ratio "
                   f"{random_crop_ratio} + resize back (train only)", 'cyan')

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.key_shape_map = key_shape_map
        self.key_model_map = key_model_map
        self.key_projection_map = key_projection_map
        self.cam_emb = cam_emb
        self.key_group_index = {key: i for i, key in enumerate(rgb_keys)}
 
        self._grid_pe_cache = {}
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.frozen = frozen
        self.share_rgb_model = share_rgb_model

        self.patch_grids = {
            key: (key_shape_map[key][1] // PATCH_SIZE,
                  key_shape_map[key][2] // PATCH_SIZE)
            for key in rgb_keys
        }

        # Which camera Grad-CAM hooks, and whether the no_grad fast path is
        # bypassed. Attribution needs activation gradients inside the trunk,
        # which the frozen path deliberately does not produce.
        self.grad_cam_key = rgb_keys[0]
        self._grad_cam_enabled = False

        if frozen:
            # Keep the trunks in eval() from the start, so a forward pass
            # before the first .train(False) still sees inference-mode norms.
            for model in key_model_map.values():
                model.eval()

        total_tokens = 0
        for key in rgb_keys:
            gh, gw = self.patch_grids[key]
            total_tokens += gh * gw
            cprint(f"[DINOv3ObsEncoder] {key}: {key_shape_map[key]} -> "
                   f"{gh}x{gw}={gh * gw} tokens -> project -> "
                   f"{gh * gw}x{out_dim} (+cam_emb #"
                   f"{self.key_group_index[key]} +sincos grid_pe)", 'cyan')
        cprint(f"[DINOv3ObsEncoder] {total_tokens} tokens per timestep across "
               f"{len(rgb_keys)} camera(s); the head sees T x {total_tokens} "
               f"conditioning tokens", 'cyan')
        cprint(f"[DINOv3ObsEncoder] {model_name} "
               f"(frozen={frozen}, shared={share_rgb_model}), "
               f"token group order: {rgb_keys}", 'cyan')

    def train(self, mode: bool = True):
        """Keep frozen backbones in eval() regardless of the parent's mode.

        requires_grad=False stops gradients but does NOT stop norm layers from
        running in training mode. For a frozen feature extractor that silently
        changes the features, so the trunks are pinned to eval() here - this
        runs on every .train() call from the policy, not just once at init.
        The trainable projections follow `mode` normally.
        """
        super().train(mode)
        if self.frozen:
            for model in self.key_model_map.values():
                model.eval()
        return self

    def set_grad_cam_key(self, key):
        """Select which camera grad_cam_target() hooks."""
        if key not in self.rgb_keys:
            raise ValueError(
                f"'{key}' is not an rgb key of this encoder ({self.rgb_keys})")
        self.grad_cam_key = key

    def grad_cam_target(self):
        """Return (layer, is_tokens) to hook for attribution.

        A ViT's only Conv2d is patch_embed.proj, the very first op, which
        carries no spatial semantics. Hook the last transformer block instead
        and let the caller reshape tokens via tokens_to_map().
        """
        model = self.key_model_map[self.grad_cam_key]
        return model.blocks[-1], True

    def tokens_to_map(self, x):
        """(B, prefix + h*w, D) -> (B, D, h, w) for the Grad-CAM camera."""
        h, w = self.patch_grids[self.grad_cam_key]
        x = x[:, self.num_prefix_tokens:, :]
        return x.transpose(1, 2).reshape(x.shape[0], -1, h, w)

    def grad_cam_mode(self, enabled=True):
        """Context manager that lets gradients flow through frozen trunks.

        Grad-CAM needs d(loss)/d(activations) INSIDE the backbone. The frozen
        forward path wraps the trunk in torch.no_grad() (correct for training:
        no graph, no activation memory), which leaves the hook with no
        gradient at all - the map would come out uniformly zero rather than
        obviously broken. This temporarily reopens the graph; parameters stay
        requires_grad=False, so nothing becomes trainable.
        """
        encoder = self

        class _Ctx:
            def __enter__(self):
                self.prev = encoder._grad_cam_enabled
                encoder._grad_cam_enabled = enabled
                return encoder

            def __exit__(self, *exc):
                encoder._grad_cam_enabled = self.prev
                return False

        return _Ctx()

    def _grid_pe(self, key, device, dtype):
        """(N, out_dim) sin-cos patch PE for one camera, cached.

        Keyed on (h_p, w_p, dtype, device): two cameras with the same patch
        raster share one tensor, and a .to(device) on the module is picked up
        on the next call rather than silently returning a wrong-device tensor.
        """
        gh, gw = self.patch_grids[key]
        cache_key = (gh, gw, dtype, str(device))
        pe = self._grid_pe_cache.get(cache_key)
        if pe is None:
            pe = _sincos_2d(self.out_dim, gh, gw, device, dtype)
            self._grid_pe_cache[cache_key] = pe
        return pe

    def _augment(self, img, expected):
        """(B*T, C, H, W) in [0,1] -> augmented, same shape.

        Crop-then-resize rather than crop-only: the patch grid is derived from
        the configured resolution, so changing H/W here would change the token
        count and desync it from cam_emb/grid_pe.
        """
        if self.random_crop_ratio is not None:
            h = max(int(expected[1] * self.random_crop_ratio), PATCH_SIZE)
            w = max(int(expected[2] * self.random_crop_ratio), PATCH_SIZE)
            top = int(torch.randint(0, expected[1] - h + 1, (1,)).item())
            left = int(torch.randint(0, expected[2] - w + 1, (1,)).item())
            img = img[..., top:top + h, left:left + w]
            img = F.interpolate(img, size=(expected[1], expected[2]),
                                mode='bilinear', align_corners=False)
        if self.color_jitter is not None:
            # ColorJitter samples ONE set of factors per call, so the whole
            # (B*T) stack gets the same jitter. That is intentional: the obs
            # steps of one sample must stay photometrically consistent, or the
            # policy sees an impossible lighting change between t-1 and t.
            img = self.color_jitter(img)
        return img

    def _extract(self, key, img, return_raw=False):
        """(B*T, C, H, W) -> (B*T, N, out_dim) for one camera.

        N = gh*gw patch tokens; no pooling. Adds the per-camera embedding and
        the sin-cos patch PE, both broadcast over the batch axis.
        """
        model = self.key_model_map[key]

        # forward_features keeps the token axis; the default forward would
        # apply the (removed) head and pooling.
        if self.frozen and not self._grad_cam_enabled:
            with torch.no_grad():
                tokens = model.forward_features(img)
        else:
            if self._grad_cam_enabled:
                # Every trunk parameter has requires_grad=False and the image
                # is a non-differentiable uint8 leaf, so autograd would build
                # NO graph at all and the Grad-CAM hook would never fire.
                # Making the INPUT require grad gives the graph a root, so
                # intermediate activations become differentiable without any
                # parameter becoming trainable.
                img = img.detach().requires_grad_(True)
            tokens = model.forward_features(img)

        if tokens.ndim != 3:
            raise RuntimeError(
                f"obs key '{key}': expected (B, N, C) tokens from "
                f"forward_features, got {tuple(tokens.shape)}"
            )

        # Drop CLS + register tokens, keep only spatial patches.
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]

        gh, gw = self.patch_grids[key]
        if patch_tokens.shape[1] != gh * gw:
            raise RuntimeError(
                f"obs key '{key}': got {patch_tokens.shape[1]} patch tokens "
                f"but the configured {self.key_shape_map[key][1]}x"
                f"{self.key_shape_map[key][2]} input implies {gh}x{gw}="
                f"{gh * gw}. num_prefix_tokens={self.num_prefix_tokens}."
            )

        # Projection is trainable, so it stays outside the no_grad above. It
        # acts on the last axis, so every token is projected independently:
        # (B*T, N, feature_dim) -> (B*T, N, out_dim).
        projected = self.key_projection_map[key](patch_tokens)

        # cam_emb: (out_dim,) broadcasts over batch and token axes -> tells
        # the head WHICH camera. grid_pe: (N, out_dim) broadcasts over the
        # batch axis only -> tells it WHERE in that camera's raster.
        cam = self.cam_emb[self.key_group_index[key]]
        grid_pe = self._grid_pe(key, projected.device, projected.dtype)
        out = projected + cam + grid_pe
        if return_raw:
            return out, patch_tokens.detach()
        return out

    def forward(self, obs_dict, return_raw_keys=None):
        """Encode all rgb keys into (B, sum_k(T * N_k), out_dim).

        Input per key: (B, T, C, H, W), uint8 or float. Values in [0, 255] are
        scaled to [0, 1] automatically.

        If return_raw_keys is a list of keys, returns (tokens, raw_tokens_dict)
        where raw_tokens_dict[key] has shape (B, T, N_p, feature_dim) detached.
        """
        if not self._logged_data_config:
            cprint(f"[DINOv3ObsEncoder] normalization from "
                   f"timm.data.resolve_model_data_config: "
                   f"mean={self._resolved_data_config['mean']}, "
                   f"std={self._resolved_data_config['std']}, "
                   f"pretrain input_size="
                   f"{self._resolved_data_config['input_size']}", 'yellow')
            self._logged_data_config = True

        embeddings = []
        raw_dict = {} if return_raw_keys is not None else None
        for key in self.rgb_keys:
            img = obs_dict[key]
            expected = self.key_shape_map[key]

            if img.ndim != 5:
                raise ValueError(
                    f"obs key '{key}': expected (B, T, C, H, W), got "
                    f"{tuple(img.shape)}. Channels-last input must be "
                    f"permuted before it reaches this encoder."
                )
            if img.shape[2] != expected[0]:
                raise ValueError(
                    f"obs key '{key}': expected {expected[0]} channels at "
                    f"dim 2 (CHW layout), got {tuple(img.shape)}. A "
                    f"(B, T, H, W, C) tensor would show C={img.shape[2]} "
                    f"here - permute to CHW first."
                )

            img = img.to(self.device)
            if not torch.is_floating_point(img):
                img = img.float()
            # uint8 [0,255] -> [0,1]. Checked on the tensor max rather than the
            # dtype so an already-scaled float tensor is not divided twice.
            if img.max() > 1.0:
                img = img / 255.0

            B, T = img.shape[:2]
            img = img.reshape(B * T, *img.shape[2:])

            if img.shape[1:] != tuple(expected):
                img = F.interpolate(
                    img, size=(expected[1], expected[2]),
                    mode='bilinear', align_corners=False)

            if self.training:
                img = self._augment(img, expected)

            img = (img - self.image_mean) / self.image_std

            want_raw = (return_raw_keys is not None and key in return_raw_keys)
            if want_raw:
                emb, raw_tokens = self._extract(key, img, return_raw=True)
                raw_dict[key] = raw_tokens.reshape(B, T, raw_tokens.shape[1], raw_tokens.shape[2])
            else:
                emb = self._extract(key, img)
            # (B*T, N, out_dim) -> (B, T*N, out_dim). Time and patch axes are
            # flattened together; the head treats the result as a flat token
            # set, so only the group order below has to stay stable.
            embeddings.append(emb.reshape(B, -1, self.out_dim))

        # Concatenate along the token axis in sorted key order.
        cat_emb = torch.cat(embeddings, dim=1)
        if return_raw_keys is not None:
            return cat_emb, raw_dict
        return cat_emb

    @torch.no_grad()
    def extract_patch_tokens(self, key: str, img: torch.Tensor) -> torch.Tensor:
        """Extract raw frozen, detached patch tokens F without projection/cam_emb/grid_pe.

        Args:
            key: camera key (e.g. 'wrist_rgb')
            img: (B, T, C, H, W) or (B, C, H, W) or channels-last (B, T, H, W, C) / (B, H, W, C).
                 uint8 [0, 255] or float [0, 1].
        Returns:
            F: (B, T, N_p, D) detached patch tokens (or (B, N_p, D) if 4D input).
        """
        if key not in self.key_model_map:
            raise KeyError(f"Camera key '{key}' not in known keys: {list(self.key_model_map.keys())}")

        is_5d = (img.ndim == 5)
        if not is_5d and img.ndim != 4:
            raise ValueError(f"Expected 4D or 5D image tensor, got {tuple(img.shape)}")

        # Handle channels-last (B, T, H, W, C) -> (B, T, C, H, W)
        if img.shape[-1] <= 4 and (img.shape[2] > 4 if is_5d else img.shape[1] > 4):
            if is_5d:
                img = img.permute(0, 1, 4, 2, 3)
            else:
                img = img.permute(0, 3, 1, 2)

        expected = self.key_shape_map[key]
        if is_5d:
            B, T = img.shape[:2]
            img = img.reshape(B * T, *img.shape[2:])
        else:
            B = img.shape[0]
            T = 1

        img = img.to(self.device)
        if not torch.is_floating_point(img):
            img = img.float()
        if img.max() > 1.0:
            img = img / 255.0

        if img.shape[1:] != tuple(expected):
            img = F.interpolate(
                img, size=(expected[1], expected[2]),
                mode='bilinear', align_corners=False)

        img = (img - self.image_mean) / self.image_std

        model = self.key_model_map[key]
        tokens = model.forward_features(img)
        patch_tokens = tokens[:, self.num_prefix_tokens:, :].detach()

        if is_5d:
            N_p = patch_tokens.shape[1]
            D = patch_tokens.shape[2]
            patch_tokens = patch_tokens.reshape(B, T, N_p, D)

        return patch_tokens

    def count_parameters(self):
        """(trainable, frozen) parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return trainable, frozen

    @torch.no_grad()
    def output_shape(self):
        example = {}
        obs_shape_meta = self.shape_meta['obs']
        for key in self.rgb_keys:
            attr = obs_shape_meta[key]
            example[key] = torch.zeros(
                (1, attr['horizon']) + tuple(attr['shape']),
                dtype=self.dtype, device=self.device)
        out = self.forward(example)
        assert len(out.shape) == 3 and out.shape[0] == 1
        return out.shape