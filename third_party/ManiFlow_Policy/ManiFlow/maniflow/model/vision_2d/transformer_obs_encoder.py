import copy

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from termcolor import cprint

from maniflow.model.common.module_attr_mixin import ModuleAttrMixin

from maniflow.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TransformerObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str='vit_base_patch16_clip_224.openai',
            global_pool: str='',
            transforms: list=None,
            n_emb: int=768,
            pretrained: bool=False,
            frozen: bool=False,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            feature_aggregation: str=None,
            downsample_ratio: int=32,
            # apply ImageNet mean/std after the /255 (required for pretrained backbones)
            imagenet_norm: bool=False,
            # photometric jitter, applied on [0,1] data before imagenet_norm
            color_jitter: dict=None,
            # let a ViT interpolate its pos-embed to a non-square / non-224 input
            dynamic_img_size: bool=True
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        timm_kwargs = dict(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )
        if model_name.startswith('vit'):
            # Without this timm's PatchEmbed asserts strict_img_size and
            # rejects anything that is not the model's native resolution.
            timm_kwargs['dynamic_img_size'] = dynamic_img_size
        model = timm.create_model(**timm_kwargs)
        self.model_name = model_name
        self.imagenet_norm = imagenet_norm

        # Must run on [0,1] values, so it cannot live in key_transform_map
        # (that is applied after imagenet_norm in forward).
        self.color_jitter = None
        if color_jitter is not None:
            self.color_jitter = torchvision.transforms.ColorJitter(**dict(color_jitter))
            cprint(f"[TransformerObsEncoder] ColorJitter: {dict(color_jitter)}", 'cyan')

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('vit'):
            feature_dim = model.embed_dim
            patch_size = model.patch_embed.patch_size
            self.patch_size = (patch_size if isinstance(patch_size, (tuple, list))
                               else (patch_size, patch_size))
            self.num_prefix_tokens = getattr(model, 'num_prefix_tokens', 1)
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
            
        # image_shape must be known before the aggregators below are sized
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        if model_name.startswith('vit'):
            # (h, w) patch grid for the ACTUAL input, not the 224x224 default.
            feature_map_shape = [image_shape[0] // self.patch_size[0],
                                 image_shape[1] // self.patch_size[1]]
            self.patch_grid = tuple(feature_map_shape)
        else:
            feature_map_shape = [x // downsample_ratio for x in image_shape]

        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            # 'avg'/'max'/None operate on the patch tokens, 'cls' on the class
            # token. Anything else has no meaning for a ViT.
            if self.feature_aggregation not in (None, 'cls', 'avg', 'max'):
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        
        if imagenet_norm:
            # (3,1,1) broadcasts against both B,T,C,H,W and B*T,C,H,W
            self.register_buffer('image_mean',
                torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), persistent=False)
            self.register_buffer('image_std',
                torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1), persistent=False)

        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            # Crop and resize per-axis: using image_shape[0] for both would
            # squash a non-square input (252x336 -> 252x252) and silently
            # change the ViT patch grid, hence the token count.
            crop_h = int(image_shape[0] * ratio)
            crop_w = int(image_shape[1] * ratio)
            transforms = [
                torchvision.transforms.RandomCrop(size=(crop_h, crop_w)),
                torchvision.transforms.Resize(size=tuple(image_shape), antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                dim = np.prod(shape)
                proj = nn.Identity()
                if dim != n_emb:
                    proj = nn.Linear(in_features=dim, out_features=n_emb)
                key_projection_map[key] = proj

                low_dim_keys.append(key)
            else:
                cprint(f"Skipping obs key {key} with type {type}", 'red')
                # raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        # Return: B, N, C
        
        if self.model_name.startswith('vit'):
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]

            # Drop CLS / register tokens so only patch tokens remain.
            patch_tokens = feature[:, self.num_prefix_tokens:, :]
            if self.feature_aggregation == 'avg':
                return torch.mean(patch_tokens, dim=1, keepdim=True)
            elif self.feature_aggregation == 'max':
                return torch.amax(patch_tokens, dim=1, keepdim=True)

            assert self.feature_aggregation is None
            return patch_tokens
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            # normalize image by hand
            if img.max() > 1.0:
                # assume input in [0, 255]
                img = img / 255.0
            if img.shape[-1] == 3:
                if len(img.shape) == 5:
                    # B, T, H, W, C --> B, T, C, H, W
                    img = img.permute(0, 1, 4, 2, 3)
                elif len(img.shape) == 4:
                    # B, H, W, C --> B, C, H, W
                    img = img.permute(0, 3, 1, 2)

            if self.color_jitter is not None and self.training:
                img = self.color_jitter(img)

            if self.imagenet_norm:
                img = (img - self.image_mean) / self.image_std

            B, T = img.shape[:2]
            assert B == batch_size
            img = img.reshape(B*T, *img.shape[2:])
            # Compare against (C,H,W); img.shape[2:] is only (H,W) and would
            # make this branch fire on every forward.
            if img.shape[1:] != self.key_shape_map[key]:
                target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]
                # do torchvision resize
                # img shape: Bx3xHxW
                # new size: Bx3xnHxnW
                img = F.interpolate(img, size=(target_H, target_W), mode='bilinear', align_corners=False)
            assert img.shape[1:] == self.key_shape_map[key]
            img = self.key_transform_map[key](img).to(self.device)
            raw_feature = self.key_model_map[key](img).to(self.device)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[key](feature)
            assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
            emb = emb.reshape(B,-1,self.n_emb)
            embeddings.append(emb)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key].to(self.device)
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B,T,-1)
            emb = self.key_projection_map[key](data)
            assert emb.shape[-1] == self.n_emb
            embeddings.append(emb)
        
        # concatenate all features along t
        result = torch.cat(embeddings, dim=1)
        return result

    def grad_cam_target(self):
        """Return (layer, is_tokens) to hook for attribution.

        A ViT's only Conv2d is patch_embed.proj, the very first op, which
        carries no spatial semantics. Hook the last transformer block instead
        and let the caller reshape tokens via tokens_to_map().
        """
        if self.model_name.startswith('vit'):
            model = next(iter(self.key_model_map.values()))
            return model.blocks[-1], True

        last_conv = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv, False

    def tokens_to_map(self, x):
        """(B, prefix + h*w, D) -> (B, D, h, w)."""
        h, w = self.patch_grid
        x = x[:, self.num_prefix_tokens:, :]
        return x.transpose(1, 2).reshape(x.shape[0], -1, h, w)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1

        return example_output.shape


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # To do: replace with your own config path
    with hydra.initialize('../maniflow/config'):
        cfg = hydra.compose('train_maniflow_transformer_bimanual_workspace')
        OmegaConf.resolve(cfg)

    shape_meta = cfg.task.shape_meta
    encoder = TransformerObsEncoder(
        shape_meta=shape_meta
    )