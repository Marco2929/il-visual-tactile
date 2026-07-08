# maniflow/model/diffusion/ditx.py

import re
import logging
import math
from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, RmsNorm
from maniflow.model.diffusion.positional_embedding import SinusoidalPosEmb
from maniflow.model.diffusion.ditx_block import DiTXBlock, AdaptiveLayerNorm
from termcolor import cprint

logger = logging.getLogger(__name__)

class FinalLayer(nn.Module):
    """
    The final layer of DIT-X, adopted from RDT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x
    
class DiTX(nn.Module):
    """
    Consistency Flow Training model with a Diffusion Transformer backbone.
    Now with FiLM-style vision conditioning through AdaLN.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 256,
        visual_cond_len: int = 1024,
        diffusion_timestep_embed_dim: int = 256,
        diffusion_target_t_embed_dim: int = 256,
        block_type: str = "DiTX",
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        pre_norm_modality: bool = False,
        language_conditioned: bool = False,
        force_conditioned: bool = False,
        language_model: str = "t5-small",
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.visual_cond_len = visual_cond_len
        self.language_conditioned = language_conditioned
        self.force_conditioned = force_conditioned
        self.pre_norm_modality = pre_norm_modality
        
        # constants
        T = horizon
        self.T = T
        self.horizon = horizon

        # input embedding stem
        self.hidden_dim = n_emb
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.vis_cond_pos_embed = nn.Parameter(
            torch.zeros(1, (visual_cond_len + 1) * n_obs_steps, n_emb)
        )
        
        # pre-norm visual modality
        if self.pre_norm_modality:
            self.vis_norm = AdaptiveLayerNorm(
                dim=n_emb,
                dim_cond=n_emb,
            )

        # ====== FiLM: Vision pooling for AdaLN ======
        # Attention-pool the visual tokens to a single vector
        self.vis_attn_pool = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, 1),  # score per token
        )
        cprint(f"[DiTX] FiLM-style vision conditioning enabled (vision pooled into AdaLN)", "green")
        # =============================================

        # timestep and target_t cond encoder
        flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, n_emb),
        )
        flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_target_t_embed_dim),
            nn.Linear(diffusion_target_t_embed_dim, diffusion_target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_target_t_embed_dim * 4, self.hidden_dim),
        )
        self.flow_timestep_encoder = flow_timestep_encoder
        self.flow_target_t_encoder = flow_target_t_encoder

        # ====== CHANGED: Adaptor now takes time + target_t + vision_pool ======
        self.timestep_target_t_adaptor = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        # =====================================================================

        # Language conditioning
        if self.language_conditioned:
            self.load_T5_encoder(
                model_name=language_model,
                freeze=True)
            self.lang_adaptor = self.build_condition_adapter(
                "mlp2x_gelu", 
                in_features=self.language_encoder_out_dim, 
                out_features=n_emb
            )
            if self.pre_norm_modality:
                self.lang_norm = AdaptiveLayerNorm(
                    dim=n_emb,
                    dim_cond=n_emb,
                )

        # Force conditioning
        if self.force_conditioned:
            self.force_cond_emb = nn.Linear(cond_dim, n_emb)
            if self.pre_norm_modality:
                self.force_norm = AdaptiveLayerNorm(
                    dim=n_emb,
                    dim_cond=n_emb,
                )
            
        # Building the transformer blocks
        self.block_type = block_type
        if block_type == "DiTX":
            self.blocks = nn.ModuleList([
                DiTXBlock(n_emb, n_head, mlp_ratio=mlp_ratio, p_drop_attn=p_drop_attn, 
                    qkv_bias=qkv_bias, qk_norm=qk_norm) for _ in range(n_layer)
            ])
            cprint(f"[DiTX Transformer] Initialized {n_layer} DiTX blocks with hidden size {n_emb}, "
                    f"num heads {n_head}, mlp ratio {mlp_ratio}, dropout {p_drop_attn}, qkv_bias {qkv_bias}, qk_norm {qk_norm}", "cyan")
        
        # Final Layer
        self.final_layer = FinalLayer(n_emb, output_dim)

        self.initialize_weights()
        cprint(f"[DiTX Transformer] Initialized weights for DiTX", "green")

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    # language encoder
    def load_T5_encoder(self, model_name, freeze=True):
        from transformers import (
            T5Config,
            T5EncoderModel,
            AutoTokenizer
        )
        T5_model_name = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        assert model_name in T5_model_name, f"Model name {model_name} not in {T5_model_name}"
        encoder_name = model_name
        pretrained_model_id = f"google-t5/{encoder_name}"
        encoder_cfg = T5Config()
        self.language_encoder = T5EncoderModel(encoder_cfg).from_pretrained(
                pretrained_model_id
            )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
        if freeze:
            self.language_encoder.eval()
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        self.language_encoder_out_dim = 512
        cprint(f"Loaded T5 encoder: {encoder_name}", "green")

    def encode_text_input_T5(self,
                             lang_cond,
                             norm_lang_embedding=False,
                             output_type="sentence",
                             device="cuda" if torch.cuda.is_available() else "cpu"
                             ):
        language_inputs = self.tokenizer(
            lang_cond,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = language_inputs["input_ids"].to(device)
        attention_mask = language_inputs["attention_mask"].to(device)
        encoder_outputs = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_embeddings = encoder_outputs.last_hidden_state
        if output_type == "token":
            return token_embeddings
        sentence_embedding = torch.mean(token_embeddings, dim=1).squeeze(1)
        if norm_lang_embedding:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=-1)
        return sentence_embedding

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize input emb
        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0) if self.input_emb.bias is not None else None

        # Initialize pos emb
        nn.init.normal_(self.pos_emb, std=0.02)

        # Initialize diffusion step encoder
        for layer in self.flow_timestep_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize diffusion target_t encoder
        for layer in self.flow_target_t_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize conditional observation embedding
        nn.init.normal_(self.vis_cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.vis_cond_obs_emb.bias, 0) if self.vis_cond_obs_emb.bias is not None else None

        # ====== Initialize FiLM vision pool ======
        nn.init.normal_(self.vis_attn_pool[1].weight, std=0.02)
        nn.init.constant_(self.vis_attn_pool[1].bias, 0)
        # ==========================================

        # Initialize the adaptor (now 3*hidden → hidden)
        nn.init.normal_(self.timestep_target_t_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_target_t_adaptor.bias, 0)

        if self.language_conditioned:
            nn.init.normal_(self.lang_adaptor[0].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[0].bias, 0) if self.lang_adaptor[0].bias is not None else None
            nn.init.normal_(self.lang_adaptor[-1].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[-1].bias, 0) if self.lang_adaptor[-1].bias is not None else None

        if self.force_conditioned:
            nn.init.normal_(self.force_cond_emb.weight, std=0.02)
            nn.init.constant_(self.force_cond_emb.bias, 0)
            if self.pre_norm_modality:
                nn.init.zeros_(self.force_norm.cond_linear.weight)
                nn.init.constant_(self.force_norm.cond_linear.bias[:self.hidden_dim], 1.)
                nn.init.zeros_(self.force_norm.cond_linear.bias[self.hidden_dim:])
        
        if self.pre_norm_modality:
            nn.init.zeros_(self.vis_norm.cond_linear.weight)
            nn.init.constant_(self.vis_norm.cond_linear.bias[:self.hidden_dim], 1.)
            nn.init.zeros_(self.vis_norm.cond_linear.bias[self.hidden_dim:])
            if self.language_conditioned:
                nn.init.zeros_(self.lang_norm.cond_linear.weight)
                nn.init.constant_(self.lang_norm.cond_linear.bias[:self.hidden_dim], 1.)
                nn.init.zeros_(self.lang_norm.cond_linear.bias[self.hidden_dim:])

        # Initialize the final layer: zero-out
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RmsNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        if self.vis_cond_pos_embed is not None:
            no_decay.add("vis_cond_pos_embed")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),)

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float = 1e-4, 
            weight_decay: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            target_t: Union[torch.Tensor, float, int], 
            vis_cond: torch.Tensor,
            lang_cond: Union[torch.Tensor, list, str] = None,
            force_cond: torch.Tensor = None,
            **kwargs):
        """
        Forward pass of the DiTX model.
        Input:
            x: (B,T,input_dim)
            timestep: (B,) or int, maniflow time step t
            target_t: (B,) or float, the target time for consistency flow training
            vis_cond: (B,T, vis_cond_dim) 
            lang_cond: (B,) or list of strings, language condition input
            force_cond: (B, T_force, D)
            **kwargs: additional arguments
        output: 
            action: (B,T,output_dim)
        """

        # process input
        input_emb = self.input_emb(sample)  # (B, T, n_emb)
        x = input_emb + self.pos_emb  # (B, T, n_emb)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        timestep_embed = self.flow_timestep_encoder(timesteps)  # (B, n_emb)

        # 2. target_t
        target_ts = target_t
        if not torch.is_tensor(target_ts): 
            target_ts = torch.tensor([target_ts], dtype=torch.float32, device=sample.device)    
        elif torch.is_tensor(target_ts) and len(target_ts.shape) == 0:
            target_ts = target_ts[None].to(sample.device)
        target_ts = target_ts.expand(sample.shape[0])
        target_t_embed = self.flow_target_t_encoder(target_ts)  # (B, n_emb)

        # 3. visual condition embedding
        vis_con_obs_emb = self.vis_cond_obs_emb(vis_cond)  # (B, L, n_emb)

        # ====== FiLM: Pool vision tokens into a global vector ======
        # Attention-weighted pooling
        attn_scores = self.vis_attn_pool(vis_con_obs_emb)  # (B, L, 1)
        attn_weights = F.softmax(attn_scores, dim=1)       # (B, L, 1)
        vis_pooled = (vis_con_obs_emb * attn_weights).sum(dim=1)  # (B, n_emb)

        # Combine: time + target_t + vision_pool → AdaLN conditioning
        time_c = torch.cat([timestep_embed, target_t_embed, vis_pooled], dim=-1)  # (B, 3*n_emb)
        time_c = self.timestep_target_t_adaptor(time_c)  # (B, n_emb)
        # ===========================================================

        # 4. Build cross-attention context (unchanged)
        vis_cond_pos_embed = self.vis_cond_pos_embed[:, :vis_cond.shape[1]]
        context_c = vis_con_obs_emb + vis_cond_pos_embed  # (B, L, n_emb)
        if self.pre_norm_modality:
            context_c = self.vis_norm(context_c, time_c)

        # 5. language condition
        if self.language_conditioned:
            assert lang_cond is not None
            lang_c = self.encode_text_input_T5(lang_cond, output_type="token", device=sample.device)
            lang_c = self.lang_adaptor(lang_c)
            if self.pre_norm_modality:
                lang_c = self.lang_norm(lang_c, time_c)
            context_c = torch.cat([context_c, lang_c], dim=1)

        # 5.5 force condition
        if self.force_conditioned and force_cond is not None:
            force_c = self.force_cond_emb(force_cond)
            if self.pre_norm_modality:
                force_c = self.force_norm(force_c, time_c)
            context_c = torch.cat([context_c, force_c], dim=1)

        # 6. transformer blocks
        for block in self.blocks:
            x = block(x, time_c, context_c)  # (B, T, n_emb)

        # 7. head
        x = self.final_layer(x)
       
        # (B, T, output_dim)
        x = x[:, -self.horizon:]
        
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(2, 10, 16).to(device)
    timestep = torch.tensor([1, 2]).to(device)
    target_t = torch.tensor([0.1, 0.2]).to(device)
    vis_cond = torch.randn(2, 256, 256).to(device)
    lang_cond = ["This is a test sentence.", "Another test sentence."]
    model = DiTX(
        input_dim=16,
        output_dim=16,
        horizon=10,
        n_obs_steps=2,
        cond_dim=256,
        visual_cond_len=128,
        diffusion_timestep_embed_dim=256,
        diffusion_target_t_embed_dim=256,
        block_type="DiTX",
        n_layer=2,
        n_head=8,
        n_emb=768,
        mlp_ratio=4.0,
        p_drop_attn=0.1,
        language_conditioned=True,
        pre_norm_modality=True,
    )
    model = model.to(device)
    output = model(sample, timestep, target_t, vis_cond, lang_cond)
    print("Output shape:", output.shape)
    assert output.shape == (2, 10, 16), "Output shape mismatch!"
    print("DiTX with FiLM vision conditioning: OK")