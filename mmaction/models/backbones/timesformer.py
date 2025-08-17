from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from mmaction.utils import get_root_logger
from einops import rearrange
from ..builder import BACKBONES

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_frames=8, drop_path=0., has_adapter=True):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        self.t_attn = nn.MultiheadAttention(d_model, n_head)
        self.t_norm = LayerNorm(d_model)
        self.T_Adapter = nn.Linear(d_model, d_model)

        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def t_attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.t_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## temporal attention
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
        # xt = self.attention(self.T_Adapter_in(self.ln_1(xt)))
        xt = self.drop_path(self.t_attention(self.t_norm(xt)))
        xt = self.T_Adapter(xt)
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + xt
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        xn = self.ln_2(x)
        # x = x + self.mlp(xn) + self.drop_path(self.scale * self.S_Adapter(xn))
        x = x + self.drop_path(self.mlp(xn))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_frames, dpr[i], True) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@BACKBONES.register_module()
class TimeSformer(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, adapter_scale=0.5, attn_type='tadapter', pretrained=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.tadapter = False
        self.num_frames = num_frames
        if attn_type == 'tadapter':
            self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
            self.tadapter = True

            # self.temporal_embedding2 = nn.Parameter(torch.zeros(1, num_frames+1, width))
            # self.cls_token_t = nn.Parameter(torch.zeros(1, 1, width))
            # trunc_normal_(self.cls_token_t, std=.02)
            # trunc_normal_(self.temporal_embedding2, std=.02)
            # self.t_blocks = TBlock(width, heads, mlp_ratio=1, num_frames=num_frames)

        self.transformer = Transformer(num_frames, width, layers, heads, scale=adapter_scale, drop_path=drop_path_rate)

        self.ln_post = LayerNorm(width)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            new_state_dict = pretrain_dict.copy()
            for key in pretrain_dict:
                if 'transformer' in key and 'attn' in key:
                    new_key = key.replace('attn','t_attn')
                    if not new_key in pretrain_dict:
                        new_state_dict[new_key] = pretrain_dict[key]
                    else:
                        new_state_dict[new_key] = pretrain_dict[new_key]
                if 'transformer' in key and 'ln_1' in key:
                    new_key = key.replace('ln_1','t_norm')
                    if not new_key in pretrain_dict:
                        new_state_dict[new_key] = pretrain_dict[key]
                    else:
                        new_state_dict[new_key] = pretrain_dict[new_key]
            pretrain_dict = new_state_dict
            msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        ## Space-only
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.tadapter:
            n = x.shape[1]
            x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
            x = x + self.temporal_embedding
            x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        ## Space-only
        # x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
        # x = torch.mean(x, 1) # averaging predictions for every frame
        ## Space-only
        x = self.ln_post(x)  # BDT
        x = x[:, 0]
        x = rearrange(x, '(b t) m -> b m t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        # x = self.head(x)

        return x


if __name__ == '__main__':
    model = TimeSformer(224, 8, 16, 768, 12, 12, 0)
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    image = torch.randn((1, 3, 8, 224, 224))
    flops = FlopCountAnalysis(model, image)
    print(flops.total())
    print(flop_count_table(flops))
    num_param = sum(p.numel() for p in model.parameters())
    print(num_param)