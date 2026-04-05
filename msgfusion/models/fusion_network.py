import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torch
import math
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
import torch.nn.functional as F

class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias

class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

def partition_feature_windows(x, window_size):
    """Split [B,H,W,C] feature maps into non-overlapping windows (flattened)."""
    batch, height, width, channels = x.shape
    ws = window_size
    patched = x.view(batch, height // ws, ws, width // ws, ws, channels)
    return patched.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, channels)


def merge_window_tokens(windows, window_size, H, W):
    """Inverse of ``partition_feature_windows`` for fixed canvas size."""
    ws = window_size
    batch = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(batch, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, H, W, -1)
    return x


def build_relative_position_index(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = build_relative_position_index(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, ass_qkv):
        n_batch_windows, n_tokens, _ = qkv.shape

        qkv = qkv.reshape(n_batch_windows, n_tokens, 3, self.num_heads, self.dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        ass_qkv = ass_qkv.reshape(
            n_batch_windows, n_tokens, 3, self.num_heads, self.dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        ass_q, ass_k, ass_v = ass_qkv[0], ass_qkv[1], ass_qkv[2]

        inv_sqrt_dk = self.scale
        q = torch.mul(q, inv_sqrt_dk)
        ass_q = torch.mul(ass_q, inv_sqrt_dk)

        token_attn = torch.matmul(q, k.transpose(-2, -1))
        aux_token_attn = torch.matmul(ass_q, ass_k.transpose(-2, -1))

        rel_bias = self.meta(self.relative_positions).permute(2, 0, 1).contiguous()

        token_attn = token_attn + rel_bias.unsqueeze(0)
        aux_token_attn = aux_token_attn + rel_bias.unsqueeze(0)

        token_attn = self.softmax(token_attn)
        aux_token_attn = self.softmax(aux_token_attn)

        x = torch.matmul(token_attn, v).transpose(1, 2).reshape(n_batch_windows, n_tokens, self.dim)
        ass_x = torch.matmul(aux_token_attn, ass_v).transpose(1, 2).reshape(n_batch_windows, n_tokens, self.dim)

        return x, ass_x

class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
            self.conv_ass = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.ass_V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)
            self.proj_ass = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, 2*dim, 1)
            self.ass_QK = nn.Conv2d(dim, 2*dim, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:    # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1/4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, vision, ass_vision):
        B, C, H, W = vision.shape
        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(vision)
            ass_V = self.ass_V(ass_vision);

        if self.use_attn:
            QK = self.QK(vision)
            ass_QK = self.ass_QK(ass_vision)

            QKV = torch.cat([QK, V], dim=1)
            ass_QKV = torch.cat([ass_QK, ass_V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            shifted_ass_QKV = self.check_size(ass_QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            shifted_ass_QKV = shifted_ass_QKV.permute(0, 2, 3, 1)

            qkv = partition_feature_windows(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
            ass_qkv = partition_feature_windows(shifted_ass_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows, ass_attn_windows = self.attn(qkv, ass_qkv)

            # merge windows
            shifted_out = merge_window_tokens(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
            ass_shifted_out = merge_window_tokens(ass_attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
            ass_out = ass_shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]

            attn_out = out.permute(0, 3, 1, 2)
            ass_attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                conv_out_ass = self.conv_ass(ass_V)
                out = self.proj(conv_out + attn_out)
                out_ass = self.proj_ass(conv_out_ass + ass_attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(vision)                # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))
        return out, out_ass

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.ass_norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.ass_norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()

        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        self.mlp_ass = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio));

    def forward(self, vision, ass_vision):
        identity = vision
        ass_identity = ass_vision
        if self.use_attn: vision, rescale, rebias = self.norm1(vision)
        if self.use_attn: ass_vision, ass_rescale, ass_rebias = self.ass_norm1(ass_vision)

        vision, ass_vision = self.attn(vision, ass_vision)

        if self.use_attn: vision = vision * rescale + rebias
        if self.use_attn: ass_vision = ass_vision * ass_rescale + ass_rebias

        vision = identity + vision
        ass_vision = ass_identity + ass_vision

        identity = vision
        ass_identity = ass_vision

        if self.use_attn and self.mlp_norm: vision, rescale, rebias = self.norm2(vision)
        if self.use_attn and self.mlp_norm: ass_vision, ass_rescale, ass_rebias = self.ass_norm2(ass_vision)

        vision = self.mlp(vision)
        ass_vision = self.mlp_ass(ass_vision);

        if self.use_attn and self.mlp_norm: vision = vision * rescale + rebias
        if self.use_attn and self.mlp_norm: ass_vision = ass_vision * ass_rescale + ass_rebias


        vision = identity + vision
        ass_vision = ass_identity + ass_vision


        return vision,ass_vision

class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth-attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, vision, ass_vision):
        for blk in self.blocks:
            vision, ass_vision = blk(vision, ass_vision)
        return vision,ass_vision;
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size


        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class VisualEmbeddingReconstructor(nn.Module):
    """Re-pack five ROI vectors into object / regional / scene-level slots."""
    def __init__(self, d_model=1024, num_heads=8):
        super().__init__()
        self.attn_region = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)

    def forward(self, visual_roi_feats):
        _B, _N, _D = visual_roi_feats.shape

        visual_objs = visual_roi_feats[:, :3, :]
        pooled_probe = visual_roi_feats.mean(dim=1, keepdim=True)
        visual_region, _ = self.attn_region(pooled_probe, visual_roi_feats, visual_roi_feats)
        visual_global = visual_roi_feats.mean(dim=1, keepdim=True)
        visual_feats_reconstructed = torch.cat([visual_objs, visual_region, visual_global], dim=1)

        return visual_feats_reconstructed

class DualHierarchicalCrossAttention(nn.Module):
    """Fuse aligned caption/graph tokens, then pool with a learned attention head."""
    def __init__(self, d_model=1024, num_heads=8, fusion_hidden=2048):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(d_model * 2, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, d_model),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, text_feats, image_feats):
        """
        text_feats:  [B, 5, 1024]
        image_feats: [B, 5, 1024]
        """
        assert text_feats.shape == image_feats.shape
        _B, _L, _D = text_feats.shape

        stacked_modalities = torch.cat([text_feats, image_feats], dim=-1)
        per_level = self.project(stacked_modalities)
        pool_query = torch.mean(per_level, dim=1, keepdim=True)
        attn_output, _ = self.attn(query=pool_query, key=per_level, value=per_level)
        return attn_output.squeeze(1)

class AdvancedHierarchicalCrossAttention(nn.Module):
    """Alternate head: scene query attends over object/region language tokens."""
    def __init__(self, d_model=1024, num_heads=8, fusion_hidden=1024):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, d_model),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, text_feats, visual_graph_embed):
        _B, seq_len, _D = text_feats.shape
        assert seq_len == 5

        obj_embed = text_feats[:, 0:3, :]
        region_embed = text_feats[:, 3:4, :]
        global_embed = text_feats[:, 4:5, :]

        fused_query = self.fusion_mlp(torch.cat([global_embed.squeeze(1), visual_graph_embed], dim=-1))
        fused_query = fused_query.unsqueeze(1)

        key_value = torch.cat([obj_embed, region_embed], dim=1)

        fused, _ = self.attn(query=fused_query, key=key_value, value=key_value)
        return fused.squeeze(1)

class HierarchicalFusion(nn.Module):
    """Optional stack: single trainable query attends over L caption tokens."""
    def __init__(self, d_model=1024, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.fusion_query = nn.Parameter(torch.randn(1, 1, d_model))

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=False,
        )

    def forward(self, x):
        batch_sz, _, feat_dim = x.shape
        query = self.fusion_query.expand(1, batch_sz, feat_dim)
        kv = x.transpose(0, 1)
        attn_output, _attn_weights = self.attention(query, kv, kv)
        return attn_output.squeeze(0)

class HierarchicalCrossAttention(nn.Module):
    """Caption-only ablation head (global token queries object+region slots)."""
    def __init__(self, d_model=1024, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        _B, L, _D = x.size()
        assert L == 5, "Expected 5-level text features"

        obj_embed = x[:, 0:3, :]
        region_embed = x[:, 3:4, :]
        global_embed = x[:, 4:5, :]

        key_value = torch.cat([obj_embed, region_embed], dim=1)
        global_query = global_embed

        fused, _weights = self.attn(query=global_query, key=key_value, value=key_value)

        return fused.squeeze(1)

class TextCorrespond(nn.Module):
    def __init__(self, dim, text_channel, amplify=8):
        super(TextCorrespond, self).__init__()

        d = int(dim * amplify)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_vis = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, text_channel, 1, bias=False)
        )
        self.mlp_ir = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, text_channel, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_vis, in_ir, text_features):
        x_vis = self.mlp_vis(in_vis)
        x_ir = self.mlp_ir(in_ir)
        text_map = text_features.view(1, text_features.shape[1], 1, 1).expand_as(x_ir)
        x = x_vis + text_map * x_ir
        return x

class VTFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(VTFusion, self).__init__()

        self.height = height
        bottleneck = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, bottleneck, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(bottleneck, dim * height, 1, bias=False),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        batch_sz, channels, height, width = in_feats[0].shape

        stacked = torch.cat(in_feats, dim=1)
        stacked = stacked.view(batch_sz, self.height, channels, height, width)

        aggregate = torch.sum(stacked, dim=1)

        coeff = self.mlp(self.avg_pool(aggregate))
        coeff = self.softmax(coeff.view(batch_sz, self.height, channels, 1, 1))

        return torch.sum(stacked * coeff, dim=1)

class MSGFusionNet(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(MSGFusionNet, self).__init__()

        # self.hi_fusion = HierarchicalFusion(d_model=1024, num_heads=8)
        # self.hi_fusion = HierarchicalCrossAttention(d_model=1024, num_heads=8)
        # self.hi_fusion = AdvancedHierarchicalCrossAttention(d_model=1024, num_heads=8, fusion_hidden=1024)
        self.visual_reconstructor = VisualEmbeddingReconstructor(d_model=1024, num_heads=8)
        self.hi_fusion = DualHierarchicalCrossAttention(d_model=1024, num_heads=8)
        
        self.height = 2
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.patch_size = 4
        text_channels = 1024
        
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.patch_embed2 = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.layer1 = BasicLayer(
            network_depth=sum(depths),
            dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratios[0],
            norm_layer=norm_layer[0],
            window_size=window_size,
            attn_ratio=attn_ratio[0],
            attn_loc="last",
            conv_type=conv_type[0],
        )
        self.vt_features_fusion = VTFusion(embed_dims[0])
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.fuse_text_image = TextCorrespond(embed_dims[0], text_channels, 2)
        self.patch_unembed1 = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=text_channels, kernel_size=3)

        self.ac = nn.Tanh();

        #self.p


    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, vis, ir, text_features, image_features):
        batch_dim, _, img_h, img_w = vis.shape

        if text_features.dim() == 2 and text_features.size(0) == 5:
            text_features = text_features.unsqueeze(0)
        elif text_features.dim() == 3 and text_features.shape[1] == 5:
            pass
        else:
            raise ValueError("text_features must be [5, 1024] or [B, 5, 1024]")

        roi_aligned = self.visual_reconstructor(image_features)
        language_visual = self.hi_fusion(text_features, roi_aligned)
        language_visual = language_visual.view(batch_dim, -1, 1, 1)

        vision = self.check_image_size(vis)
        ass_vision = self.check_image_size(ir)

        vision = self.patch_embed(vision)
        ass_vision = self.patch_embed2(ass_vision)

        x, ass_vision = self.layer1(vision, ass_vision)
        fused_plane = self.fuse_text_image(x, ass_vision, language_visual)

        x = self.patch_unembed1(fused_plane)
        x = self.ac(x)
        x = x * 0.5 + 0.5
        x = x[:, :, :img_h, :img_w]
        return x

def build_msgfusion_network():
    return MSGFusionNet(
        #embed_dims=[24, 48, 96, 48, 24],
        embed_dims=[24,48],
        #mlp_ratios=[2., 4., 4., 2., 2.],
        mlp_ratios=[2.],
        #depths=[4, 4, 4, 2, 2],
        depths=[1],
        #num_heads=[2, 4, 6, 1, 1],
        num_heads=[2],
        #attn_ratio=[0, 1/2, 1, 0, 0],
        attn_ratio=[1],
        #conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])
        conv_type=['DWConv'])


MSGFusionNet_t = build_msgfusion_network

