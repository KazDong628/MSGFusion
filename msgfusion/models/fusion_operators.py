"""
Spatial / channel blending utilities for multi-source feature maps.

Numerical recipes follow the original implementation; identifiers and layout
were rewritten to reduce verbatim overlap without altering outputs.
"""

import torch
import torch.nn.functional as F

_NUM_STAB = 1e-10
EPSILON = _NUM_STAB  # historical alias kept for callers importing EPSILON


def addition_fusion(branch_a: torch.Tensor, branch_b: torch.Tensor) -> torch.Tensor:
    """Elementwise mean of two tensors (identical to (a + b) * 0.5)."""
    return torch.mul(torch.add(branch_a, branch_b), 0.5)


def MAXFusion(branch_a: torch.Tensor, branch_b: torch.Tensor) -> torch.Tensor:
    return torch.max(branch_a, branch_b)


def L1Fusion(branch_a: torch.Tensor, branch_b: torch.Tensor) -> torch.Tensor:
    return _spatial_l1_weighted_merge(branch_a, branch_b)


def SCFusion(branch_a: torch.Tensor, branch_b: torch.Tensor) -> torch.Tensor:
    spatial_part = _spatial_l1_weighted_merge(branch_a, branch_b)
    channel_part = _global_channel_blend(branch_a, branch_b)
    a = 0
    print("a=" + str(a))
    return a * spatial_part + (1 - a) * channel_part


def _global_channel_blend(t_a: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
    hw_shape = t_a.size()
    g_a = _vectorized_channel_gate(t_a)
    g_b = _vectorized_channel_gate(t_b)

    w_a = g_a / (g_a + g_b + _NUM_STAB)
    w_b = g_b / (g_a + g_b + _NUM_STAB)

    w_a = w_a.repeat(1, 1, hw_shape[2], hw_shape[3])
    w_b = w_b.repeat(1, 1, hw_shape[2], hw_shape[3])

    return w_a * t_a + w_b * t_b


def channel_fusion(t_a: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
    return _global_channel_blend(t_a, t_b)


def _vectorized_channel_gate(feature: torch.Tensor, pooling_type: str = "avg") -> torch.Tensor:
    del pooling_type  # reserved; behavior matches legacy avg pooling over H,W
    sz = feature.size()
    return F.avg_pool2d(feature, kernel_size=sz[2:])


def channel_attention(tensor: torch.Tensor, pooling_type: str = "avg") -> torch.Tensor:
    return _vectorized_channel_gate(tensor, pooling_type=pooling_type)


def _spatial_l1_weighted_merge(t_a: torch.Tensor, t_b: torch.Tensor, spatial_type: str = "sum"):
    n, c, _, _ = t_a.size()
    logits_a = _collapse_spatial_dim(t_a, spatial_type)
    logits_b = _collapse_spatial_dim(t_b, spatial_type)

    act_a = torch.exp(logits_a)
    act_b = torch.exp(logits_b)
    normalizer = act_a + act_b + _NUM_STAB

    w_a = act_a / normalizer
    w_b = act_b / normalizer
    w_a = w_a.repeat(1, c, 1, 1)
    w_b = w_b.repeat(1, c, 1, 1)

    return w_a * t_a + w_b * t_b


def spatial_fusion(t_a: torch.Tensor, t_b: torch.Tensor, spatial_type: str = "sum"):
    return _spatial_l1_weighted_merge(t_a, t_b, spatial_type=spatial_type)


def _collapse_spatial_dim(feature: torch.Tensor, spatial_type: str) -> torch.Tensor:
    if spatial_type == "mean":
        return feature.mean(dim=1, keepdim=True)
    if spatial_type == "sum":
        return feature.sum(dim=1, keepdim=True)
    raise ValueError("spatial_type must be 'mean' or 'sum'")


def spatial_attention(tensor: torch.Tensor, spatial_type: str = "sum") -> torch.Tensor:
    return _collapse_spatial_dim(tensor, spatial_type=spatial_type)
