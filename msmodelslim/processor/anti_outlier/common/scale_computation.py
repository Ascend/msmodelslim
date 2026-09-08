#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from msmodelslim.utils.exception import MisbehaviorError
from msmodelslim.utils.logging import get_logger

from .smooth_types import SmoothContext


@torch.no_grad()
def validate_and_process_tensors(context: SmoothContext, tmp_device, dtype) -> torch.Tensor:
    valid_tensors = [
        tensor.to(dtype=dtype).to(tmp_device)
        for tensor in context.tensors
        if tensor is not None and torch.is_tensor(tensor)
    ]
    tensors = torch.cat(valid_tensors, dim=0)
    act = tensors.view(-1, tensors.shape[-1])
    return act


@dataclass
class MQGAScaleParams:
    act_scales: torch.Tensor
    weight_scales: torch.Tensor
    best_alpha: float
    best_beta: float
    num_key_value_groups: int
    head_dim: int


@torch.no_grad()
def compute_weight_scale(weight: torch.Tensor, dtype) -> torch.Tensor:
    weight_abs = weight.abs()
    weight_max = weight_abs.max(dim=0, keepdim=True)[0]
    w_scale = weight_max.max(dim=0)[0]
    w_scale = w_scale.to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
    get_logger().debug("Weight scale shape: %s, min: %.6f, max: %.6f", w_scale.shape, w_scale.min(), w_scale.max())
    return w_scale


@torch.no_grad()
def compute_multi_weight_scale(weights: List[torch.Tensor], dtype) -> torch.Tensor:
    weight_stat = [w.abs().max(dim=0, keepdim=True)[0] for w in weights]
    w_scale = torch.cat(weight_stat, dim=0)
    w_scale = w_scale.max(dim=0)[0].to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
    get_logger().debug(
        "Multi-weight scale shape: %s, min: %.6f, max: %.6f", w_scale.shape, w_scale.min(), w_scale.max()
    )
    return w_scale


@torch.no_grad()
def apply_smooth_scale_shift(layer, scales, shift=None):
    device = layer.weight.device
    dtype = layer.weight.dtype
    layer.weight.mul_(scales)

    if shift is not None:
        shift = shift.to(device).to(dtype)
        if layer.bias is None:
            bias_shape = (layer.weight.shape[0],)
            layer.bias = nn.Parameter(torch.zeros(bias_shape, device=device, dtype=dtype))

        layer.bias.add_(shift)


@torch.no_grad()
def prepare_mqga_parameters(num_attention_heads: int, num_key_value_heads: int) -> Tuple[int, int]:
    ratio = num_attention_heads // num_key_value_heads
    scales_pad_size = 0
    return ratio, scales_pad_size


@torch.no_grad()
def reduce_scales_for_mqga_mean(
    scales: torch.Tensor, shape_ratio: int, num_attention_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce MQGA scales using mean method

    Assumes K V activation heads are broadcast in the following pattern:
    [h1, h2, h3, h4] -> [h1, ... , h1, h2,..., h2, h3, ..., h3, h4, ..., h4]

    Args:
        scales: Scale tensor
        shape_ratio: Shape ratio
        num_attention_heads: Number of attention heads

    Returns:
        Tuple of (updated_scales, reduced_scales)
    """
    num_q_heads = num_attention_heads

    # Validate divisibility: num_q_heads must be divisible by shape_ratio
    if num_q_heads % shape_ratio != 0:
        raise MisbehaviorError(
            f"num_attention_heads ({num_q_heads}) must be divisible by shape_ratio ({shape_ratio}), "
            f"but got remainder {num_q_heads % shape_ratio}",
            action="Please check the model configuration to ensure num_attention_heads is divisible by shape_ratio",
        )

    num_kv_heads = num_q_heads // shape_ratio
    head_emb_size = scales.size(0) // num_q_heads
    reduced_scales, updated_scales = [], []
    copied_scales_slices = scales.split(scales.size(0) // num_kv_heads)

    for gr_idx in range(num_kv_heads):
        subset_of_scales = copied_scales_slices[gr_idx].view(-1, head_emb_size)
        repeat_num = subset_of_scales.size(0)
        reduced_scales.append(subset_of_scales.mean(0))
        updated_scales.append(torch.cat([reduced_scales[-1]] * repeat_num))
    return torch.cat(updated_scales), torch.cat(reduced_scales)


@torch.no_grad()
def reduce_scales_for_mqga_max(params: MQGAScaleParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce MQGA scales using max method

    Args:
        params: MQGA scale parameters

    Returns:
        Tuple of (o_scales, v_scales)
    """
    act_scales = params.act_scales.view(-1, params.num_key_value_groups, params.head_dim).max(dim=1)[0]
    weight_scales = params.weight_scales.view(-1, params.num_key_value_groups, params.head_dim).max(dim=1)[0]
    group_scales = (
        (act_scales.pow(params.best_alpha) / weight_scales.pow(params.best_beta))
        .to(torch.float32)
        .clamp(min=1e-5)
        .to(dtype=weight_scales.dtype)
    )
    o_scales = torch.repeat_interleave(
        group_scales.view(-1, params.head_dim), repeats=params.num_key_value_groups, dim=0
    )
    o_scales = o_scales.reshape(-1)
    v_scales = group_scales.reshape(-1)
    return o_scales, v_scales


class BaseScaleCalculator(ABC):
    """Abstract base class for scale calculators

    Defines the unified interface for scale computation in smooth quantization
    """

    @abstractmethod
    def compute_smooth_scale(self, a_scale: torch.Tensor, w_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_ov_scales(
        self, a_scale: torch.Tensor, w_scale: torch.Tensor, num_attention_heads: int, num_key_value_heads: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class IterSmoothScaleCalculator(BaseScaleCalculator):
    """Scale calculator for IterSmooth

    Features:
    - Uses alpha to control smoothing, beta = 1 - alpha
    - Supports shift parameter
    - Uses mean method for MQGA
    """

    def __init__(self, alpha: float, scale_min: float = 1e-5):
        self.alpha = alpha
        self.scale_min = scale_min

    @torch.no_grad()
    def compute_smooth_scale(self, a_scale: torch.Tensor, w_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        device = w_scale.device
        dtype = w_scale.dtype
        w_scale = w_scale.max(dim=0)[0].to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
        scales = (
            (a_scale.pow(self.alpha).to(device) / w_scale.pow(1 - self.alpha))
            .to(torch.float32)
            .clamp(min=self.scale_min)
            .to(dtype=dtype)
        )
        return scales

    @torch.no_grad()
    def compute_ov_scales(
        self, a_scale: torch.Tensor, w_scale: torch.Tensor, num_attention_heads: int, num_key_value_heads: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scales = self.compute_smooth_scale(a_scale, w_scale)
        shape_ratio, _ = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        o_scales, v_scales = reduce_scales_for_mqga_mean(scales, shape_ratio, num_attention_heads)
        return o_scales, v_scales


class OASQScaleCalculator(BaseScaleCalculator):
    """Scale calculator for OASQ

    Features:
    - Uses max_iters to control Iterative OASQ Approach
    """

    def __init__(
        self,
        base_z_threshold: float = 2.5,
        target_outlier_min: float = 0.01,
        target_outlier_max: float = 0.15,
        max_iters: int = 8,
        z_step: float = 0.2,
    ):
        self.base_z_threshold = base_z_threshold
        self.target_outlier_min = target_outlier_min
        self.target_outlier_max = target_outlier_max
        self.max_iters = max_iters
        self.z_step = z_step

    @staticmethod
    def _reduce_weight_scale(w_scale: torch.Tensor) -> torch.Tensor:
        """Reduce weight tensor to per-input-channel non-negative stats matching ``a_scale``."""
        w = w_scale.to(torch.float32)
        if w.dim() > 1:
            # Prefer abs-max so OASQ sqrt/log1p formulas never see negative magnitudes.
            w = w.abs().amax(dim=tuple(range(w.dim() - 1)))
        return w.reshape(-1)

    @torch.no_grad()
    def compute_smooth_scale(self, a_scale: torch.Tensor, w_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        a = a_scale.to(torch.float32).reshape(-1)
        w = self._reduce_weight_scale(w_scale)
        if w.numel() != a.numel():
            raise ValueError(f"OASQ weight/act scale length mismatch: w={tuple(w.shape)} vs a={tuple(a.shape)}")
        mean = a.mean()
        std = a.std(unbiased=False) + 1e-6
        z_thr = self.base_z_threshold
        best_scales = None
        for _ in range(self.max_iters):
            z_scores = (a - mean) / std
            outlier_mask = torch.abs(z_scores) > z_thr
            outlier_ratio = outlier_mask.float().mean().item()
            scales = torch.ones_like(a)
            normal_mask = ~outlier_mask
            # normal channels
            scales[normal_mask] = torch.sqrt(w[normal_mask]) / (torch.sqrt(a[normal_mask]) + 1e-6)
            # outlier channels
            scales[outlier_mask] = torch.log1p(w[outlier_mask]) / (torch.log1p(a[outlier_mask]) + 1e-6)
            scales = scales / (scales.mean() + 1e-6)
            get_logger().debug("[OASQ] z_thr=%.3f, outlier_ratio=%.4f", z_thr, outlier_ratio)
            if best_scales is None:
                best_scales = scales.clone()
            if self.target_outlier_min <= outlier_ratio <= self.target_outlier_max:
                best_scales = scales.clone()
                break
            if outlier_ratio > self.target_outlier_max:
                z_thr += self.z_step
            elif outlier_ratio < self.target_outlier_min:
                z_thr -= self.z_step

        if best_scales is None:
            raise RuntimeError("OASQ compute_smooth_scale produced no scales; check max_iters")
        return best_scales

    @torch.no_grad()
    def compute_ov_scales(
        self,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        num_attention_heads: int,
        num_key_value_heads: int,
        scales: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scales is None:
            scales = self.compute_smooth_scale(a_scale, w_scale)
        shape_ratio, _ = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        o_scales, v_scales = reduce_scales_for_mqga_mean(scales, shape_ratio, num_attention_heads)
        return o_scales, v_scales


class FlexSmoothScaleCalculator(BaseScaleCalculator):
    """Scale calculator for FlexSmoothQuant

    Features:
    - Uses both alpha and beta parameters
    - Supports two MQGA methods: mean and max
    - Searches for optimal parameters based on quantization error
    """

    def __init__(self, alpha: float, beta: float, group_method: str = 'max'):
        self.alpha = alpha
        self.beta = beta
        self.group_method = group_method

    @torch.no_grad()
    def compute_smooth_scale(self, a_scale: torch.Tensor, w_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        scales = (
            (a_scale.pow(self.alpha) / w_scale.pow(self.beta)).to(torch.float32).clamp(min=1e-5).to(dtype=a_scale.dtype)
        )
        return scales

    @torch.no_grad()
    def compute_ov_scales(
        self, a_scale: torch.Tensor, w_scale: torch.Tensor, num_attention_heads: int, num_key_value_heads: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_ratio, _ = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        head_dim = w_scale.shape[0] // num_attention_heads
        group_method = kwargs.get('group_method', self.group_method)

        if group_method == 'max':
            mqga_params = MQGAScaleParams(
                act_scales=a_scale,
                weight_scales=w_scale,
                best_alpha=self.alpha,
                best_beta=self.beta,
                num_key_value_groups=shape_ratio,
                head_dim=head_dim,
            )
            o_scales, v_scales = reduce_scales_for_mqga_max(mqga_params)
            get_logger().debug(
                "group method: max, O scales shape: %s, V scales shape: %s", o_scales.shape, v_scales.shape
            )
        else:
            scales = self.compute_smooth_scale(a_scale, w_scale)
            o_scales, v_scales = reduce_scales_for_mqga_mean(scales, shape_ratio, num_attention_heads)
            get_logger().debug(
                "group method: mean, O scales shape: %s, V scales shape: %s", o_scales.shape, v_scales.shape
            )

        return o_scales, v_scales


class FlexAWQSSZScaleCalculator(BaseScaleCalculator):
    """Scale calculator for FlexAWQSSZ

    Features:
    - Searches for optimal alpha based on quantization config and actual quantization error
    - Beta is usually fixed at 0
    - Primarily uses max method for MQGA
    - Uses mean of activations instead of max
    """

    def __init__(self, alpha: float, beta: float = 0.0):
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def compute_smooth_scale(self, a_scale: torch.Tensor, w_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        scales = (
            (a_scale.pow(self.alpha) / w_scale.pow(self.beta)).to(torch.float32).clamp(min=1e-5).to(dtype=a_scale.dtype)
        )
        return scales

    @torch.no_grad()
    def compute_ov_scales(
        self, a_scale: torch.Tensor, w_scale: torch.Tensor, num_attention_heads: int, num_key_value_heads: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_ratio, _ = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        head_dim = w_scale.shape[0] // num_attention_heads
        mqga_params = MQGAScaleParams(
            act_scales=a_scale,
            weight_scales=w_scale,
            best_alpha=self.alpha,
            best_beta=self.beta,
            num_key_value_groups=shape_ratio,
            head_dim=head_dim,
        )
        o_scales, v_scales = reduce_scales_for_mqga_max(mqga_params)
        return o_scales, v_scales
