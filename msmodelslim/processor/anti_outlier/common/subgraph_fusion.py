#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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
from typing import Dict, Optional

import torch
from torch import nn

from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    UpDownSubgraph,
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    Subgraph,
)
from msmodelslim.core.context import get_current_context


@torch.no_grad()
def apply_smooth_scale_shift(
    layer: nn.Module,
    scales: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
    layer_name: Optional[str] = None,
) -> None:
    """Apply smooth scale and shift to a layer, and optionally record to context.

    Args:
        layer: The layer to apply scales and shifts to
        scales: The scale tensor to apply
        shift: Optional shift tensor to apply
        layer_name: Optional layer name for context recording
    """
    device = layer.weight.device
    dtype = layer.weight.dtype
    layer.weight.mul_(scales)

    # Record scales to context if layer_name is provided and context is available
    if layer_name is not None:
        ctx = get_current_context()
        if ctx is not None:
            namespace = ctx["smooth_scales"]
            namespace.debug[f"{layer_name}.scales"] = scales.cpu()
            if shift is not None:
                namespace.debug[f"{layer_name}.shift"] = shift.cpu()

    if shift is not None:
        shift = shift.to(device).to(dtype)
        if layer.bias is None:
            # If no bias exists, create new bias parameter
            bias_shape = (layer.weight.shape[0],)
            layer.bias = nn.Parameter(torch.zeros(bias_shape, device=device, dtype=dtype, requires_grad=False))
        layer.bias.add_(shift)


class SubgraphFusionStrategy(ABC):
    @abstractmethod
    def apply_fusion(
        self, subgraph: Subgraph, scales: Dict[str, torch.Tensor], shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        pass


class OVSubgraphFusion(SubgraphFusionStrategy):
    @torch.no_grad()
    def apply_fusion(
        self, subgraph: OVSubgraph, scales: Dict[str, torch.Tensor], shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        o_scales = scales['o_scales']
        v_scales = scales['v_scales']

        o_shift = shifts.get('o_shift') if shifts else None
        v_shift = shifts.get('v_shift') if shifts else None

        # Get layer names from subgraph if available
        o_proj_name = getattr(subgraph, "o_proj_name", None)
        v_proj_name = getattr(subgraph, "v_proj_name", None)

        if getattr(subgraph.v_proj, "bias", None) is not None:
            subgraph.v_proj.bias.mul_(1.0 / v_scales)

        apply_smooth_scale_shift(subgraph.o_proj, o_scales.view(1, -1), o_shift, o_proj_name)
        apply_smooth_scale_shift(subgraph.v_proj, 1.0 / v_scales.view(-1, 1), v_shift, v_proj_name)


class UpDownSubgraphFusion(SubgraphFusionStrategy):
    @torch.no_grad()
    def apply_fusion(
        self,
        subgraph: UpDownSubgraph,
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        scales_tensor = scales['scales']

        down_shift = shifts.get('down_shift') if shifts else None
        up_shift = shifts.get('up_shift') if shifts else None

        # Get layer names from subgraph if available
        down_proj_name = getattr(subgraph, "down_proj_name", None)
        up_proj_name = getattr(subgraph, "up_proj_name", None)

        if getattr(subgraph.up_proj, "bias", None) is not None:
            subgraph.up_proj.bias.mul_(1.0 / scales_tensor)

        apply_smooth_scale_shift(subgraph.down_proj, scales_tensor.view(1, -1), down_shift, down_proj_name)
        apply_smooth_scale_shift(subgraph.up_proj, 1.0 / scales_tensor.view(-1, 1), up_shift, up_proj_name)


class LinearLinearSubgraphFusion(SubgraphFusionStrategy):
    @torch.no_grad()
    def apply_fusion(
        self,
        subgraph: LinearLinearSubgraph,
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        scales_tensor = scales['scales']

        linear2_shift = shifts.get('linear2_shift') if shifts else None
        linear1_shift = shifts.get('linear1_shift') if shifts else None

        # Get layer names from subgraph if available
        linear1_name = getattr(subgraph, "linear1_name", None)
        linear2_name = getattr(subgraph, "linear2_name", None)

        if getattr(subgraph.linear1, "bias", None) is not None:
            subgraph.linear1.bias.mul_(1.0 / scales_tensor)

        apply_smooth_scale_shift(subgraph.linear2, scales_tensor.view(1, -1), linear2_shift, linear2_name)
        apply_smooth_scale_shift(
            subgraph.linear1,
            1.0 / scales_tensor.view(-1, 1),
            linear1_shift,
            linear1_name,
        )


class NormLinearSubgraphFusion(SubgraphFusionStrategy):
    @torch.no_grad()
    def apply_fusion(
        self,
        subgraph: NormLinearSubgraph,
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        scales_tensor = scales['scales']
        linear_shifts = shifts.get('linear_shifts', []) if shifts else []
        norm_shift = shifts.get('norm_shift') if shifts else None

        # Get layer names from subgraph if available
        linear_names = getattr(subgraph, "linear_names", None)

        for idx, fc in enumerate(subgraph.linears):
            linear_shift = linear_shifts[idx] if idx < len(linear_shifts) else None
            linear_name = linear_names[idx] if linear_names and idx < len(linear_names) else None
            apply_smooth_scale_shift(fc, scales_tensor.view(1, -1), linear_shift, linear_name)

        # Note: norm layer doesn't have a name field in the subgraph, so we pass None
        if getattr(subgraph.norm, "bias", None) is not None:
            subgraph.norm.bias.mul_(1.0 / scales_tensor)
        apply_smooth_scale_shift(subgraph.norm, (1.0 / scales_tensor).squeeze(), norm_shift, None)


class NonFusionSubgraphFusion(SubgraphFusionStrategy):
    def apply_fusion(
        self,
        subgraph: NonFusionSubgraph,
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        # 暂不支持shift
        scales_tensor = scales['scales']

        # Get layer names from subgraph if available
        linear_names = getattr(subgraph, "linear_names", None)

        for idx, fc in enumerate(subgraph.linears):
            linear_name = linear_names[idx] if linear_names and idx < len(linear_names) else None
            apply_smooth_scale_shift(fc, scales_tensor.view(1, -1), None, linear_name)


class SubgraphFusionFactory:
    _fusers = {
        OVSubgraph: OVSubgraphFusion(),
        UpDownSubgraph: UpDownSubgraphFusion(),
        LinearLinearSubgraph: LinearLinearSubgraphFusion(),
        NormLinearSubgraph: NormLinearSubgraphFusion(),
        NonFusionSubgraph: NonFusionSubgraphFusion(),
    }

    @classmethod
    def get_fuser(cls, subgraph: Subgraph) -> Optional[SubgraphFusionStrategy]:
        for registered_type, fuser in cls._fusers.items():
            if isinstance(subgraph, registered_type):
                return fuser
        return None

    @classmethod
    def apply_fusion_to_subgraph(
        cls, subgraph: Subgraph, scales: Dict[str, torch.Tensor], shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        fuser = cls.get_fuser(subgraph)
        if fuser is None:
            raise ValueError(f"Unsupported subgraph type: {type(subgraph).__name__}")
        fuser.apply_fusion(subgraph, scales, shifts)
