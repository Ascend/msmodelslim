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
# pylint: disable=useless-return

from typing import Type, Tuple

import torch

from msmodelslim.ir.qal.qregistry import QFuncRegistry
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    UpDownSubgraph,
)
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    Subgraph,
)
from msmodelslim.utils.logging import get_logger
from ..common import IterSmoothConfig, SmoothContext, IterSmoothScaleCalculator, SubgraphFusionFactory


@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def iter_smooth(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    """
    使用iter_smooth算法进行异常值抑制

    Args:
        subgraph: 应用iter_smooth算法的子图，支持以下类型：
            NormLinearSubgraph
            LinearLinearSubgraph
            OVSubgraph
            UpDownSubgraph
        config: IterSmooth算法配置
        context: 上下文，用于输入激活的smooth_scale，并记录权重的smooth_scale

    Returns:
        None: 无返回值

    """
    return QFuncRegistry.dispatch("iter_smooth", (type(subgraph), config.version), *(subgraph, config, context))


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_ov(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
    a_scale = context.a_smooth_scale
    w_scale = subgraph.o_proj.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    o_scales, v_scales = calculator.compute_ov_scales(
        a_scale, w_scale, subgraph.num_attention_heads, subgraph.key_value_heads
    )
    shifts = {}
    if config.shift:
        shifts['o_shift'] = torch.mm(context.shift.unsqueeze(0), subgraph.o_proj.weight.data.clone().T).squeeze(0)
        shifts['v_shift'] = context.shift * -1 * (1.0 / scales)
        if subgraph.v_proj.bias is not None:
            subgraph.v_proj.bias.mul_(1.0 / scales)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph, scales={'o_scales': o_scales, 'v_scales': v_scales}, shifts=shifts if shifts else None
    )
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_up_down(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
    a_scale = context.a_smooth_scale
    w_scale = subgraph.down_proj.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        shifts['down_shift'] = torch.mm(context.shift.unsqueeze(0), subgraph.down_proj.weight.data.clone().T).squeeze(0)
        shifts['up_shift'] = context.shift * -1 * (1.0 / scales)
        if subgraph.up_proj.bias is not None:
            subgraph.up_proj.bias.mul_(1.0 / scales)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph, scales={'scales': scales}, shifts=shifts if shifts else None
    )
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_linear_linear(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
    a_scale = context.a_smooth_scale
    w_scale = subgraph.linear2.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        shifts['linear2_shift'] = torch.mm(context.shift.unsqueeze(0), subgraph.linear2.weight.data.clone().T).squeeze(
            0
        )
        shifts['linear1_shift'] = context.shift * -1 * (1.0 / scales)
        if subgraph.linear1.bias is not None:
            subgraph.linear1.bias.mul_(1.0 / scales)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph, scales={'scales': scales}, shifts=shifts if shifts else None
    )
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_norm_linear(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
    a_scale = context.a_smooth_scale
    w_scale = []
    for fc in subgraph.linears:
        fc_weight = fc.weight
        stat = fc_weight.abs().max(dim=0, keepdim=True)[0]
        w_scale.append(stat)
    w_scale = torch.cat(w_scale, dim=0)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        linear_shifts = []
        for fc in subgraph.linears:
            linear_shift = torch.mm(context.shift.unsqueeze(0), fc.weight.data.clone().T).squeeze(0)
            linear_shifts.append(linear_shift)
        shifts['linear_shifts'] = linear_shifts
        shifts['norm_shift'] = context.shift * -1 * (1.0 / scales)
        if subgraph.norm.bias is not None:
            subgraph.norm.bias.mul_(1.0 / scales)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph, scales={'scales': scales}, shifts=shifts if shifts else None
    )
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NonFusionSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_non_fusion_linear(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext):
    """
    Apply iter_smooth to a NonFusionSubgraph for outlier suppression.

    Computes per-channel smooth scales from activation statistics (context.a_smooth_scale)
    and weight statistics (per-output-channel max abs weight across subgraph linears).
    Then applies fusion (weight/norm scaling) via SubgraphFusionFactory and registers
    a forward_pre_hook (NonFusionSmoothQuantHookIR) on each linear so that smooth
    scaling is applied at inference time.
    """

    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
    a_scale = context.a_smooth_scale

    if len(subgraph.linears) < 1:
        raise ValueError("NonFusionSubgraph must have at least one linear layer")

    w_scale = []
    for linear in subgraph.linears:
        stat = linear.weight.abs().max(dim=0, keepdim=True)[0]
        w_scale.append(stat)
    w_scale = torch.cat(w_scale, dim=0)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        get_logger().warning(
            "NonFusionSubgraphFusion does not support shifts; shifts will be ignored.",
        )
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph, scales={'scales': scales}, shifts=shifts if shifts else None
    )

    return scales
