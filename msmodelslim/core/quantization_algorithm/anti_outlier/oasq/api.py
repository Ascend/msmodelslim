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

from typing import Optional, Type, Tuple

import torch

from msmodelslim.core.quantization_schema.qal.qregistry import QFuncRegistry
from msmodelslim.core.quantization_algorithm.anti_outlier.common.subgraph_type import (
    UpDownSubgraph,
)
from msmodelslim.core.quantization_algorithm.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    Subgraph,
)
from msmodelslim.utils.logging import get_logger
from ..common import (
    OASQConfig,
    SmoothContext,
    OASQScaleCalculator,
    SubgraphFusionFactory,
)


def _oasq_calculator(config: OASQConfig) -> OASQScaleCalculator:
    if config.max_iters is None:
        return OASQScaleCalculator()
    return OASQScaleCalculator(max_iters=config.max_iters)


def _require_shift_tensor(config: OASQConfig, context: SmoothContext) -> torch.Tensor:
    if context.shift is None:
        raise ValueError(
            "OASQConfig.shift=True requires context.shift to be set "
            "(asymmetric activation statistics must be collected)."
        )
    return context.shift


@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def oasq(subgraph: Subgraph, config: OASQConfig, context: SmoothContext) -> Optional[torch.Tensor]:
    """
    OASQ (Outlier-Aware Smoothing Quantization)

    Args:
        subgraph: type of Subgraph for OASQ
            NormLinearSubgraph
            LinearLinearSubgraph
            OVSubgraph
            UpDownSubgraph
        config: OASQ Config
        context: 上下文，用于输入激活的smooth_scale，并记录权重的smooth_scale

    Returns:
        None for fusion subgraphs; NonFusion returns the computed scales tensor.
    """
    return QFuncRegistry.dispatch(
        "oasq",
        (type(subgraph), config.version),
        *(subgraph, config, context),
    )


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="oasq")
def oasq_impl_ov(subgraph: Subgraph, config: OASQConfig, context: SmoothContext) -> None:
    calculator = _oasq_calculator(config)
    a_scale = context.a_smooth_scale
    # Raw weight; OASQScaleCalculator reduces with abs-max (IterSmooth-style call site).
    w_scale = subgraph.o_proj.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    o_scales, v_scales = calculator.compute_ov_scales(
        a_scale,
        w_scale,
        subgraph.num_attention_heads,
        subgraph.key_value_heads,
        scales=scales,
    )
    shifts = {}
    if config.shift:
        shift = _require_shift_tensor(config, context)
        shifts["o_shift"] = torch.mm(shift.unsqueeze(0), subgraph.o_proj.weight.data.clone().T).squeeze(0)
        shifts["v_shift"] = shift * -1 * (1.0 / scales)
        # bias scaling is applied once inside OVSubgraphFusion
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={"o_scales": o_scales, "v_scales": v_scales},
        shifts=shifts if shifts else None,
    )


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="oasq")
def oasq_impl_up_down(subgraph: Subgraph, config: OASQConfig, context: SmoothContext) -> None:
    calculator = _oasq_calculator(config)
    a_scale = context.a_smooth_scale
    w_scale = subgraph.down_proj.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        shift = _require_shift_tensor(config, context)
        shifts["down_shift"] = torch.mm(shift.unsqueeze(0), subgraph.down_proj.weight.data.clone().T).squeeze(0)
        shifts["up_shift"] = shift * -1 * (1.0 / scales)
        # bias scaling is applied once inside UpDownSubgraphFusion
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={"scales": scales},
        shifts=shifts if shifts else None,
    )


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="oasq")
def oasq_impl_linear_linear(subgraph: Subgraph, config: OASQConfig, context: SmoothContext) -> None:
    calculator = _oasq_calculator(config)
    a_scale = context.a_smooth_scale
    w_scale = subgraph.linear2.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        shift = _require_shift_tensor(config, context)
        shifts["linear2_shift"] = torch.mm(shift.unsqueeze(0), subgraph.linear2.weight.data.clone().T).squeeze(0)
        shifts["linear1_shift"] = shift * -1 * (1.0 / scales)
        # bias scaling is applied once inside LinearLinearSubgraphFusion
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={"scales": scales},
        shifts=shifts if shifts else None,
    )


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="oasq")
def oasq_impl_norm_linear(subgraph: Subgraph, config: OASQConfig, context: SmoothContext) -> None:
    calculator = _oasq_calculator(config)
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
        shift = _require_shift_tensor(config, context)
        linear_shifts = []
        for fc in subgraph.linears:
            linear_shift = torch.mm(shift.unsqueeze(0), fc.weight.data.clone().T).squeeze(0)
            linear_shifts.append(linear_shift)
        shifts["linear_shifts"] = linear_shifts
        shifts["norm_shift"] = shift * -1 * (1.0 / scales)
        # bias scaling is applied once inside NormLinearSubgraphFusion
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={"scales": scales},
        shifts=shifts if shifts else None,
    )


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NonFusionSubgraph, 1), api_name="oasq")
def oasq_impl_non_fusion_linear(subgraph: Subgraph, config: OASQConfig, context: SmoothContext) -> torch.Tensor:
    """
    Apply OASQ to a NonFusionSubgraph for outlier suppression.

    Computes per-channel smooth scales from activation statistics (context.a_smooth_scale)
    and weight statistics (per-output-channel max abs weight across subgraph linears).
    Then applies fusion (weight/norm scaling) via SubgraphFusionFactory and registers
    a forward_pre_hook (NonFusionSmoothQuantHookIR) on each linear so that smooth
    scaling is applied at inference time.
    """
    calculator = _oasq_calculator(config)
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
        subgraph,
        scales={"scales": scales},
        shifts=shifts if shifts else None,
    )

    return scales
