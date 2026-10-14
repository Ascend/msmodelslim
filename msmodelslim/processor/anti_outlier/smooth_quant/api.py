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
    NormLinearSubgraph,
)
from msmodelslim.processor.anti_outlier.common.subgraph_type import Subgraph
from ..common import SmoothQuantConfig, SmoothContext, IterSmoothScaleCalculator, SubgraphFusionFactory


@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def smooth_quant(subgraph: Subgraph, config: SmoothQuantConfig, context: SmoothContext) -> None:
    """
    使用smooth_quant算法进行异常值抑制
    """
    return QFuncRegistry.dispatch("smooth_quant", (type(subgraph), config.version), *(subgraph, config, context))


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="smooth_quant")
def smooth_quant_impl_norm_linear(subgraph: Subgraph, config: SmoothQuantConfig, context: SmoothContext) -> None:
    # smooth_quant的scale计算逻辑同iter_smooth，仅scale_min固定为1e-5
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=1e-5)
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
