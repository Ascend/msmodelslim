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
# pylint: disable=logging-fstring-interpolation,useless-return

from typing import Type, Tuple

import torch
from torch import nn

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
from .alpha_beta_search import FlexSmoothAlphaBetaSearcher, FlexAWQSSZAlphaBetaSearcher
from ..common import (
    FlexSmoothQuantConfig,
    FlexAWQSSZConfig,
    SmoothContext,
    validate_and_process_tensors,
    compute_weight_scale,
    compute_multi_weight_scale,
    FlexSmoothScaleCalculator,
    FlexAWQSSZScaleCalculator,
    SubgraphFusionFactory,
)


@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def flex_smooth_quant(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    """
    使用flex_smooth_quant算法进行异常值抑制

    Args:
        subgraph: 应用flex_smooth_quant算法的子图，支持以下类型：
            NormLinearSubgraph
            LinearLinearSubgraph
            OVSubgraph
            UpDownSubgraph
            NonFusionSubgraph（非融合子图）
        config: FlexSmoothQuant算法配置
        context: 上下文，用于输入激活的smooth_scale，并记录权重的smooth_scale

    Returns:
        None: 无返回值

    """
    return QFuncRegistry.dispatch("flex_smooth_quant", (type(subgraph), config.version), *(subgraph, config, context))


@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def flex_awq_ssz(subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext) -> None:
    """
    使用flex_awq_ssz算法进行异常值抑制
    """
    return QFuncRegistry.dispatch("flex_awq_ssz", (type(subgraph), config.version), *(subgraph, config, context))


# ============== FlexSmoothQuant Implementation ==============


def get_optimal_alpha_beta_flex_smooth(
    config: FlexSmoothQuantConfig, act: torch.Tensor, fc_weights: torch.Tensor
) -> Tuple[float, float]:
    if config.alpha is None or config.beta is None:
        searcher = FlexSmoothAlphaBetaSearcher(act_sym=True, search_step=0.05)
        best_alpha, best_beta, final_mse = searcher.search_alpha_beta(act, fc_weights)
        get_logger().debug("Found optimal alpha: %.6f, beta: %.6f, final MSE: %.6f", best_alpha, best_beta, final_mse)
    else:
        best_alpha = config.alpha
        best_beta = config.beta
        get_logger().debug("Using provided alpha: %s, beta: %s", best_alpha, best_beta)
    return best_alpha, best_beta


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_ov(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.v_proj.parameters()).device
    dtype = subgraph.o_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = subgraph.o_proj.weight

    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_weight_scale(subgraph.o_proj.weight, dtype)

    group_method = 'mean'
    if config.extra_config is not None and config.extra_config.get('group_method') == 'max':
        group_method = 'max'

    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta, group_method=group_method)
    o_scales, v_scales = calculator.compute_ov_scales(
        a_scale, w_scale, subgraph.num_attention_heads, subgraph.key_value_heads
    )

    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'o_scales': o_scales, 'v_scales': v_scales})
    get_logger().debug("OV smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_up_down(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.up_proj.parameters()).device
    dtype = subgraph.down_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = subgraph.down_proj.weight
    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_weight_scale(subgraph.down_proj.weight, dtype)
    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'scales': scales})
    get_logger().debug("Up-Down smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_linear_linear(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.linear1.parameters()).device
    dtype = subgraph.linear2.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = subgraph.linear2.weight
    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_weight_scale(subgraph.linear2.weight, dtype)
    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'scales': scales})
    get_logger().debug("Linear-Linear smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_norm_linear(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.norm.parameters()).device
    dtype = subgraph.linears[0].weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = torch.cat([fc.weight for fc in subgraph.linears], dim=0)
    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_multi_weight_scale([fc.weight for fc in subgraph.linears], dtype)
    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'scales': scales})
    get_logger().debug("Norm-Linear smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NonFusionSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_non_fusion_linear(
    subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext
) -> torch.Tensor:
    """
    对 NonFusionSubgraph 应用 flex_smooth_quant 进行异常值抑制（非融合子图接口）。

    从 context 获取激活，对 subgraph.linears 的权重做 alpha/beta 搜索得到最优 scale，
    通过 SubgraphFusionFactory 做权重融合，并在每个 linear 上注册 NonFusionSmoothQuantHookIR，
    在推理时对输入做 scale 校正。
    """
    if len(subgraph.linears) < 1:
        raise ValueError("NonFusionSubgraph must have at least one linear layer")

    tmp_device = next(subgraph.linears[0].parameters()).device
    dtype = subgraph.linears[0].weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = torch.cat([fc.weight for fc in subgraph.linears], dim=0)
    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_multi_weight_scale([fc.weight for fc in subgraph.linears], dtype)
    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta, group_method='max')
    scales = calculator.compute_smooth_scale(a_scale, w_scale)

    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'scales': scales})
    return scales


# ============== FlexAWQSSZ Implementation ==============


@torch.no_grad()
def get_optimal_alpha_beta(config: FlexAWQSSZConfig, act, linear: nn.Linear):
    if config.alpha is None:
        searcher = FlexAWQSSZAlphaBetaSearcher(qconfig=config.qconfig)
        best_alpha, normal_mse_best = searcher.search_alpha(act, linear)
        get_logger().debug(f"Found optimal alpha: {best_alpha:.6f}, MSE: {normal_mse_best:.6f}")
    else:
        best_alpha = config.alpha
        get_logger().debug(f"Using provided alpha: {best_alpha}")
    best_beta = 0
    return best_alpha, best_beta


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_ov(subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.v_proj.parameters()).device
    dtype = subgraph.o_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    # flex awq ssz 的 act 尺度计算使用mean
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.o_proj)

    w_scale = compute_weight_scale(subgraph.o_proj.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    o_scales, v_scales = calculator.compute_ov_scales(
        act_scales, w_scale, subgraph.num_attention_heads, subgraph.key_value_heads
    )

    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'o_scales': o_scales, 'v_scales': v_scales})
    get_logger().debug("OV smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_up_down(subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.up_proj.parameters()).device
    dtype = subgraph.down_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.down_proj)
    w_scale = compute_weight_scale(subgraph.down_proj.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'scales': scales})
    get_logger().debug("Up-Down smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_linear_linear(subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.linear1.parameters()).device
    dtype = subgraph.linear2.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.linear2)
    w_scale = compute_weight_scale(subgraph.linear2.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'scales': scales})
    get_logger().debug("Linear-Linear smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_norm_linear(subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.norm.parameters()).device
    dtype = subgraph.linears[0].weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    # 仅用前两层做校准
    if len(subgraph.linears) > 3:
        fc_weights = torch.cat([fc.weight for fc in subgraph.linears[0:2]], dim=0)
    else:
        fc_weights = torch.cat([fc.weight for fc in subgraph.linears], dim=0)

    get_logger().debug("Activation scale shape: %s, Weight shape: %s", act_scales.shape, fc_weights.shape)
    merged_linear = nn.Linear(
        in_features=fc_weights.shape[1],
        out_features=fc_weights.shape[0],
        bias=False,
        device=fc_weights.device,
        dtype=fc_weights.dtype,
    )
    merged_linear.weight.data = fc_weights
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, merged_linear)

    weights = [fc.weight for fc in subgraph.linears]
    w_scale = compute_multi_weight_scale(weights, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    get_logger().debug("Computed smooth scales shape: %s", scales.shape)
    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, scales={'scales': scales})
    get_logger().debug("Norm-Linear smoothing completed successfully")
    return
