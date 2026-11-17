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

from typing import Type, Tuple

import torch

from msmodelslim.core.quantization_schema.qal.qregistry import QFuncRegistry
from msmodelslim.core.quantization_algorithm.anti_outlier.common.subgraph_type import (
    UpDownSubgraph,
)
from msmodelslim.core.quantization_algorithm.anti_outlier.awq.common import AWQConfig, AWQContext
from msmodelslim.core.quantization_algorithm.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    Subgraph,
)
from msmodelslim.utils.logging import get_logger
from msmodelslim.core.quantization_algorithm.anti_outlier.common.subgraph_fusion import SubgraphFusionFactory


@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def awq(subgraph: Subgraph, config: AWQConfig, context: AWQContext) -> None:
    """Activation-aware Weight Quantization (AWQ) for anti-outlier processing.

    Args:
        subgraph: The subgraph to apply AWQ to. Supported types:
            - NonFusionSubgraph
            - NormLinearSubgraph
            - LinearLinearSubgraph
            - OVSubgraph
            - UpDownSubgraph
        config: AWQ algorithm configuration.
        context: Runtime context with activation statistics and block kwargs.

    Returns:
        None
    """
    return QFuncRegistry.dispatch(
        "awq",
        (type(subgraph), config.version),
        *(subgraph, config, context),
    )


def _awq_fusion_apply(subgraph: Subgraph, best_scales: torch.Tensor) -> None:
    """Apply the AWQ scales to the subgraph using SubgraphFusionFactory."""
    get_logger().info("Applying AWQ fusion with best scales: %s", best_scales)
    if isinstance(subgraph, OVSubgraph):
        fusion_scales = {"o_scales": best_scales, "v_scales": best_scales}
    else:
        fusion_scales = {"scales": best_scales}
    SubgraphFusionFactory.apply_fusion_to_subgraph(subgraph, fusion_scales)


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NonFusionSubgraph, 1), api_name="awq")
def awq_impl_non_fusion(subgraph: NonFusionSubgraph, config: AWQConfig, context: AWQContext) -> None:
    """AWQ for NonFusion subgraphs: targets=linears (no source)."""
    get_logger().info("AWQ NonFusion: start")
    linears = subgraph.linears
    if not linears:
        get_logger().warning("AWQ NonFusion: skipped, no linears")
        return
    best_scales = config.awq_searcher.search(linears, context)
    if best_scales is not None:
        _awq_fusion_apply(subgraph, best_scales)
    get_logger().info("AWQ NonFusion: %s", "done" if best_scales is not None else "skipped")


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="awq")
def awq_impl_norm_linear(subgraph: NormLinearSubgraph, config: AWQConfig, context: AWQContext) -> None:
    """AWQ for NormLinear subgraphs: source=norm, targets=linears."""
    get_logger().info("AWQ NormLinear: start")
    linears = subgraph.linears
    best_scales = config.awq_searcher.search(linears, context)
    if best_scales is not None:
        _awq_fusion_apply(subgraph, best_scales)
    get_logger().info("AWQ NormLinear: %s", "done" if best_scales is not None else "skipped")


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="awq")
def awq_impl_linear_linear(subgraph: LinearLinearSubgraph, config: AWQConfig, context: AWQContext) -> None:
    """AWQ for LinearLinear subgraphs: source=linear1, targets=[linear2]."""
    get_logger().info("AWQ LinearLinear: start")
    linears = [subgraph.linear2]
    best_scales = config.awq_searcher.search(linears, context)
    if best_scales is not None:
        _awq_fusion_apply(subgraph, best_scales)
    get_logger().info("AWQ LinearLinear: %s", "done" if best_scales is not None else "skipped")


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="awq")
def awq_impl_ov(subgraph: OVSubgraph, config: AWQConfig, context: AWQContext) -> None:
    """AWQ for OV subgraphs: source=v_proj, targets=[o_proj]."""
    get_logger().info("AWQ OV: start")
    linears = [subgraph.o_proj]
    best_scales = config.awq_searcher.search(linears, context)
    if best_scales is not None:
        _awq_fusion_apply(subgraph, best_scales)
    get_logger().info("AWQ OV: %s", "done" if best_scales is not None else "skipped")


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="awq")
def awq_impl_up_down(subgraph: UpDownSubgraph, config: AWQConfig, context: AWQContext) -> None:
    """AWQ for UpDown subgraphs: source=up_proj, targets=[down_proj]."""
    get_logger().info("AWQ UpDown: start")
    linears = [subgraph.down_proj]
    best_scales = config.awq_searcher.search(linears, context)
    if best_scales is not None:
        _awq_fusion_apply(subgraph, best_scales)
    get_logger().info("AWQ UpDown: %s", "done" if best_scales is not None else "skipped")
