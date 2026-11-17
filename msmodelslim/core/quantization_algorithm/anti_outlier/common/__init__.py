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

Anti-outlier common utilities.
"""

from .fused_linear import VirtualVModuleFromQKVFused, VirtualVModuleFromKVFused
from .smooth_components import HookManager, StatsCollector, SubgraphRegistry, StatKey
from .smooth_types import (
    SmoothQuantContext,
    SmoothQuantConfig,
    IterSmoothContext,
    IterSmoothConfig,
    OASQConfig,
    OASQContext,
    FlexSmoothQuantContext,
    FlexSmoothQuantConfig,
    FlexAWQSSZContext,
    FlexAWQSSZConfig,
    SmoothContext,
)
from .scale_computation import (
    BaseScaleCalculator,
    FlexAWQSSZScaleCalculator,
    FlexSmoothScaleCalculator,
    IterSmoothScaleCalculator,
    OASQScaleCalculator,
    MQGAScaleParams,
    apply_smooth_scale_shift,
    compute_multi_weight_scale,
    compute_weight_scale,
    prepare_mqga_parameters,
    reduce_scales_for_mqga_max,
    reduce_scales_for_mqga_mean,
    validate_and_process_tensors,
)
from .subgraph_fusion import (
    LinearLinearSubgraphFusion,
    NormLinearSubgraphFusion,
    OVSubgraphFusion,
    SubgraphFusionFactory,
    SubgraphFusionStrategy,
    UpDownSubgraphFusion,
)

__all__ = [
    # Fused Linear
    'VirtualVModuleFromQKVFused',
    'VirtualVModuleFromKVFused',
    # Smooth Components
    'HookManager',
    'StatsCollector',
    'SubgraphRegistry',
    'StatKey',
    # Schema Types
    'SmoothQuantContext',
    'SmoothQuantConfig',
    'IterSmoothContext',
    'IterSmoothConfig',
    'OASQConfig',
    'OASQContext',
    'FlexSmoothQuantContext',
    'FlexSmoothQuantConfig',
    'FlexAWQSSZContext',
    'FlexAWQSSZConfig',
    'SmoothContext',
    # Scale Computation
    'IterSmoothScaleCalculator',
    'OASQScaleCalculator',
    'FlexSmoothScaleCalculator',
    'FlexAWQSSZScaleCalculator',
    'BaseScaleCalculator',
    'validate_and_process_tensors',
    'compute_weight_scale',
    'compute_multi_weight_scale',
    'apply_smooth_scale_shift',
    'prepare_mqga_parameters',
    'reduce_scales_for_mqga_mean',
    'reduce_scales_for_mqga_max',
    'MQGAScaleParams',
    # Subgraph Fusion
    'SubgraphFusionFactory',
    'SubgraphFusionStrategy',
    'OVSubgraphFusion',
    'UpDownSubgraphFusion',
    'LinearLinearSubgraphFusion',
    'NormLinearSubgraphFusion',
]
