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

Anti-outlier processor module.

Provides processors for outlier suppression algorithms. Config classes are imported
for registration with AutoProcessorConfig; API imports register quantization helpers.
"""

from .awq import AWQProcessorConfig, AWQProcessor
from .common import HookManager, StatsCollector, SubgraphRegistry
from .flex_smooth import (
    FlexSmoothQuantProcessorConfig,
    FlexSmoothQuantProcessor,
    FlexAWQSSZProcessorConfig,
    FlexAWQSSZProcessor,
)
from .flex_smooth.api import flex_smooth_quant, flex_awq_ssz
from .iter_smooth import IterSmoothProcessorConfig, IterSmoothProcessor
from .iter_smooth.api import iter_smooth
from .oasq import OASQProcessorConfig, OASQProcessor
from .smooth_quant import SmoothQuantProcessorConfig, SmoothQuantProcessor
from .smooth_quant.api import smooth_quant

__all__ = [
    # Processors
    "SmoothQuantProcessor",
    "SmoothQuantProcessorConfig",
    "IterSmoothProcessor",
    "IterSmoothProcessorConfig",
    "OASQProcessor",
    "OASQProcessorConfig",
    "FlexSmoothQuantProcessor",
    "FlexSmoothQuantProcessorConfig",
    "FlexAWQSSZProcessor",
    "FlexAWQSSZProcessorConfig",
    "AWQProcessor",
    "AWQProcessorConfig",
    "SubgraphRegistry",
    "HookManager",
    "StatsCollector",
    # API registration side-effects (exported for explicit re-export)
    "flex_smooth_quant",
    "flex_awq_ssz",
    "iter_smooth",
    "smooth_quant",
]
