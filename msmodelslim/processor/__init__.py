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

__all__ = [
    "AdaptRotationProcessor",
    "AdaptRotationProcessorConfig",
    "AutoProcessorConfig",
    "AutoroundQuantProcessor",
    "TrainableLinearQuantProcessor",
    "TrainableLinearQuantProcessorConfig",
    "UnaryAnalysisProcessorConfig",
    "UnaryAnalysisProcessor",
    "BinaryAnalysisProcessorConfig",
    "BinaryAnalysisProcessor",
    "BinaryOperatorModelWiseProcessorConfig",
    "BinaryOperatorModelWiseProcessor",
    "LinearProcessorConfig",
    "LinearQuantProcessor",
    "SmoothQuantProcessorConfig",
    "SmoothQuantProcessor",
    "IterSmoothProcessorConfig",
    "IterSmoothProcessor",
    "OASQProcessor",
    "OASQProcessorConfig",
    "FlexSmoothQuantProcessorConfig",
    "FlexSmoothQuantProcessor",
    "FlexAWQSSZProcessorConfig",
    "FlexAWQSSZProcessor",
    "LinearQuantProcessor",
    "LoadProcessorConfig",
    "LoadProcessor",
    "GroupProcessorConfig",
    "GroupProcessor",
    "DynamicCacheProcessorConfig",
    "DynamicCacheQuantProcessor",
    "FA3QuantProcessorConfig",
    "FA3QuantProcessor",
    "FloatSparseProcessorConfig",
    "FloatSparseProcessor",
    "QuaRotProcessorConfig",
    "QuaRotProcessor",
    'FlatQuantProcessorConfig',
    'FlatQuantProcessor',
    "AWQProcessorConfig",
    "AWQProcessor",
    'SVDResidualProcessorConfig',
    'SVDResidualProcessor',
]

from .analysis import (
    UnaryAnalysisProcessorConfig,
    UnaryAnalysisProcessor,
    BinaryAnalysisProcessorConfig,
    BinaryAnalysisProcessor,
    BinaryOperatorModelWiseProcessorConfig,
    BinaryOperatorModelWiseProcessor,
)
from .anti_outlier import (
    SmoothQuantProcessorConfig,
    SmoothQuantProcessor,
    IterSmoothProcessorConfig,
    IterSmoothProcessor,
    OASQProcessor,
    OASQProcessorConfig,
    FlexSmoothQuantProcessorConfig,
    FlexSmoothQuantProcessor,
    FlexAWQSSZProcessorConfig,
    FlexAWQSSZProcessor,
)
from .base import AutoProcessorConfig
from .container.group import GroupProcessorConfig, GroupProcessor
from .memory.load import LoadProcessorConfig, LoadProcessor
from .quant.attention import DynamicCacheProcessorConfig, DynamicCacheQuantProcessor
from .quant.autoround import AutoroundQuantProcessor
from .trainable_linear_quant import (
    TrainableLinearQuantProcessor,
    TrainableLinearQuantProcessorConfig,
)
from .quant.fa3 import FA3QuantProcessorConfig, FA3QuantProcessor
from .quant.linear import LinearProcessorConfig, LinearQuantProcessor
from .quarot import QuaRotProcessor, QuaRotProcessorConfig
from .adapt_rotation import AdaptRotationProcessor, AdaptRotationProcessorConfig
from .sparse.float_sparse import FloatSparseProcessorConfig, FloatSparseProcessor
from .flat_quant import FlatQuantProcessorConfig, FlatQuantProcessor
from .anti_outlier import AWQProcessor, AWQProcessorConfig
from .svd_residual import SVDResidualProcessorConfig, SVDResidualProcessor
