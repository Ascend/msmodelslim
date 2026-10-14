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

Offline weight conversion: core types, configuration schemas, and protocols.

This package defines the stable contracts used by ``app/convert`` and
``processor/convert``. Implementations are added incrementally; callers
should depend on these interfaces rather than concrete orchestration code.
"""

from msmodelslim.core.convert.catalog import (
    DependencyMap,
    PreprocessResult,
    RestoreRule,
    TensorCatalog,
    TensorEntry,
)
from msmodelslim.core.convert.config import (
    ConvertConfig,
    ConvertDefaults,
    ConvertRule,
    ModuleRule,
    ParallelConfig,
    WeightMappingRule,
    WeightOpConfig,
)
from msmodelslim.core.convert.protocol import (
    ConvertContext,
    ICheckpointReader,
    IConvertExecutor,
    IIRTransformProcessor,
    IPreprocessExecutor,
    IRTaskBuilder,
    IVirtualModelTreeBuilder,
)
from msmodelslim.core.convert.edges import RouteConstraints, TransformEdge
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.tasks import IRResult, IRTask, RoutedTask
from msmodelslim.core.convert.types import (
    IRKind,
    LossLevel,
    SourceIR,
    TensorRef,
    TensorRole,
)

__all__ = [
    "ConvertConfig",
    "ConvertDefaults",
    "ConvertRule",
    "ModuleRule",
    "ParallelConfig",
    "WeightMappingRule",
    "WeightOpConfig",
    "DependencyMap",
    "PreprocessResult",
    "RestoreRule",
    "TensorCatalog",
    "TensorEntry",
    "ConvertContext",
    "ICheckpointReader",
    "IConvertExecutor",
    "IIRTransformProcessor",
    "IPreprocessExecutor",
    "IRTaskBuilder",
    "IVirtualModelTreeBuilder",
    "RouteConstraints",
    "IRRouter",
    "TransformEdge",
    "IRResult",
    "IRTask",
    "RoutedTask",
    "IRKind",
    "LossLevel",
    "SourceIR",
    "TensorRef",
    "TensorRole",
]
