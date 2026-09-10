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

from .activation import (
    Fa3DynamicParamSet,
    Fa3DynamicStructureSpec,
    Fa3PerBlockIrBuilder,
    Fa3PerHeadIrBuilder,
    Fa3PerHeadParamSet,
    Fa3PerHeadStructureSpec,
    Fa3PerTokenIrBuilder,
)
from .base import ApplyMode, IrBuilder, IrStructureSpec, TransformResult, set_module
from .kv_cache import KvCacheIrBuilder, KvCacheParamSet, KvCacheStructureSpec
from .registry import get_builder_by_ir_type, try_get_builder_for_module
from .w8a8_dynamic import W8A8DynamicIrBuilder, W8A8DynamicParamSet, W8A8DynamicStructureSpec
from .w8a8_mx_dynamic import (
    DEFAULT_MX_AXES,
    W8A8MXDynamicIrBuilder,
    W8A8MXDynamicParamSet,
    W8A8MXDynamicStructureSpec,
)
from .w8a8_static import W8A8StaticIrBuilder, W8A8StaticParamSet, W8A8StaticStructureSpec

# Import concrete helpers so QABCRegistry.register side effects run.
_ = (
    W8A8StaticIrBuilder,
    W8A8DynamicIrBuilder,
    W8A8MXDynamicIrBuilder,
    Fa3PerHeadIrBuilder,
    Fa3PerTokenIrBuilder,
    Fa3PerBlockIrBuilder,
    KvCacheIrBuilder,
)

__all__ = [
    "ApplyMode",
    "IrBuilder",
    "IrStructureSpec",
    "TransformResult",
    "set_module",
    "get_builder_by_ir_type",
    "try_get_builder_for_module",
    "W8A8StaticIrBuilder",
    "W8A8StaticStructureSpec",
    "W8A8StaticParamSet",
    "W8A8DynamicIrBuilder",
    "W8A8DynamicStructureSpec",
    "W8A8DynamicParamSet",
    "DEFAULT_MX_AXES",
    "W8A8MXDynamicIrBuilder",
    "W8A8MXDynamicStructureSpec",
    "W8A8MXDynamicParamSet",
    "Fa3PerHeadIrBuilder",
    "Fa3PerHeadStructureSpec",
    "Fa3PerHeadParamSet",
    "Fa3PerTokenIrBuilder",
    "Fa3PerBlockIrBuilder",
    "Fa3DynamicStructureSpec",
    "Fa3DynamicParamSet",
    "KvCacheIrBuilder",
    "KvCacheStructureSpec",
    "KvCacheParamSet",
]
