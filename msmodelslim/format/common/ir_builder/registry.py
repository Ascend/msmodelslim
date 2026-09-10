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

from typing import Optional, Type

from torch import nn

from msmodelslim.utils.exception import UnsupportedError

from .base import IrBuilder


def get_builder_by_ir_type(ir_type: Type[nn.Module]) -> IrBuilder:
    """Resolve IrBuilder for an IR class via QABCRegistry."""
    return IrBuilder.create(ir_type)


def try_get_builder_for_module(module: nn.Module) -> Optional[IrBuilder]:
    """Best-effort lookup by ``type(module)`` (exact key match only)."""
    try:
        return IrBuilder.create(type(module))
    except UnsupportedError:
        return None
