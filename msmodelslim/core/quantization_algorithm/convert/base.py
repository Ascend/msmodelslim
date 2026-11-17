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

Base class for convert-only processors (data-free, no forward).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel


class BaseConvertProcessor(ABC):
    """
    Convenience base implementing ``IIRTransformProcessor`` flags.

    Subclasses implement ``transform`` only; register with ``IRRouter.register_processor``.
    """

    name: str
    src_ir: IRKind
    dst_ir: IRKind
    requires_forward: bool = False
    requires_calibration: bool = False
    loss_level: str = LossLevel.LOSSY.value

    @abstractmethod
    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module: ...
