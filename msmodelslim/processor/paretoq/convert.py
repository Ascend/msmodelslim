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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext, IIRTransformProcessor
from msmodelslim.core.convert.types import IRKind, LossLevel

__all__ = [
    "ParetoQToW4A4DynamicProcessor",
    "ParetoQToW4A8DynamicProcessor",
    "get_paretoq_convert_processors",
]


class _ParetoQDeployProcessor:
    """
    convert ``IIRTransformProcessor`` 公共基类。

    不继承 ``BaseConvertProcessor``，避免 ``processor.convert`` 包初始化环
    （registry → virtual_module → modelslim_convert → factory → registry）。
    """

    name: str
    src_ir = IRKind.PARETOQ_FLOAT
    dst_ir: IRKind
    requires_forward = False
    requires_calibration = False
    loss_level = LossLevel.LOSSY.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        # 懒导入，避免 convert ↔ impl 顶层环。
        from msmodelslim.processor.paretoq.impl import ParetoQImpl

        return ParetoQImpl.transform_for_convert(module, context)


class ParetoQToW4A8DynamicProcessor(_ParetoQDeployProcessor):
    """``IIRTransformProcessor``：``PARETOQ_FLOAT → W4A8_DYNAMIC``。"""

    name = "ParetoQToW4A8DynamicProcessor"
    dst_ir = IRKind.W4A8_DYNAMIC


class ParetoQToW4A4DynamicProcessor(_ParetoQDeployProcessor):
    """``IIRTransformProcessor``：``PARETOQ_FLOAT → W4A4_DYNAMIC``。"""

    name = "ParetoQToW4A4DynamicProcessor"
    dst_ir = IRKind.W4A4_DYNAMIC


def get_paretoq_convert_processors() -> List[IIRTransformProcessor]:
    """算法侧闭环：返回 ParetoQ → deploy 所需的 convert 边。"""
    return [
        ParetoQToW4A8DynamicProcessor(),
        ParetoQToW4A4DynamicProcessor(),
    ]
