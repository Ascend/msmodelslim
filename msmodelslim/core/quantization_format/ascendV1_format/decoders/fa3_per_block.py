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

AscendV1 FA3 per-block dynamic decoder. Inverse of the per-block saver handler.

  dynamic : no tensors; scheme only from {attn}.quant_type
"""

from typing import Optional, Type

from torch import nn

from msmodelslim.core.quantization_format.common.ir_builder.activation import (
    Fa3DynamicParamSet,
    Fa3DynamicStructureSpec,
)
from msmodelslim.core.quantization_schema.activation import FakeQuantActivationPerBlock, HIF4FakeQuantActivationPerBlock
from msmodelslim.core.quantization_schema.const import (
    fp4_e2m1_per_block_sym,
    hif4_per_block_sym,
    mxfp4_per_block_sym,
    mxfp8_per_block_sym,
)
from msmodelslim.core.quantization_schema.qal import QScheme

from .base import Fa3IrDecoder


class Fa3PerBlockDecoder(Fa3IrDecoder):
    """Decode AscendV1 FA3 per-block dynamic schemes without tensors on disk."""

    _IR_CLS_MAP = {
        mxfp4_per_block_sym: FakeQuantActivationPerBlock,
        mxfp8_per_block_sym: FakeQuantActivationPerBlock,
        fp4_e2m1_per_block_sym: FakeQuantActivationPerBlock,
        hif4_per_block_sym: HIF4FakeQuantActivationPerBlock,
    }

    def ir_type(self, scheme: QScheme) -> Type[nn.Module]:
        return self._IR_CLS_MAP[scheme]

    def structure(self, prefix: str, group_size: int = 0, scheme: Optional[QScheme] = None) -> Fa3DynamicStructureSpec:
        del prefix, group_size
        return Fa3DynamicStructureSpec(scheme=scheme)

    def params(self, prefix: str, device, group_size: int = 0, scheme: Optional[QScheme] = None) -> Fa3DynamicParamSet:
        del prefix, group_size
        return Fa3DynamicParamSet(scale=None, scheme=scheme)
