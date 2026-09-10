#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

AscendV1 FA3 per-token dynamic decoder. Inverse of AscendV1Saver dynamic handlers.

  dynamic : no tensors; scheme only from {attn}.quant_type
  Handles both INT8 and FP8 dynamic per-token; the scheme is provided by the caller
  (it never needs to be re-derived from a dtype string).
"""

from typing import Optional, Type

from torch import nn

from msmodelslim.format.common.ir_builder.activation import Fa3DynamicParamSet, Fa3DynamicStructureSpec
from msmodelslim.ir.activation_dynamic import FakeQuantActivationPerToken
from msmodelslim.ir.qal import QScheme

from .base import Fa3IrDecoder


class Fa3PerTokenDecoder(Fa3IrDecoder):
    """Decode AscendV1 FA3 per-token dynamic (INT8 + FP8). No tensors on disk."""

    def ir_type(self, scheme: QScheme) -> Type[nn.Module]:
        del scheme
        return FakeQuantActivationPerToken

    def structure(self, prefix: str, group_size: int = 0, scheme: Optional[QScheme] = None) -> Fa3DynamicStructureSpec:
        del prefix, group_size
        return Fa3DynamicStructureSpec(scheme=scheme)

    def params(self, prefix: str, device, group_size: int = 0, scheme: Optional[QScheme] = None) -> Fa3DynamicParamSet:
        del prefix, group_size, device
        return Fa3DynamicParamSet(scale=None, scheme=scheme)
