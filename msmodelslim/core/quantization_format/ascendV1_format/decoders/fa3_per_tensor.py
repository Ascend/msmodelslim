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

AscendV1 FA3 per-tensor static decoder. Inverse of the per-tensor saver handler.

  static per-tensor : {prefix}.scale / .offset tagged FAQuant
"""

from typing import Optional, Type

import torch
from torch import nn

from msmodelslim.core.quantization_format.common.ir_builder.activation import (
    Fa3PerTensorParamSet,
    Fa3PerTensorStructureSpec,
)
from msmodelslim.core.quantization_format.common.tensor_utils import require_keys, squeeze_shape
from msmodelslim.core.quantization_schema.const import fp8_e4m3_per_tensor_sym, hif8_per_tensor_sym
from msmodelslim.core.quantization_schema.activation import (
    FP8FakeQuantActivationPerTensor,
    HIF8FakeQuantActivationPerTensor,
)
from msmodelslim.core.quantization_schema.qal import QScheme

from .base import Fa3IrDecoder

FAQUANT_LABEL = "FAQuant"
_STATIC_REQUIRED = ("scale",)
_TENSOR_ACTION = "Please ensure the export was produced by AscendV1Saver per-tensor FA3 handlers."


class Fa3PerTensorDecoder(Fa3IrDecoder):
    """Decode per-tensor static FA3 scale using the same FAQuant tensor contract."""

    _IR_CLS_MAP = {
        hif8_per_tensor_sym: HIF8FakeQuantActivationPerTensor,
        fp8_e4m3_per_tensor_sym: FP8FakeQuantActivationPerTensor,
    }

    def ir_type(self, scheme: QScheme) -> Type[nn.Module]:
        return self._IR_CLS_MAP[scheme]

    def structure(
        self, prefix: str, group_size: int = 0, scheme: Optional[QScheme] = None
    ) -> Fa3PerTensorStructureSpec:
        del group_size
        require_keys(self.store, prefix, _STATIC_REQUIRED, FAQUANT_LABEL, _TENSOR_ACTION)
        return Fa3PerTensorStructureSpec(
            input_scale_shape=squeeze_shape(self.store.get_shape(f"{prefix}.scale")),
            scheme=scheme,
        )

    def params(
        self, prefix: str, device: torch.device, group_size: int = 0, scheme: Optional[QScheme] = None
    ) -> Fa3PerTensorParamSet:
        del group_size
        require_keys(self.store, prefix, _STATIC_REQUIRED, FAQUANT_LABEL, _TENSOR_ACTION)
        scale = self.store.get(f"{prefix}.scale").to(device=device, dtype=torch.float32).reshape(1)
        return Fa3PerTensorParamSet(scale=scale, scheme=scheme)
