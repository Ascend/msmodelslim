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

AscendV1 FA3 per-head static decoder. Inverse of AscendV1Saver per-head handlers.

  static per-head : {prefix}.scale / .offset tagged FAQuant
  Handles both INT8 and FP8 static per-head; the scheme is provided by the caller.
"""

from typing import Optional, Type

import torch
from torch import nn

from msmodelslim.format.common.ir_builder.activation import Fa3PerHeadParamSet, Fa3PerHeadStructureSpec
from msmodelslim.format.common.tensor_utils import require_keys, squeeze_shape
from msmodelslim.ir.const import fp8_e4m3_per_head_sym, int8_per_head_sym
from msmodelslim.ir.fp8_activation_static import FP8FakeQuantActivationPerHead
from msmodelslim.ir.int8_activation_static import INT8FakeQuantActivationPerHead
from msmodelslim.ir.qal import QScheme

from .base import Fa3IrDecoder

FAQUANT_LABEL = "FAQuant"
_STATIC_REQUIRED = ("scale",)
_STATIC_ACTION = "Please ensure the export was produced by AscendV1Saver per-head FA3 handlers."


class Fa3PerHeadDecoder(Fa3IrDecoder):
    """Decode AscendV1 FA3 per-head static tensors (INT8 + FP8)."""

    _IR_CLS_MAP = {
        int8_per_head_sym: INT8FakeQuantActivationPerHead,
        fp8_e4m3_per_head_sym: FP8FakeQuantActivationPerHead,
    }

    def ir_type(self, scheme: QScheme) -> Type[nn.Module]:
        return self._IR_CLS_MAP[scheme]

    def structure(self, prefix: str, group_size: int = 0, scheme: Optional[QScheme] = None) -> Fa3PerHeadStructureSpec:
        del group_size
        store = self.store
        require_keys(store, prefix, _STATIC_REQUIRED, FAQUANT_LABEL, _STATIC_ACTION)
        return Fa3PerHeadStructureSpec(
            input_scale_shape=squeeze_shape(store.get_shape(f"{prefix}.scale")),
            scheme=scheme,
        )

    def params(
        self, prefix: str, device: torch.device, group_size: int = 0, scheme: Optional[QScheme] = None
    ) -> Fa3PerHeadParamSet:
        del group_size
        store = self.store
        require_keys(store, prefix, _STATIC_REQUIRED, FAQUANT_LABEL, _STATIC_ACTION)
        scale = store.get(f"{prefix}.scale").to(device=device, dtype=torch.float32)
        while scale.ndim > 1 and scale.shape[-1] == 1:
            scale = scale.squeeze(-1)
        return Fa3PerHeadParamSet(scale=scale, scheme=scheme)
