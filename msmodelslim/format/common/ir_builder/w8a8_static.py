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

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from msmodelslim.ir.const import int8_per_channel_sym, int8_per_tensor_asym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.ir.w8a8_static import W8A8StaticFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError

from .base import ApplyMode, IrBuilder, IrStructureSpec, TransformResult, set_module


def _meta_empty(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=torch.device("meta"))


@dataclass
class W8A8StaticStructureSpec(IrStructureSpec):
    weight_shape: Tuple[int, ...]
    input_scale_shape: Tuple[int, ...]
    input_offset_shape: Tuple[int, ...]
    weight_scale_shape: Tuple[int, ...]
    bias_shape: Optional[Tuple[int, ...]] = None


@dataclass
class W8A8StaticParamSet:
    """Plain-tensor param set for W8A8 static FakeQuant Linear (IR-semantic DTO)."""

    weight_int8: torch.Tensor
    input_scale: torch.Tensor
    input_offset: torch.Tensor
    weight_scale: torch.Tensor
    bias: Optional[torch.Tensor]


def _copy_module_params(dst: nn.Module, src: nn.Module, device: torch.device) -> None:
    if any(p.device.type == "meta" for p in dst.parameters()):
        dst.to_empty(device=device)
    with torch.no_grad():
        src_params = dict(src.named_parameters())
        for name, param in dst.named_parameters():
            if name not in src_params:
                continue
            param.copy_(src_params[name].to(device=param.device, dtype=param.dtype))


@QABCRegistry.register(dispatch_key=W8A8StaticFakeQuantLinear, abc_class=IrBuilder)
class W8A8StaticIrBuilder(IrBuilder):
    def transform_shell(
        self,
        model: nn.Module,
        module_name: str,
        source_module: nn.Module,
        *,
        structure: W8A8StaticStructureSpec,
    ) -> TransformResult:
        if not isinstance(structure, W8A8StaticStructureSpec):
            raise SchemaValidateError(
                f"W8A8StaticIrBuilder expects W8A8StaticStructureSpec, got {type(structure).__name__}",
                action="Please pass IR-semantic structure from the Format translator.",
            )
        weight = _meta_empty(structure.weight_shape, torch.int8)
        input_scale = _meta_empty(structure.input_scale_shape, torch.float32)
        input_offset = _meta_empty(structure.input_offset_shape, torch.float32)
        weight_scale = _meta_empty(structure.weight_scale_shape, torch.float32)
        bias = None
        if structure.bias_shape is not None:
            bias = _meta_empty(structure.bias_shape, torch.float32)

        scale_for_ir = input_scale.reshape(()) if input_scale.numel() == 1 else input_scale
        offset_for_ir = input_offset.reshape(()) if input_offset.numel() == 1 else input_offset
        x_q_param = QParam(
            scheme=int8_per_tensor_asym,
            ext={"scale": scale_for_ir, "offset": offset_for_ir},
        )
        w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": weight_scale})
        w_q = QStorage(dtype=QDType.INT8, value=weight)
        new_module = W8A8StaticFakeQuantLinear(x_q_param, w_q_param, w_q, bias)
        set_module(model, module_name, new_module)
        return TransformResult(mode=ApplyMode.REPLACE, module=new_module)

    def bind_from_param_set(self, ir_module: nn.Module, param_set: W8A8StaticParamSet) -> None:
        if not isinstance(ir_module, W8A8StaticFakeQuantLinear):
            raise SchemaValidateError(
                f"Expected W8A8StaticFakeQuantLinear, got {type(ir_module).__name__}",
                action="Please run transform_shell before bind_from_param_set.",
            )
        if not isinstance(param_set, W8A8StaticParamSet):
            raise SchemaValidateError(
                f"Expected W8A8StaticParamSet, got {type(param_set).__name__}",
                action="Please translate Format tensors into a W8A8StaticParamSet first.",
            )
        # Assemble QParam / QStorage from plain tensors (IR knowledge, not Format's).
        scale_for_ir = (
            param_set.input_scale.reshape(()) if param_set.input_scale.numel() == 1 else param_set.input_scale
        )
        offset_for_ir = (
            param_set.input_offset.reshape(()) if param_set.input_offset.numel() == 1 else param_set.input_offset
        )
        x_q_param = QParam(
            scheme=int8_per_tensor_asym,
            ext={"scale": scale_for_ir, "offset": offset_for_ir},
        )
        w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": param_set.weight_scale})
        w_q = QStorage(dtype=QDType.INT8, value=param_set.weight_int8)
        filled = W8A8StaticFakeQuantLinear(x_q_param, w_q_param, w_q, param_set.bias)
        device = torch.device("cpu")
        for p in filled.parameters():
            device = p.device
            break
        _copy_module_params(ir_module, filled, device)
