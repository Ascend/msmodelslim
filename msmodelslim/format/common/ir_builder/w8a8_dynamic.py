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
from typing import Optional, Tuple, Union

import torch
from torch import nn

from msmodelslim.ir.const import int8_per_channel_sym, int8_per_group_sym, int8_per_token_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.ir.w8a8_dynamic import (
    W8A8DynamicPerChannelFakeQuantLinear,
    W8A8DynamicPerGroupFakeQuantLinear,
)
from msmodelslim.utils.exception import SchemaValidateError

from .base import ApplyMode, IrBuilder, IrStructureSpec, TransformResult, set_module
from .w8a8_static import _copy_module_params

DynamicFakeQuantLinear = Union[W8A8DynamicPerChannelFakeQuantLinear, W8A8DynamicPerGroupFakeQuantLinear]


def _meta_empty(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=torch.device("meta"))


@dataclass
class W8A8DynamicStructureSpec(IrStructureSpec):
    weight_shape: Tuple[int, ...]
    weight_scale_shape: Tuple[int, ...]
    group_size: int = 0
    weight_offset_shape: Optional[Tuple[int, ...]] = None
    bias_shape: Optional[Tuple[int, ...]] = None


@dataclass
class W8A8DynamicParamSet:
    """Plain-tensor param set for W8A8 dynamic FakeQuant Linear (IR-semantic DTO)."""

    weight_int8: torch.Tensor
    weight_scale: torch.Tensor
    weight_offset: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]
    group_size: int = 0


@QABCRegistry.multi_register(
    dispatch_key=[
        W8A8DynamicPerChannelFakeQuantLinear,
        W8A8DynamicPerGroupFakeQuantLinear,
    ],
    abc_type=IrBuilder,
)
class W8A8DynamicIrBuilder(IrBuilder):
    def transform_shell(
        self,
        model: nn.Module,
        module_name: str,
        source_module: nn.Module,
        *,
        structure: W8A8DynamicStructureSpec,
    ) -> TransformResult:
        if not isinstance(structure, W8A8DynamicStructureSpec):
            raise SchemaValidateError(
                f"W8A8DynamicIrBuilder expects W8A8DynamicStructureSpec, got {type(structure).__name__}",
                action="Please pass IR-semantic structure from the Format translator.",
            )
        weight = _meta_empty(structure.weight_shape, torch.int8)
        bias = None
        if structure.bias_shape is not None:
            bias = _meta_empty(structure.bias_shape, torch.float32)
        x_q_param = QParam(scheme=int8_per_token_sym, ext={})

        if structure.group_size and structure.group_size > 0:
            weight_scale = _meta_empty(structure.weight_scale_shape, torch.float32)
            offset_shape = structure.weight_offset_shape or structure.weight_scale_shape
            weight_offset = _meta_empty(offset_shape, torch.float32)
            w_q_param = QParam(
                scheme=int8_per_group_sym,
                ext={
                    "scale": weight_scale,
                    "offset": weight_offset,
                    "group_size": structure.group_size,
                },
            )
            w_q = QStorage(dtype=QDType.INT8, value=weight)
            new_module: DynamicFakeQuantLinear = W8A8DynamicPerGroupFakeQuantLinear(x_q_param, w_q_param, w_q, bias)
        else:
            weight_scale = _meta_empty(structure.weight_scale_shape, torch.float32)
            w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": weight_scale})
            w_q = QStorage(dtype=QDType.INT8, value=weight)
            new_module = W8A8DynamicPerChannelFakeQuantLinear(x_q_param, w_q_param, w_q, bias)

        set_module(model, module_name, new_module)
        return TransformResult(mode=ApplyMode.REPLACE, module=new_module)

    def bind_from_param_set(self, ir_module: nn.Module, param_set: W8A8DynamicParamSet) -> None:
        if not isinstance(ir_module, (W8A8DynamicPerChannelFakeQuantLinear, W8A8DynamicPerGroupFakeQuantLinear)):
            raise SchemaValidateError(
                f"Expected W8A8 dynamic FakeQuant, got {type(ir_module).__name__}",
                action="Please run transform_shell before bind_from_param_set.",
            )
        if not isinstance(param_set, W8A8DynamicParamSet):
            raise SchemaValidateError(
                f"Expected W8A8DynamicParamSet, got {type(param_set).__name__}",
                action="Please translate Format tensors into a W8A8DynamicParamSet first.",
            )
        # Assemble QParam / QStorage from plain tensors (IR knowledge).
        x_q_param = QParam(scheme=int8_per_token_sym, ext={})
        if param_set.group_size and param_set.group_size > 0:
            weight_offset = param_set.weight_offset
            if weight_offset is None:
                weight_offset = torch.zeros_like(param_set.weight_scale)
            w_q_param = QParam(
                scheme=int8_per_group_sym,
                ext={
                    "scale": param_set.weight_scale,
                    "offset": weight_offset,
                    "group_size": param_set.group_size,
                },
            )
        else:
            weight_scale = param_set.weight_scale
            if weight_scale.ndim > 1 and weight_scale.shape[-1] == 1:
                weight_scale = weight_scale.squeeze(-1)
            w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": weight_scale})
        w_q = QStorage(dtype=QDType.INT8, value=param_set.weight_int8)
        if param_set.group_size and param_set.group_size > 0:
            filled: DynamicFakeQuantLinear = W8A8DynamicPerGroupFakeQuantLinear(
                x_q_param, w_q_param, w_q, param_set.bias
            )
        else:
            filled = W8A8DynamicPerChannelFakeQuantLinear(x_q_param, w_q_param, w_q, param_set.bias)
        device = torch.device("cpu")
        for p in filled.parameters():
            device = p.device
            break
        _copy_module_params(ir_module, filled, device)
