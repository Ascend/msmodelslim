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
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from msmodelslim.ir.const import mxfp8_per_block_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.ir.w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError

from .base import ApplyMode, IrBuilder, IrStructureSpec, TransformResult, set_module
from .w8a8_static import _copy_module_params

# Product MX Linear blocks the last dim (LinearQConfig ext["axes"] = -1).
DEFAULT_MX_AXES = -1


def _meta_empty(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=torch.device("meta"))


def _mx_q_params(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_offset: torch.Tensor,
    w_axes: int,
) -> Tuple[QParam, QParam, QStorage]:
    x_q_param = QParam(scheme=mxfp8_per_block_sym, ext={"axes": w_axes})
    w_q_param = QParam(
        scheme=mxfp8_per_block_sym,
        ext={
            "axes": w_axes,
            "scale": weight_scale,
            "offset": weight_offset,
        },
    )
    w_q = QStorage(dtype=QDType.MXFP8, value=weight)
    return x_q_param, w_q_param, w_q


@dataclass
class W8A8MXDynamicStructureSpec(IrStructureSpec):
    weight_shape: Tuple[int, ...]
    weight_scale_shape: Tuple[int, ...]
    w_axes: int = DEFAULT_MX_AXES
    bias_shape: Optional[Tuple[int, ...]] = None


@dataclass
class W8A8MXDynamicParamSet:
    """Plain-tensor param set for W8A8 MX dynamic FakeQuant Linear (IR-semantic DTO)."""

    weight: torch.Tensor  # bfloat16 (MX E4M3 grid)
    weight_scale: torch.Tensor  # bfloat16 shared exponent
    weight_offset: torch.Tensor  # zeros_like(scale)
    bias: Optional[torch.Tensor]
    w_axes: int = DEFAULT_MX_AXES


@QABCRegistry.register(dispatch_key=W8A8MXDynamicPerBlockFakeQuantLinear, abc_class=IrBuilder)
class W8A8MXDynamicIrBuilder(IrBuilder):
    """Rebuild ``W8A8MXDynamicPerBlockFakeQuantLinear`` from AscendV1 ``W8A8_MXFP8`` tensors.

    Format ToIr adapter must invert save then restore pre-save IR dtypes (bf16):
    disk ``weight`` is ``float8_e4m3fn`` → ``.to(bfloat16)`` (MX grid values, NPU-runnable);
    ``weight_scale = (stored_uint8.to(float32) - 127).to(bfloat16)`` then ``unsqueeze(w_axes)``.
    Offset is not exported; bind uses ``zeros_like(scale)``.
    """

    def transform_shell(
        self,
        model: nn.Module,
        module_name: str,
        source_module: nn.Module,
        *,
        structure: W8A8MXDynamicStructureSpec,
    ) -> TransformResult:
        del source_module
        if not isinstance(structure, W8A8MXDynamicStructureSpec):
            raise SchemaValidateError(
                f"W8A8MXDynamicIrBuilder expects W8A8MXDynamicStructureSpec, got {type(structure).__name__}",
                action="Please pass IR-semantic structure from the Format translator.",
            )
        weight = _meta_empty(structure.weight_shape, torch.bfloat16)
        weight_scale = _meta_empty(structure.weight_scale_shape, torch.bfloat16)
        weight_offset = _meta_empty(structure.weight_scale_shape, torch.bfloat16)
        bias = None
        if structure.bias_shape is not None:
            bias = _meta_empty(structure.bias_shape, torch.bfloat16)
        x_q_param, w_q_param, w_q = _mx_q_params(weight, weight_scale, weight_offset, structure.w_axes)
        new_module = W8A8MXDynamicPerBlockFakeQuantLinear(x_q_param, w_q_param, w_q, bias)
        set_module(model, module_name, new_module)
        return TransformResult(mode=ApplyMode.REPLACE, module=new_module)

    def bind_from_param_set(self, ir_module: nn.Module, param_set: W8A8MXDynamicParamSet) -> None:
        if not isinstance(ir_module, W8A8MXDynamicPerBlockFakeQuantLinear):
            raise SchemaValidateError(
                f"Expected W8A8MXDynamicPerBlockFakeQuantLinear, got {type(ir_module).__name__}",
                action="Please run transform_shell before bind_from_param_set.",
            )
        if not isinstance(param_set, W8A8MXDynamicParamSet):
            raise SchemaValidateError(
                f"Expected W8A8MXDynamicParamSet, got {type(param_set).__name__}",
                action="Please translate Format tensors into a W8A8MXDynamicParamSet first.",
            )
        # Assemble QParam / QStorage from plain tensors (IR knowledge).
        x_q_param, w_q_param, w_q = _mx_q_params(
            param_set.weight, param_set.weight_scale, param_set.weight_offset, param_set.w_axes
        )
        filled = W8A8MXDynamicPerBlockFakeQuantLinear(x_q_param, w_q_param, w_q, param_set.bias)
        device = torch.device("cpu")
        for p in filled.parameters():
            device = p.device
            break
        _copy_module_params(ir_module, filled, device)
