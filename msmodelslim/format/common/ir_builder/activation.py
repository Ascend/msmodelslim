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
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch
from torch import nn

from msmodelslim.ir.activation_dynamic import FakeQuantActivationPerBlock, FakeQuantActivationPerToken
from msmodelslim.ir.fp8_activation_static import FP8FakeQuantActivationPerHead
from msmodelslim.ir.int8_activation_static import INT8FakeQuantActivationPerHead
from msmodelslim.ir.qal import QDType, QParam, QScheme
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.utils.exception import SchemaValidateError

from .base import ApplyMode, IrBuilder, IrStructureSpec, TransformResult, set_module
from .w8a8_static import _copy_module_params


def _meta_empty(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=torch.device("meta"))


@dataclass
class Fa3PerHeadStructureSpec(IrStructureSpec):
    input_scale_shape: Tuple[int, ...]
    scheme: QScheme


@dataclass
class Fa3PerHeadParamSet:
    """Plain-tensor param set for FA3 per-head static activation (IR-semantic DTO)."""

    scale: torch.Tensor
    scheme: QScheme


@dataclass
class Fa3DynamicStructureSpec(IrStructureSpec):
    scheme: QScheme


@dataclass
class Fa3DynamicParamSet:
    """Plain-tensor param set for FA3 dynamic activation (IR-semantic DTO)."""

    scale: Optional[torch.Tensor]  # None for dynamic (no tensors on disk)
    scheme: QScheme


def _per_head_ir_cls(scheme: QScheme) -> Type[nn.Module]:
    if scheme.dtype == QDType.INT8:
        return INT8FakeQuantActivationPerHead
    return FP8FakeQuantActivationPerHead


@QABCRegistry.multi_register(
    dispatch_key=[INT8FakeQuantActivationPerHead, FP8FakeQuantActivationPerHead],
    abc_type=IrBuilder,
)
class Fa3PerHeadIrBuilder(IrBuilder):
    def transform_shell(
        self,
        model: nn.Module,
        module_name: str,
        source_module: nn.Module,
        *,
        structure: Fa3PerHeadStructureSpec,
    ) -> TransformResult:
        del source_module
        if not isinstance(structure, Fa3PerHeadStructureSpec):
            raise SchemaValidateError(
                f"Fa3PerHeadIrBuilder expects Fa3PerHeadStructureSpec, got {type(structure).__name__}",
                action="Please pass IR-semantic structure from the Format translator.",
            )
        scale = _meta_empty(structure.input_scale_shape, torch.float32)
        x_q_param = QParam(scheme=structure.scheme, ext={"scale": scale})
        ir_cls = _per_head_ir_cls(structure.scheme)
        new_module = ir_cls(x_q_param)
        set_module(model, module_name, new_module)
        return TransformResult(mode=ApplyMode.INSERT, module=new_module)

    def bind_from_param_set(self, ir_module: nn.Module, param_set: Fa3PerHeadParamSet) -> None:
        if not isinstance(ir_module, (INT8FakeQuantActivationPerHead, FP8FakeQuantActivationPerHead)):
            raise SchemaValidateError(
                f"Expected FA3 per-head FakeQuantActivation, got {type(ir_module).__name__}",
                action="Please run transform_shell before bind_from_param_set.",
            )
        if not isinstance(param_set, Fa3PerHeadParamSet):
            raise SchemaValidateError(
                f"Expected Fa3PerHeadParamSet, got {type(param_set).__name__}",
                action="Please translate Format tensors into a Fa3PerHeadParamSet first.",
            )
        x_q_param = QParam(scheme=param_set.scheme, ext={"scale": param_set.scale})
        filled = type(ir_module)(x_q_param)
        device = torch.device("cpu")
        for param in filled.parameters():
            device = param.device
            break
        _copy_module_params(ir_module, filled, device)


@QABCRegistry.register(dispatch_key=FakeQuantActivationPerToken, abc_class=IrBuilder)
class Fa3PerTokenIrBuilder(IrBuilder):
    def transform_shell(
        self,
        model: nn.Module,
        module_name: str,
        source_module: nn.Module,
        *,
        structure: Fa3DynamicStructureSpec,
    ) -> TransformResult:
        del source_module
        if not isinstance(structure, Fa3DynamicStructureSpec):
            raise SchemaValidateError(
                f"Fa3PerTokenIrBuilder expects Fa3DynamicStructureSpec, got {type(structure).__name__}",
                action="Please pass IR-semantic structure from the Format translator.",
            )
        new_module = FakeQuantActivationPerToken(QParam(scheme=structure.scheme))
        set_module(model, module_name, new_module)
        return TransformResult(mode=ApplyMode.INSERT, module=new_module)

    def bind_from_param_set(self, ir_module: nn.Module, param_set: Fa3DynamicParamSet) -> None:
        if not isinstance(ir_module, FakeQuantActivationPerToken):
            raise SchemaValidateError(
                f"Expected FakeQuantActivationPerToken, got {type(ir_module).__name__}",
                action="Please run transform_shell before bind_from_param_set.",
            )
        if not isinstance(param_set, Fa3DynamicParamSet):
            raise SchemaValidateError(
                f"Expected Fa3DynamicParamSet, got {type(param_set).__name__}",
                action="Please translate Format tensors into a Fa3DynamicParamSet first.",
            )
        # Dynamic: no tensors to copy, scheme already set by transform_shell.


@QABCRegistry.register(dispatch_key=FakeQuantActivationPerBlock, abc_class=IrBuilder)
class Fa3PerBlockIrBuilder(IrBuilder):
    def transform_shell(
        self,
        model: nn.Module,
        module_name: str,
        source_module: nn.Module,
        *,
        structure: Fa3DynamicStructureSpec,
    ) -> TransformResult:
        del source_module
        if not isinstance(structure, Fa3DynamicStructureSpec):
            raise SchemaValidateError(
                f"Fa3PerBlockIrBuilder expects Fa3DynamicStructureSpec, got {type(structure).__name__}",
                action="Please pass IR-semantic structure from the Format translator.",
            )
        new_module = FakeQuantActivationPerBlock(QParam(scheme=structure.scheme))
        set_module(model, module_name, new_module)
        return TransformResult(mode=ApplyMode.INSERT, module=new_module)

    def bind_from_param_set(self, ir_module: nn.Module, param_set: Fa3DynamicParamSet) -> None:
        if not isinstance(ir_module, FakeQuantActivationPerBlock):
            raise SchemaValidateError(
                f"Expected FakeQuantActivationPerBlock, got {type(ir_module).__name__}",
                action="Please run transform_shell before bind_from_param_set.",
            )
        if not isinstance(param_set, Fa3DynamicParamSet):
            raise SchemaValidateError(
                f"Expected Fa3DynamicParamSet, got {type(param_set).__name__}",
                action="Please translate Format tensors into a Fa3DynamicParamSet first.",
            )
        # Dynamic: no tensors to copy, scheme already set by transform_shell.
