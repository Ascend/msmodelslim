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

KvCacheIrBuilder: IrBuilder for FakeQuantDynamicCache (KV-cache fake-quant IR).

transform_shell  — create a meta-parameter FakeQuantDynamicCache and INSERT it
                   (replacing the FA3QuantPlaceHolder left by inject_fa3_placeholders).
bind_from_param_set — fill the IR module from a plain-tensor KvCacheParamSet
                       (scale / offset decoded by the Format decoder).
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from msmodelslim.ir.attention import FakeQuantDynamicCache
from msmodelslim.ir.qal import QParam
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.utils.exception import SchemaValidateError

from .base import ApplyMode, IrBuilder, IrStructureSpec, TransformResult, set_module


def _meta_empty(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Create an empty tensor on the ``meta`` device (no real storage)."""
    return torch.empty(shape, dtype=dtype, device=torch.device("meta"))


@dataclass
class KvCacheStructureSpec(IrStructureSpec):
    """IR-semantic structure description for FakeQuantDynamicCache."""

    scale_shape: Tuple[int, ...]
    offset_shape: Tuple[int, ...]
    scheme: object  # QScheme


@dataclass
class KvCacheParamSet:
    """Plain-tensor param set for KV-cache FakeQuant (IR-semantic DTO)."""

    kv_cache_scale: torch.Tensor
    kv_cache_offset: torch.Tensor
    scheme: object  # QScheme


def _copy_module_params(dst: nn.Module, src: nn.Module, device: torch.device) -> None:
    """Copy parameters from ``src`` to ``dst`` (materialising meta tensors if needed)."""
    if any(p.device.type == "meta" for p in dst.parameters()):
        dst.to_empty(device=device)
    with torch.no_grad():
        src_params = dict(src.named_parameters())
        for name, param in dst.named_parameters():
            if name not in src_params:
                continue
            param.copy_(src_params[name].to(device=param.device, dtype=param.dtype))


@QABCRegistry.register(dispatch_key=FakeQuantDynamicCache, abc_class=IrBuilder)
class KvCacheIrBuilder(IrBuilder):
    """IrBuilder bound to ``FakeQuantDynamicCache`` via QABCRegistry."""

    def transform_shell(
        self,
        model: nn.Module,
        module_name: str,
        source_module: nn.Module,
        *,
        structure: KvCacheStructureSpec,
    ) -> TransformResult:
        """Create a meta-parameter ``FakeQuantDynamicCache`` and INSERT it into ``model``.

        ``source_module`` is the ``FA3QuantPlaceHolder`` left by
        ``inject_fa3_placeholders``; it is replaced by the real IR module.
        """
        if not isinstance(structure, KvCacheStructureSpec):
            raise SchemaValidateError(
                f"KvCacheIrBuilder expects KvCacheStructureSpec, got {type(structure).__name__}",
                action="Please pass IR-semantic structure from the Format decoder.",
            )
        scale = _meta_empty(structure.scale_shape, torch.float32)
        offset = _meta_empty(structure.offset_shape, torch.float32)
        x_q_param = QParam(
            scheme=structure.scheme,
            ext={"scale": scale, "offset": offset},
        )
        new_module = FakeQuantDynamicCache(x_q_param)
        set_module(model, module_name, new_module)
        return TransformResult(mode=ApplyMode.INSERT, module=new_module)

    def bind_from_param_set(self, ir_module: nn.Module, param_set: KvCacheParamSet) -> None:
        """Fill an already-transformed ``FakeQuantDynamicCache`` from plain tensors."""
        if not isinstance(ir_module, FakeQuantDynamicCache):
            raise SchemaValidateError(
                f"Expected FakeQuantDynamicCache, got {type(ir_module).__name__}",
                action="Please run transform_shell before bind_from_param_set.",
            )
        if not isinstance(param_set, KvCacheParamSet):
            raise SchemaValidateError(
                f"Expected KvCacheParamSet, got {type(param_set).__name__}",
                action="Please translate Format tensors into a KvCacheParamSet first.",
            )
        x_q_param = QParam(
            scheme=param_set.scheme,
            ext={"scale": param_set.kv_cache_scale, "offset": param_set.kv_cache_offset},
        )
        filled = FakeQuantDynamicCache(x_q_param)
        device = torch.device("cpu")
        for p in filled.parameters():
            device = p.device
            break
        _copy_module_params(ir_module, filled, device)
