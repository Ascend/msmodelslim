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

虚拟模块源 IR 推断与 tensor 绑定（convert_design.md §5.3.2）。

``infer_source_ir``：module_rule 显式声明优先，否则按 tensor_bindings 启发式判断 FP8_BLOCK / INT4 / FLOAT。
``bind_tensors_for_module``：将 ``tensor_map`` 模板 ``{module}`` 展开为 catalog 中的实际 key。
"""

from __future__ import annotations

import fnmatch

from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeModule
from msmodelslim.core.convert.config import ConvertConfig, ModuleRule
from msmodelslim.ir.kernels import WEIGHT_SCALE_INV_SUFFIX
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef


def infer_source_ir(module: ModelFreeModule, config: ConvertConfig) -> SourceIR:
    """解析虚拟模块的源 IR，供 IRRouter 选路。"""
    for rule in config.module_rules:
        if fnmatch.fnmatch(module.full_name, rule.match):
            if rule.source_ir is not None:
                return SourceIR(kind=rule.source_ir, source_format=rule.source_format, evidence=["module_rule"])
            if rule.source_format == "fp8_block":
                return SourceIR(kind=IRKind.FP8_BLOCK, source_format=rule.source_format, evidence=["module_rule"])

    bindings = module.tensor_bindings
    keys = set(bindings.keys())
    if "weight_scale_inv" in keys or any(k.endswith("scale_inv") for k in keys):
        return SourceIR(kind=IRKind.FP8_BLOCK, evidence=["weight_scale_inv"])
    if "weight_packed" in keys:
        return SourceIR(kind=IRKind.INT4_PACKED, evidence=["weight_packed"])
    if "weight" in keys and bindings["weight"].dtype and "float8" in bindings["weight"].dtype.lower():
        return SourceIR(kind=IRKind.FP8_BLOCK, evidence=[bindings["weight"].dtype])
    if "weight" in keys:
        return SourceIR(kind=IRKind.FLOAT, evidence=[bindings["weight"].dtype])
    return SourceIR(kind=IRKind.UNKNOWN, confidence=0.0)


def bind_tensors_for_module(module_path: str, rule: ModuleRule, catalog_keys: set[str]) -> dict[str, TensorRef]:
    """
    按 module_rule.tensor_map 生成逻辑名 -> TensorRef（shard/dtype 由 virtual_tree enrich 后填充）。
    """
    bindings: dict[str, TensorRef] = {}
    for logical, pattern in rule.tensor_map.items():
        key = pattern.replace("{module}", module_path)
        if key not in catalog_keys:
            if logical == "weight_scale":
                inv = module_path + WEIGHT_SCALE_INV_SUFFIX
                if inv in catalog_keys:
                    key, logical = inv, "weight_scale_inv"
        if key in catalog_keys:
            bindings[logical] = TensorRef(logical_name=logical, key=key, shard="", dtype="", shape=())
    return bindings
