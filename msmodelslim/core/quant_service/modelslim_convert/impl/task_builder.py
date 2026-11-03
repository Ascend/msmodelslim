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

DefaultIRTaskBuilder（convert_design.md §9.2）。

遍历虚拟树中 ``ModelFreeLinear``，按 ``convert_rules`` 生成 ``IRTask``。
``inverse_weight_map`` 使用预处理阶段的 ``DependencyMap``。
加载 key 经 ``checkpoint_key`` 取物理名：fused 用 ``fused_from``，rename 用 ``renamed_from``。
"""

from __future__ import annotations

import fnmatch

from torch import nn

from msmodelslim.core.quant_service.modelslim_convert.impl.mxfp8_route import can_direct_mxfp8
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear, checkpoint_key
from msmodelslim.core.convert.auto_routes import resolve_auto_route
from msmodelslim.core.convert.catalog import DependencyMap, TensorCatalog
from msmodelslim.core.convert.config import ConvertRule
from msmodelslim.core.convert.edges import RouteConstraints
from msmodelslim.core.convert.protocol import ConvertContext, IRTaskBuilder
from msmodelslim.core.convert.tasks import IRTask
from msmodelslim.core.convert.types import IRKind
from msmodelslim.utils.exception import UnsupportedError


class DefaultIRTaskBuilder(IRTaskBuilder):
    def build(
        self,
        context: ConvertContext,
        tree: nn.Module,
        catalog: TensorCatalog,
    ) -> list[IRTask]:
        tasks: list[IRTask] = []
        dep_map = context.preprocess_result.dependency_map if context.preprocess_result is not None else DependencyMap()
        if context.preprocess_result is None:
            for key, entry in catalog.items():
                dep_map.add_owner(key, entry.shard)
                renamed = (entry.meta or {}).get("renamed_from")
                if renamed:
                    dep_map.add_owner(renamed, entry.shard)

        for name, mod in tree.named_modules():
            if name == "" or not isinstance(mod, ModelFreeLinear):
                continue
            rule = _match_convert_rule(name, context.config.convert_rules)
            if rule is None or rule.action != "transform":
                continue
            if mod.target_ir is None:
                mod.target_ir = rule.target_ir

            route = _select_convert_route(name, mod, rule)
            constraints = RouteConstraints(explicit_route=route)

            # 加载计划使用物理 key（fused / renamed_from），lazy_init 再按 meta 切片
            load_keys = [checkpoint_key(ref) for ref in mod.tensor_bindings.values()]
            inv_map = dep_map.inverse_load_map(load_keys)

            tasks.append(
                IRTask(
                    module_path=name,
                    source_ir=mod.source_ir,
                    target_ir=rule.target_ir,
                    tensor_bindings=mod.tensor_bindings,
                    inverse_weight_map=inv_map,
                    route_constraints=constraints,
                ),
            )
        return tasks


def _select_convert_route(module_name: str, module: ModelFreeLinear, rule: ConvertRule) -> list[IRKind]:
    source = module.source_ir.kind
    target = rule.target_ir
    fp8_to_mxfp8 = source == IRKind.FP8_BLOCK and target == IRKind.W8A8_MXFP8
    if rule.route == "auto":
        return resolve_auto_route(source, target, direct_mxfp8=fp8_to_mxfp8 and can_direct_mxfp8(module))

    route = normalize_explicit_route(source, list(rule.route))
    if fp8_to_mxfp8 and _has_direct_mxfp8_edge(route) and not can_direct_mxfp8(module):
        raise UnsupportedError(
            f"{module_name}: explicit route FP8_BLOCK -> W8A8_MXFP8 requires E4M3 weight, "
            "E8M0 weight_scale_inv, and a K-block that is a multiple of 32.",
            action="Use route: auto to fall back to FP8_BLOCK -> FLOAT -> W8A8_MXFP8, "
            "or fix the layer dtype and shape.",
        )
    return route


def normalize_explicit_route(source: IRKind, route: list[IRKind]) -> list[IRKind]:
    """显式 route 缺起点时补上 source_ir，避免 FP8_BLOCK -> FP8_BLOCK 自环。"""
    if not route or route[0] != source:
        return [source, *route]
    return list(route)


def _has_direct_mxfp8_edge(route: list[IRKind]) -> bool:
    return any(route[i : i + 2] == [IRKind.FP8_BLOCK, IRKind.W8A8_MXFP8] for i in range(len(route) - 1))


def _match_convert_rule(module_path: str, rules: list[ConvertRule]) -> ConvertRule | None:
    """首个 fnmatch 命中的 convert_rule 生效。"""
    for rule in rules:
        if fnmatch.fnmatch(module_path, rule.match):
            return rule
    return None
