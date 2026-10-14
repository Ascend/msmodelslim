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
``inverse_weight_map`` 使用预处理阶段的 ``DependencyMap``；fused 逻辑 key 映射到 ``fused_from`` 物理 key。
"""

from __future__ import annotations

import fnmatch

from torch import nn

from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.core.convert.auto_routes import resolve_auto_route
from msmodelslim.core.convert.catalog import DependencyMap, TensorCatalog
from msmodelslim.core.convert.config import ConvertRule
from msmodelslim.core.convert.edges import RouteConstraints
from msmodelslim.core.convert.protocol import ConvertContext, IRTaskBuilder
from msmodelslim.core.convert.tasks import IRTask


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

        for name, mod in tree.named_modules():
            if name == "" or not isinstance(mod, ModelFreeLinear):
                continue
            rule = _match_convert_rule(name, context.config.convert_rules)
            if rule is None or rule.action != "transform":
                continue
            if mod.target_ir is None:
                mod.target_ir = rule.target_ir

            if rule.route == "auto":
                route = resolve_auto_route(mod.source_ir.kind, rule.target_ir)
            else:
                # explicit_route 须以 source_ir 起、target_ir 止。
                # 若 YAML 的 route 已显式包含起点（如 [FP8_BLOCK, FLOAT, W8A8_MXFP8]），
                # 直接采用；否则补上 source_ir 作为起点，避免出现 FP8_BLOCK->FP8_BLOCK 自环边。
                route = list(rule.route)
                if not route or route[0] != mod.source_ir.kind:
                    route = [mod.source_ir.kind, *route]
            constraints = RouteConstraints(explicit_route=route)

            # 加载计划使用物理 key（fused 源），lazy_init 再按 meta 切片
            load_keys = [(ref.meta or {}).get("fused_from") or ref.key for ref in mod.tensor_bindings.values()]
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


def _match_convert_rule(module_path: str, rules: list[ConvertRule]) -> ConvertRule | None:
    """首个 fnmatch 命中的 convert_rule 生效。"""
    for rule in rules:
        if fnmatch.fnmatch(module_path, rule.match):
            return rule
    return None
