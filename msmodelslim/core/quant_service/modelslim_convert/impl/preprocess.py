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

PreprocessExecutor（convert_design.md §6）。

职责：对原始 ``TensorCatalog`` 应用 ``preprocess_rules``，仅做结构变换（rename / split 等），
不做数值量化。输出供虚拟树使用的逻辑 catalog 与 ``DependencyMap``。
"""

from __future__ import annotations

from tqdm import tqdm

from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.ops import apply_preprocess_ops
from msmodelslim.core.convert.catalog import PreprocessResult, TensorCatalog, TensorEntry, build_dependency_map
from msmodelslim.core.convert.config import WeightMappingRule
from msmodelslim.core.convert.protocol import ConvertContext, IPreprocessExecutor
from msmodelslim.utils.logging import get_logger

logger = get_logger()


class PreprocessExecutor(IPreprocessExecutor):
    """顺序应用每条 ``WeightMappingRule``，并汇总依赖图。"""

    def run(
        self,
        context: ConvertContext,
        raw_catalog: TensorCatalog,
        rules: list[WeightMappingRule],
    ) -> PreprocessResult:
        # 深拷贝 catalog，避免修改 reader 侧缓存
        catalog = TensorCatalog()
        for key, entry in tqdm(raw_catalog.items(), desc="preprocess copy catalog", leave=False):
            catalog.add(
                TensorEntry(
                    key=key,
                    shard=entry.shard,
                    dtype=entry.dtype,
                    shape=entry.shape,
                    meta=dict(entry.meta),
                ),
            )

        num_experts = _resolve_num_experts(context)

        for rule in tqdm(rules, desc="preprocess rules"):
            apply_preprocess_ops(catalog, rule, num_experts=num_experts)

        dep_map = build_dependency_map(catalog.to_weight_map(), catalog=catalog)
        logger.info("Preprocess done: %d catalog keys", len(catalog))
        return PreprocessResult(
            catalog=catalog,
            dependency_map=dep_map,
            applied_rules=[r.id for r in rules],
        )


def _resolve_num_experts(context: ConvertContext) -> int:
    """``split_fused_gate_up`` 所需专家数：优先 model ``config.json``，否则默认 256。"""
    cfg = context.reader.read_model_config() if context.reader is not None else {}
    for candidate in (
        cfg.get("num_experts"),
        (cfg.get("text_config") or {}).get("num_experts"),
        (cfg.get("language_config") or {}).get("num_experts"),
    ):
        if candidate is not None:
            return int(candidate)
    return 256
