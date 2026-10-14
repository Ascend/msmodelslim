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

预处理结构算子（convert_design.md §6.2）。

借鉴 HuggingFace WeightConverter / ConversionOps：本模块只改写 ``TensorCatalog`` 的 key 与 meta，
不在预处理阶段物化大张量；物化推迟到 ``virtual_module.lazy_init`` + ``fused_load``。

已支持算子：
  - ``rename``：正则捕获组重命名 key
  - ``split_fused_gate_up``：Qwen3.5 BF16 MoE 3D gate_up_proj → per-expert gate/up 逻辑 key
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

from msmodelslim.core.convert.catalog import RestoreRule, TensorCatalog, TensorEntry
from msmodelslim.core.convert.config import WeightMappingRule, WeightOpConfig


@dataclass
class PreprocessApplyResult:
    """单条 preprocess_rule 的应用结果。"""

    catalog: TensorCatalog
    restore_rules: list[RestoreRule]
    added_keys: list[str]


def apply_preprocess_ops(
    catalog: TensorCatalog,
    rule: WeightMappingRule,
    num_experts: int | None = None,
) -> PreprocessApplyResult:
    """顺序执行 rule.ops 中的结构操作。"""
    added: list[str] = []
    restore: list[RestoreRule] = []

    for op in rule.ops:
        if op.type == "rename":
            added = _op_rename(catalog, rule, added)
        elif op.type == "split_fused_gate_up":
            n_exp = int(op.params.get("num_experts", num_experts or 0))
            if n_exp <= 0:
                raise ValueError("split_fused_gate_up requires num_experts in op.params or model config.json")
            added, restore = _op_split_fused_gate_up(
                catalog,
                rule,
                n_exp,
                op.params,
                added,
                restore,
            )
        else:
            raise NotImplementedError(f"preprocess op {op.type!r} not implemented")

    return PreprocessApplyResult(catalog, restore, added)


def _op_rename(
    catalog: TensorCatalog,
    rule: WeightMappingRule,
    added: list[str],
) -> list[str]:
    """``re.fullmatch`` + ``expand`` 生成新 key，保留原 shard。"""
    src_pat = rule.source_patterns[0]
    tgt_tmpl = rule.target_patterns[0]
    for key in list(catalog.keys()):
        m = re.fullmatch(src_pat, key)
        if not m:
            continue
        entry = catalog.get(key)
        if entry is None:
            continue
        new_key = m.expand(tgt_tmpl)
        catalog.add(
            TensorEntry(
                key=new_key,
                shard=entry.shard,
                dtype=entry.dtype,
                shape=entry.shape,
                meta={**entry.meta, "renamed_from": key},
            ),
        )
        added.append(new_key)
    return added


def _op_split_fused_gate_up(
    catalog: TensorCatalog,
    rule: WeightMappingRule,
    num_experts: int,
    params: dict,
    added: list[str],
    restore: list[RestoreRule],
) -> tuple[list[str], list[RestoreRule]]:
    """
    将 ``*.mlp.experts.gate_up_proj``（shape [E, 2*I, H]）展开为 E×2 个 2D 逻辑 key。

    每个逻辑 entry 的 ``meta`` 含 ``fused_from`` / ``expert_id`` / ``projection``，
    供 ``fused_load.load_logical_tensor`` 在 lazy_init 时切片。
    """
    pattern = rule.source_patterns[0] if rule.source_patterns else "*.mlp.experts.gate_up_proj"
    split_dim = int(params.get("split_dim", 1))
    chunk_parts = params.get("projections", ["gate_proj", "up_proj"])

    for key in list(catalog.keys()):
        if not fnmatch.fnmatch(key, pattern):
            continue
        entry = catalog.get(key)
        if entry is None:
            continue
        prefix = key[: -len(".gate_up_proj")] if key.endswith(".gate_up_proj") else key

        for expert_id in range(num_experts):
            for proj in chunk_parts:
                logical_key = f"{prefix}.{expert_id}.{proj}.weight"
                catalog.add(
                    TensorEntry(
                        key=logical_key,
                        shard=entry.shard,
                        dtype=entry.dtype,
                        shape=(),
                        meta={
                            "fused_from": key,
                            "layout": "gate_up_interleaved",
                            "expert_id": expert_id,
                            "projection": proj,
                            "split_dim": split_dim,
                            "num_experts": num_experts,
                            "chunk_parts": list(chunk_parts),
                        },
                    ),
                )
                added.append(logical_key)

        catalog._entries.pop(key, None)
        if rule.reversible:
            restore.append(
                RestoreRule(
                    id=f"{rule.id}_restore",
                    source_patterns=[f"{prefix}.*.gate_proj.weight", f"{prefix}.*.up_proj.weight"],
                    target_pattern=key,
                    ops=[WeightOpConfig(type="merge_gate_up", params=params)],
                    when="before_save" if params.get("restore_on_hf", True) else "never",
                ),
            )

    return added, restore
