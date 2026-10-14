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

预处理产物：逻辑 TensorCatalog、依赖图、恢复计划（convert_design.md §6、§6.4）。

``TensorCatalog`` 是虚拟树与任务调度的唯一权重索引源；逻辑 key 可与 index.json 物理 key 不同
（例如 preprocess 生成的 per-expert gate_proj.weight，实际数据仍在 fused gate_up_proj shard 内）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from msmodelslim.core.convert.config import WeightOpConfig


@dataclass
class TensorEntry:
    """Catalog 中单条逻辑张量的元数据（载荷延迟加载）。"""

    key: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    meta: dict = field(default_factory=dict)


class TensorCatalog:
    """key -> TensorEntry；预处理后可包含仅存在于 meta 的虚拟 key。"""

    def __init__(self) -> None:
        self._entries: dict[str, TensorEntry] = {}

    def add(self, entry: TensorEntry) -> None:
        self._entries[entry.key] = entry

    def get(self, key: str) -> TensorEntry | None:
        return self._entries.get(key)

    def keys(self) -> Iterator[str]:
        return iter(self._entries)

    def items(self) -> Iterator[tuple[str, TensorEntry]]:
        return iter(self._entries.items())

    def __len__(self) -> int:
        return len(self._entries)

    def to_weight_map(self) -> dict[str, str]:
        """逻辑 key -> shard 文件名（与 index.json 同构，供 DependencyMap 使用）。"""
        return {k: v.shard for k, v in self._entries.items()}

    @classmethod
    def from_raw_weight_map(
        cls,
        weight_map: dict[str, str],
        header_by_key: dict[str, tuple[str, tuple[int, ...]]] | None = None,
    ) -> TensorCatalog:
        """从原始 weight_map 引导 catalog（可选附带 header 缓存）。"""
        catalog = cls()
        header_by_key = header_by_key or {}
        for key, shard in weight_map.items():
            dtype, shape = header_by_key.get(key, ("UNKNOWN", ()))
            catalog.add(TensorEntry(key=key, shard=shard, dtype=dtype, shape=shape))
        return catalog


@dataclass
class RestoreRule:
    """
    保存前反向结构变换规则（例如 per-expert → fused gate_up_proj）。

    ``when`` 通常为 ``before_save``；``merge_gate_up`` 算子尚未在 SaveProcessorAdapter 实现。
    """

    id: str
    source_patterns: list[str]
    target_pattern: str
    ops: list[WeightOpConfig] = field(default_factory=list)
    when: str = "before_save"


class DependencyMap:
    """
    IR 任务加载依赖（类比 model_free_ptq 的 inverse_weight_map 规划）。

    - ``add_owner``：逻辑/物理 key 所在 shard
    - ``add_dependency``：逻辑 key 依赖的其它 key（如 fused_from）
    - ``inverse_load_map``：单任务需要打开的 shard -> tensor 名列表
    """

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}
        self._deps: dict[str, set[str]] = {}

    def add_owner(self, key: str, shard: str) -> None:
        self._owners[key] = shard

    def add_dependency(self, owner: str, dependency: str) -> None:
        self._deps.setdefault(owner, set()).add(dependency)

    def dependencies_of(self, key: str) -> set[str]:
        return set(self._deps.get(key, ()))

    def inverse_load_map(self, keys: list[str]) -> dict[str, list[str] | None]:
        shard_to_names: dict[str, set[str]] = {}
        for key in keys:
            shard = self._owners.get(key)
            if shard is None:
                continue
            shard_to_names.setdefault(shard, set()).add(key)
        return {shard: sorted(names) if names else None for shard, names in shard_to_names.items()}


@dataclass
class PreprocessResult:
    """PreprocessExecutor 输出，挂到 ``ConvertContext.preprocess_result``。"""

    catalog: TensorCatalog
    dependency_map: DependencyMap = field(default_factory=DependencyMap)
    applied_rules: list[str] = field(default_factory=list)


def build_dependency_map(
    weight_map: dict[str, str],
    catalog: TensorCatalog | None = None,
) -> DependencyMap:
    """构建默认依赖图；``fused_from`` meta 会写入 ``add_dependency``。"""
    deps = DependencyMap()
    for key, shard in weight_map.items():
        deps.add_owner(key, shard)
    if catalog is not None:
        for key, entry in catalog.items():
            fused = entry.meta.get("fused_from")
            if fused and fused in weight_map:
                deps.add_dependency(key, fused)
    return deps
