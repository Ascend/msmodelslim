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

# 按 scope / include / exclude / 子图类型过滤 ``AdapterConfig`` 列表。

from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence

from msmodelslim.utils.config_map import ConfigSet

from .adapter_types import AdapterConfig

EntryModule = Literal["source_or_first_target", "source_only"]


def adapter_entry_module_name(
    adapter_config: AdapterConfig,
    entry: EntryModule = "source_or_first_target",
) -> Optional[str]:
    """用于 scope / include / exclude 匹配的模块名。"""
    mapping = adapter_config.mapping
    if not mapping:
        return None
    if entry == "source_only":
        return mapping.source
    if mapping.source:
        return mapping.source
    if mapping.targets:
        return mapping.targets[0]
    return None


def filter_adapter_configs(
    adapter_configs: Sequence[AdapterConfig],
    scope: str,
    enabled_subgraph_types: Sequence[str],
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    entry: EntryModule = "source_or_first_target",
) -> List[AdapterConfig]:
    """过滤 adapter 子图配置（与 ``BaseSmoothProcessor`` 原逻辑一致）。"""
    enabled = set(enabled_subgraph_types)
    layer_prefix = f"{scope}." if scope else ""
    include_set = ConfigSet(include) if include else ConfigSet(["*"])
    exclude_set = ConfigSet(exclude) if exclude else ConfigSet([])

    result: List[AdapterConfig] = []
    for adapter_config in adapter_configs:
        if adapter_config.subgraph_type not in enabled:
            continue
        module_name = adapter_entry_module_name(adapter_config, entry)
        if not module_name:
            continue
        if not module_name.startswith(layer_prefix):
            continue
        if module_name not in include_set:
            continue
        if module_name in exclude_set:
            continue
        result.append(adapter_config)
    return result


def filter_adapter_configs_for_processor(
    adapter_configs: Sequence[AdapterConfig],
    config: Any,
    scope: str,
    enabled_subgraph_types: Optional[Sequence[str]] = None,
    entry: EntryModule = "source_or_first_target",
) -> List[AdapterConfig]:
    """按 Processor / Op 配置对象过滤（``enable_subgraph_type`` + include/exclude）。"""
    enabled = enabled_subgraph_types
    if enabled is None:
        enabled = config.enable_subgraph_type
    return filter_adapter_configs(
        adapter_configs,
        scope=scope,
        enabled_subgraph_types=enabled,
        include=getattr(config, "include", None),
        exclude=getattr(config, "exclude", None),
        entry=entry,
    )
