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

配置文档生成器的稳定数据契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ClassDocSections:
    """配置类 docstring 中可稳定抽取的固定小节。"""

    summary: str
    description: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()


@dataclass
class FieldRecord:
    name: str
    type_name: str
    required: bool
    default: str
    constraint: str
    description: str
    nested_models: List[str] = field(default_factory=list)
    is_union_type: bool = False
    discriminator: Optional[str] = None


@dataclass
class ScenarioRecord:
    """YAML-declared end-to-end configuration scenario."""

    title: str
    description: str = ""
    yaml: str = ""


@dataclass
class ModelRecord:
    qualname: str
    class_name: str
    slug: str
    title: str
    summary: str
    source_rel: str
    yaml_path: str
    category: str
    type_tag: Optional[str]
    fields: List[FieldRecord]
    constraints: List[str]
    nested_refs: List[Tuple[str, str, str]]
    parents: List[Tuple[str, str, str]] = field(default_factory=list)
    # field_path → type 分派基础类显示名（如 select_best → SelectBestConfig）
    dispatch_bases: Dict[str, str] = field(default_factory=dict)
    example_yaml: str = ""
    notes: List[str] = field(default_factory=list)
    scenarios: List[ScenarioRecord] = field(default_factory=list)
    # 类 docstring 首段（无 docstring 时为空字符串）；用于嵌套配置块的类级描述
    class_summary: str = ""
    # 类 docstring「说明」小节；保持为独立条目，避免混入配置概述
    class_description: List[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return f"{self.slug}.md"
