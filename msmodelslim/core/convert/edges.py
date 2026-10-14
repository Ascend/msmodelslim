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

IR graph edge definition (shared by router and tasks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from msmodelslim.core.convert.types import IRKind, LossLevel


@dataclass
class RouteConstraints:
    """Explicit routing requirements from convert_rules."""

    required_nodes: list[IRKind] = field(default_factory=list)
    forbidden_edges: list[tuple[IRKind, IRKind]] = field(default_factory=list)
    explicit_route: list[IRKind] | None = None


@dataclass(frozen=True)
class TransformEdge:
    """One edge in the convert IR graph."""

    src_ir: IRKind
    dst_ir: IRKind
    processor_name: str
    loss_level: LossLevel = LossLevel.LOSSY
    requirements: tuple[str, ...] = ()
    cost: int = 1
