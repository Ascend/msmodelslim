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

Global optimal IR routes for ``route: auto``.

Given source IR (inferred from virtual-tree tensor dtypes) and target IR
(from convert rules), return the canonical transform chain.
"""

from __future__ import annotations

from msmodelslim.core.convert.types import IRKind
from msmodelslim.utils.exception import UnsupportedError

# (src_ir, dst_ir) -> full route including both endpoints
_BEST_ROUTES: dict[tuple[IRKind, IRKind], list[IRKind]] = {
    (IRKind.FP8_BLOCK, IRKind.FLOAT): [IRKind.FP8_BLOCK, IRKind.FLOAT],
    (IRKind.INT4_PACKED, IRKind.FLOAT): [IRKind.INT4_PACKED, IRKind.FLOAT],
    (IRKind.FP8_BLOCK, IRKind.W8A8_MXFP8): [
        IRKind.FP8_BLOCK,
        IRKind.FLOAT,
        IRKind.W8A8_MXFP8,
    ],
    (IRKind.FLOAT, IRKind.W8A8_MXFP8): [IRKind.FLOAT, IRKind.W8A8_MXFP8],
}


def resolve_auto_route(src_ir: IRKind, dst_ir: IRKind) -> list[IRKind]:
    """Look up the optimal route for (src_ir, dst_ir); raise if unsupported."""
    if src_ir == dst_ir:
        return [src_ir]
    route = _BEST_ROUTES.get((src_ir, dst_ir))
    if route is None:
        raise UnsupportedError(
            f"No auto route from {src_ir.value} to {dst_ir.value}. "
            "Specify an explicit route in convert_rules or extend auto_routes.",
        )
    return list(route)
