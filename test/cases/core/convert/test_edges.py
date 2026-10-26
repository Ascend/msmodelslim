#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

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

from dataclasses import FrozenInstanceError

import pytest

from msmodelslim.core.convert.edges import RouteConstraints, TransformEdge
from msmodelslim.core.convert.types import IRKind, LossLevel


def test_route_constraints_defaults_are_independent():
    first = RouteConstraints()
    second = RouteConstraints()

    assert first.required_nodes == []
    assert first.forbidden_edges == []
    assert first.explicit_route is None

    first.required_nodes.append(IRKind.FLOAT)
    first.forbidden_edges.append((IRKind.FLOAT, IRKind.FP8_BLOCK))
    assert second.required_nodes == []
    assert second.forbidden_edges == []


def test_transform_edge_defaults_and_frozen_behavior():
    edge = TransformEdge(src_ir=IRKind.FLOAT, dst_ir=IRKind.FP8_BLOCK, processor_name="quantize_fp8")

    assert edge.loss_level is LossLevel.LOSSY
    assert edge.requirements == ()
    assert edge.cost == 1

    with pytest.raises(FrozenInstanceError):
        edge.cost = 2
