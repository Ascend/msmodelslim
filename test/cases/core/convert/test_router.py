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

msmodelslim.core.convert.router 模块的单元测试
"""

from unittest.mock import MagicMock

import pytest

from msmodelslim.core.convert.edges import RouteConstraints, TransformEdge
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.utils.exception import UnsupportedError


def create_test_router_with_fp8_mxfp8_edges() -> IRRouter:
    """构造带 FP8→FLOAT→MXFP8 边的测试路由，避免依赖 processor 包循环导入。"""
    router = IRRouter()
    router.register_edge(
        TransformEdge(
            src_ir=IRKind.FP8_BLOCK,
            dst_ir=IRKind.FLOAT,
            processor_name="DequantToFloatProcessor",
            loss_level=LossLevel.LOSSLESS,
        ),
    )
    router.register_edge(
        TransformEdge(
            src_ir=IRKind.FLOAT,
            dst_ir=IRKind.W8A8_MXFP8,
            processor_name="MxFp8QuantProcessor",
            loss_level=LossLevel.LOSSY,
        ),
    )
    return router


class TestIRRouter:
    """测试 IRRouter 类"""

    def test_resolve_return_empty_when_src_equals_dst(self):
        router = IRRouter()
        assert router.resolve(IRKind.FLOAT, IRKind.FLOAT) == []  # 校验同 IR 无需转换

    def test_resolve_raise_error_when_no_route_exists(self):
        router = IRRouter()
        with pytest.raises(UnsupportedError, match="No route"):
            router.resolve(IRKind.INT4_PACKED, IRKind.W8A8_MXFP8)

    def test_resolve_return_fp8_to_mxfp8_chain_when_edges_registered(self):
        router = create_test_router_with_fp8_mxfp8_edges()
        edges = router.resolve(IRKind.FP8_BLOCK, IRKind.W8A8_MXFP8)
        names = [e.processor_name for e in edges]
        assert names == ["DequantToFloatProcessor", "MxFp8QuantProcessor"]  # 校验最短路径

    def test_resolve_return_explicit_route_edges_when_constraints_given(self):
        router = create_test_router_with_fp8_mxfp8_edges()
        constraints = RouteConstraints(
            explicit_route=[IRKind.FP8_BLOCK, IRKind.FLOAT, IRKind.W8A8_MXFP8],
        )
        edges = router.resolve(IRKind.FP8_BLOCK, IRKind.W8A8_MXFP8, constraints=constraints)
        assert [e.dst_ir for e in edges] == [IRKind.FLOAT, IRKind.W8A8_MXFP8]  # 校验显式路由

    def test_validate_route_pass_when_edges_registered(self):
        router = create_test_router_with_fp8_mxfp8_edges()
        router.validate_route([IRKind.FP8_BLOCK, IRKind.FLOAT, IRKind.W8A8_MXFP8])

    def test_get_processor_return_instance_when_registered(self):
        router = IRRouter()
        mock_proc = MagicMock()
        mock_proc.name = "DequantToFloatProcessor"
        mock_proc.src_ir = IRKind.FP8_BLOCK
        mock_proc.dst_ir = IRKind.FLOAT
        mock_proc.requires_forward = False
        mock_proc.requires_calibration = False
        mock_proc.loss_level = "lossless"
        router.register_processor(mock_proc)
        proc = router.get_processor("DequantToFloatProcessor")
        assert proc.name == "DequantToFloatProcessor"  # 校验处理器检索
