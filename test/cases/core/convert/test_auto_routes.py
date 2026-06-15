#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.convert.auto_routes 模块的单元测试
"""

import pytest

from msmodelslim.core.convert.auto_routes import resolve_auto_route
from msmodelslim.core.convert.types import IRKind
from msmodelslim.utils.exception import UnsupportedError


class TestResolveAutoRoute:
    """测试 resolve_auto_route 函数"""

    def test_resolve_auto_route_return_fp8_to_float_when_src_fp8_dst_float(self):
        route = resolve_auto_route(IRKind.FP8_BLOCK, IRKind.FLOAT)
        assert route == [IRKind.FP8_BLOCK, IRKind.FLOAT]

    def test_resolve_auto_route_return_fp8_to_mxfp8_via_float_when_src_fp8_dst_mxfp8(self):
        route = resolve_auto_route(IRKind.FP8_BLOCK, IRKind.W8A8_MXFP8)
        assert route == [IRKind.FP8_BLOCK, IRKind.FLOAT, IRKind.W8A8_MXFP8]

    def test_resolve_auto_route_return_float_to_mxfp8_when_src_float_dst_mxfp8(self):
        route = resolve_auto_route(IRKind.FLOAT, IRKind.W8A8_MXFP8)
        assert route == [IRKind.FLOAT, IRKind.W8A8_MXFP8]

    def test_resolve_auto_route_return_int4_packed_to_float_when_src_int4_dst_float(self):
        route = resolve_auto_route(IRKind.INT4_PACKED, IRKind.FLOAT)
        assert route == [IRKind.INT4_PACKED, IRKind.FLOAT]

    def test_resolve_auto_route_return_same_ir_when_src_equals_dst(self):
        route = resolve_auto_route(IRKind.FLOAT, IRKind.FLOAT)
        assert route == [IRKind.FLOAT]  # 校验同 IR 仅返回自身

    def test_resolve_auto_route_raise_error_when_pair_not_in_table(self):
        with pytest.raises(UnsupportedError, match="No auto route"):
            resolve_auto_route(IRKind.INT4_PACKED, IRKind.W8A8_MXFP8)
