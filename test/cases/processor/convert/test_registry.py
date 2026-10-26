#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.processor.convert.registry 模块的单元测试
"""

# Prime the quant_service package before importing processor.convert to avoid a
# circular import inside msmodelslim.processor.convert.__init__.
import msmodelslim.core.quant_service.modelslim_convert.virtual_module  # noqa: F401
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.types import IRKind
from msmodelslim.processor.convert.registry import register_convert_processors


class TestRegisterConvertProcessors:
    def test_registers_all_processors_on_given_router(self):
        router = IRRouter()
        returned = register_convert_processors(router)

        assert returned is router
        assert [e.processor_name for e in router.resolve(IRKind.FP8_BLOCK, IRKind.FLOAT)] == ["DequantToFloatProcessor"]
        assert [e.processor_name for e in router.resolve(IRKind.INT4_PACKED, IRKind.FLOAT)] == [
            "Int4PackedToFloatProcessor"
        ]
        assert [e.processor_name for e in router.resolve(IRKind.FLOAT, IRKind.W8A8_MXFP8)] == ["MxFp8QuantProcessor"]

    def test_default_registration_is_cached_singleton_when_called_twice(self):
        first = IRRouter.default()
        second = register_convert_processors()

        assert first is second
        assert first is IRRouter.default()

    def test_default_registration_does_not_duplicate_edges_when_called_twice(self):
        router = IRRouter.default()
        register_convert_processors()

        assert len(router.resolve(IRKind.FP8_BLOCK, IRKind.FLOAT)) == 1
