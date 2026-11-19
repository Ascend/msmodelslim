#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quantization_algorithm.convert.registry 模块的单元测试
"""

# Prime the quant_service package before importing core.quantization_algorithm.convert to avoid a
# circular import inside msmodelslim.core.quantization_algorithm.convert.__init__.
import msmodelslim.core.quant_service.modelslim_convert.virtual_module  # noqa: F401
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.types import IRKind
from msmodelslim.core.quantization_algorithm.convert.registry import register_convert_processors


class TestRegisterConvertProcessors:
    def test_registers_all_processors_on_given_router(self):
        router = IRRouter()
        returned = register_convert_processors(router)

        assert returned is router
        assert [e.processor_name for e in router.resolve(IRKind.FP8_BLOCK, IRKind.FLOAT)] == [
            "Fp8BlockToFloatProcessor"
        ]
        assert [e.processor_name for e in router.resolve(IRKind.INT4_PACKED, IRKind.FLOAT)] == [
            "Int4PackedToFloatProcessor"
        ]
        assert [e.processor_name for e in router.resolve(IRKind.FLOAT, IRKind.W8A8_MXFP8)] == ["MxFp8QuantProcessor"]
        assert [e.processor_name for e in router.resolve(IRKind.FLOAT, IRKind.W4A4_INT4)] == ["W4A4Int4QuantProcessor"]
        assert [e.processor_name for e in router.resolve(IRKind.FLOAT, IRKind.W8A16_INT8)] == [
            "W8A16Int8QuantProcessor"
        ]
        assert [e.processor_name for e in router.resolve(IRKind.FLOAT, IRKind.W8A16_INT8_ASYM)] == [
            "W8A16Int8AsymQuantProcessor"
        ]
        assert [e.processor_name for e in router.resolve(IRKind.FLOAT, IRKind.W8A8_INT8)] == ["W8A8Int8QuantProcessor"]
        assert [e.processor_name for e in router.resolve(IRKind.FLOAT, IRKind.W4A8_INT4)] == ["W4A8Int4QuantProcessor"]
        assert [e.processor_name for e in router.resolve(IRKind.MXFP4_PACKED, IRKind.W4A8_INT4)] == [
            "MxFp4PackedToFloatProcessor",
            "W4A8Int4QuantProcessor",
        ]
        assert [e.processor_name for e in router.resolve(IRKind.FP8_BLOCK, IRKind.W8A16_INT8)] == [
            "Fp8BlockToFloatProcessor",
            "W8A16Int8QuantProcessor",
        ]
        assert [e.processor_name for e in router.resolve(IRKind.FP8_BLOCK, IRKind.W8A16_INT8_ASYM)] == [
            "Fp8BlockToFloatProcessor",
            "W8A16Int8AsymQuantProcessor",
        ]
        assert [e.processor_name for e in router.resolve(IRKind.FP8_BLOCK, IRKind.W8A8_INT8)] == [
            "Fp8BlockToFloatProcessor",
            "W8A8Int8QuantProcessor",
        ]

    def test_default_registration_is_cached_singleton_when_called_twice(self):
        first = IRRouter.default()
        second = register_convert_processors()

        assert first is second
        assert first is IRRouter.default()

    def test_default_registration_does_not_duplicate_edges_when_called_twice(self):
        router = IRRouter.default()
        register_convert_processors()

        assert len(router.resolve(IRKind.FP8_BLOCK, IRKind.FLOAT)) == 1
