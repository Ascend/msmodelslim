#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.processor.convert.base 模块的单元测试
"""

# pylint: disable=abstract-class-instantiated

import pytest
from torch import nn

# Prime the quant_service package before importing processor.convert to avoid a
# circular import inside msmodelslim.processor.convert.__init__.
import msmodelslim.core.quant_service.modelslim_convert.virtual_module  # noqa: F401
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.processor.convert.base import BaseConvertProcessor


class _DemoConvertProcessor(BaseConvertProcessor):
    """Minimal concrete processor exercising the convenience contract."""

    name = "DemoConvertProcessor"
    src_ir = IRKind.FLOAT
    dst_ir = IRKind.W8A8_MXFP8

    def transform(self, module, context):
        return module


class TestBaseConvertProcessorDefaults:
    def test_flags_default_to_data_free(self):
        proc = _DemoConvertProcessor()
        assert proc.requires_forward is False
        assert proc.requires_calibration is False

    def test_loss_level_defaults_to_lossy(self):
        assert _DemoConvertProcessor().loss_level == LossLevel.LOSSY.value

    def test_declared_name_and_ir_are_preserved(self):
        proc = _DemoConvertProcessor()
        assert proc.name == "DemoConvertProcessor"
        assert proc.src_ir is IRKind.FLOAT
        assert proc.dst_ir is IRKind.W8A8_MXFP8

    def test_transform_is_invoked_from_subclass(self):
        module = nn.Linear(3, 3)
        assert _DemoConvertProcessor().transform(module, None) is module


class TestBaseConvertProcessorAbstractness:
    def test_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseConvertProcessor()

    def test_transform_is_declared_abstract(self):
        assert "transform" in BaseConvertProcessor.__abstractmethods__

    def test_subclass_without_transform_stays_abstract(self):
        class _Incomplete(BaseConvertProcessor):
            name = "Incomplete"
            src_ir = IRKind.FLOAT
            dst_ir = IRKind.FLOAT

        with pytest.raises(TypeError):
            _Incomplete()
