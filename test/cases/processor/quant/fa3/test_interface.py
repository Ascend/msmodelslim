#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.processor.quant.fa3.interface 模块的单元测试
"""

# pylint: disable=abstract-class-instantiated

import pytest
import torch
from torch import nn

from msmodelslim.processor.quant.fa3 import FA3QuantAdapterInterface, FA3QuantPlaceHolder
from msmodelslim.utils.exception import SchemaValidateError


class _Adapter(FA3QuantAdapterInterface):
    def __init__(self):
        self.calls = []

    def inject_fa3_placeholders(self, root_name, root_module, should_inject):
        self.calls.append((root_name, root_module, should_inject))


class TestFA3QuantAdapterInterface:
    def test_interface_is_abstract(self):
        with pytest.raises(TypeError):
            FA3QuantAdapterInterface()

    def test_declares_inject_fa3_placeholders_as_abstract(self):
        assert FA3QuantAdapterInterface.__abstractmethods__ == {"inject_fa3_placeholders"}

    def test_complete_subclass_is_instantiable(self):
        adapter = _Adapter()
        adapter.inject_fa3_placeholders("model", nn.Module(), lambda name: name.endswith("fa_q"))
        assert len(adapter.calls) == 1
        assert adapter.calls[0][0] == "model"


class TestFA3QuantPlaceHolder:
    def test_default_ratio_is_one(self):
        placeholder = FA3QuantPlaceHolder()
        assert placeholder.get_ratio() == 1.0

    def test_explicit_ratio_is_preserved(self):
        assert FA3QuantPlaceHolder(ratio=0.25).get_ratio() == 0.25

    def test_forward_is_identity(self):
        placeholder = FA3QuantPlaceHolder(ratio=0.5)
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = placeholder(x)
        assert out is x

    @pytest.mark.parametrize("ratio", [-0.1, 1.1, 2, "0.5", True])
    def test_invalid_ratio_is_rejected(self, ratio):
        with pytest.raises(SchemaValidateError):
            FA3QuantPlaceHolder(ratio=ratio)
