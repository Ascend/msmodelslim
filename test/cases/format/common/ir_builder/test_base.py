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

Unit tests for msmodelslim.format.common.ir_builder.base.
"""

from torch import nn

from msmodelslim.format.common.ir_builder.base import ApplyMode, IrBuilder, IrStructureSpec, TransformResult, set_module
from msmodelslim.format.common.ir_builder.w8a8_static import W8A8StaticIrBuilder
from msmodelslim.ir.w8a8_static import W8A8StaticFakeQuantLinear


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity(), nn.Linear(2, 2)])


class TestApplyMode:
    def test_enum_values_are_stable(self):
        assert ApplyMode.REPLACE.value == "replace"
        assert ApplyMode.INSERT.value == "insert"
        assert ApplyMode.SKIP.value == "skip"


class TestTransformResult:
    def test_module_defaults_to_none(self):
        result = TransformResult(mode=ApplyMode.SKIP)

        assert result.mode is ApplyMode.SKIP
        assert result.module is None


class TestSetModule:
    def test_replaces_nested_submodule_by_dotted_key(self):
        model = _Holder()
        replacement = nn.Linear(3, 3)

        set_module(model, "blocks.1", replacement)

        assert model.blocks[1] is replacement


class TestIrBuilder:
    def test_supports_fake_quant_reload_is_true(self):
        assert IrBuilder.supports_fake_quant_reload() is True

    def test_create_returns_builder_registered_for_ir_type(self):
        builder = IrBuilder.create(W8A8StaticFakeQuantLinear)

        assert isinstance(builder, W8A8StaticIrBuilder)


class TestIrStructureSpec:
    def test_is_plain_instantiable_marker(self):
        assert isinstance(IrStructureSpec(), IrStructureSpec)
