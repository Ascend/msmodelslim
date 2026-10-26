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

Unit tests for msmodelslim.format.common.ir_builder.registry.
"""

import pytest
from torch import nn

from msmodelslim.format.common.ir_builder.registry import get_builder_by_ir_type, try_get_builder_for_module
from msmodelslim.format.common.ir_builder.w8a8_static import W8A8StaticIrBuilder
from msmodelslim.ir.w8a8_static import W8A8StaticFakeQuantLinear
from msmodelslim.utils.exception import UnsupportedError


class TestGetBuilderByIrType:
    def test_returns_registered_builder_for_supported_ir_type(self):
        assert isinstance(get_builder_by_ir_type(W8A8StaticFakeQuantLinear), W8A8StaticIrBuilder)

    def test_raises_unsupported_error_for_unknown_ir_type(self):
        with pytest.raises(UnsupportedError):
            get_builder_by_ir_type(nn.Linear)


class TestTryGetBuilderForModule:
    def test_returns_none_for_unregistered_module(self):
        assert try_get_builder_for_module(nn.Linear(2, 2)) is None
