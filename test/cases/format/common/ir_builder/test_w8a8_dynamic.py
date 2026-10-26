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

Unit tests for msmodelslim.format.common.ir_builder.w8a8_dynamic.
"""

import pytest
from torch import nn

from msmodelslim.format.common.ir_builder.base import ApplyMode
from msmodelslim.format.common.ir_builder.w8a8_dynamic import (
    W8A8DynamicIrBuilder,
    W8A8DynamicStructureSpec,
)
from msmodelslim.ir.w8a8_dynamic import (
    W8A8DynamicPerChannelFakeQuantLinear,
    W8A8DynamicPerGroupFakeQuantLinear,
)
from msmodelslim.utils.exception import SchemaValidateError


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(2, 3)


class TestW8A8DynamicIrBuilder:
    def test_transform_shell_builds_per_channel_module_when_group_size_zero(self):
        model = _Holder()
        structure = W8A8DynamicStructureSpec(weight_shape=(3, 2), weight_scale_shape=(3,), group_size=0)

        result = W8A8DynamicIrBuilder().transform_shell(model, "attn", nn.Linear(2, 3), structure=structure)

        assert result.mode is ApplyMode.REPLACE
        assert isinstance(result.module, W8A8DynamicPerChannelFakeQuantLinear)
        assert model.attn is result.module

    def test_transform_shell_builds_per_group_module_when_group_size_positive(self):
        model = _Holder()
        structure = W8A8DynamicStructureSpec(weight_shape=(3, 2), weight_scale_shape=(3, 2), group_size=2)

        result = W8A8DynamicIrBuilder().transform_shell(model, "attn", nn.Linear(2, 3), structure=structure)

        assert isinstance(result.module, W8A8DynamicPerGroupFakeQuantLinear)

    def test_transform_shell_raises_when_structure_type_mismatch(self):
        with pytest.raises(SchemaValidateError):
            W8A8DynamicIrBuilder().transform_shell(_Holder(), "attn", nn.Linear(2, 3), structure=object())
