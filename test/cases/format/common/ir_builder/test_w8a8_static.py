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

Unit tests for msmodelslim.format.common.ir_builder.w8a8_static.
"""

import pytest
from torch import nn

from msmodelslim.format.common.ir_builder.base import ApplyMode
from msmodelslim.format.common.ir_builder.w8a8_static import (
    W8A8StaticIrBuilder,
    W8A8StaticStructureSpec,
)
from msmodelslim.ir.w8a8_static import W8A8StaticFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(2, 3)


class TestW8A8StaticIrBuilder:
    def test_transform_shell_replaces_with_static_fake_quant_linear(self):
        model = _Holder()
        structure = W8A8StaticStructureSpec(
            weight_shape=(3, 2),
            input_scale_shape=(1,),
            input_offset_shape=(1,),
            weight_scale_shape=(3,),
        )

        result = W8A8StaticIrBuilder().transform_shell(model, "attn", nn.Linear(2, 3), structure=structure)

        assert result.mode is ApplyMode.REPLACE
        assert isinstance(result.module, W8A8StaticFakeQuantLinear)
        assert model.attn is result.module

    def test_transform_shell_raises_when_structure_type_mismatch(self):
        with pytest.raises(SchemaValidateError):
            W8A8StaticIrBuilder().transform_shell(_Holder(), "attn", nn.Linear(2, 3), structure=object())

    def test_bind_from_param_set_raises_when_ir_module_wrong_type(self):
        with pytest.raises(SchemaValidateError):
            W8A8StaticIrBuilder().bind_from_param_set(nn.Identity(), object())
