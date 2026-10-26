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

Unit tests for msmodelslim.format.common.ir_builder.w8a8_mx_dynamic.
"""

import pytest
from torch import nn

from msmodelslim.format.common.ir_builder.base import ApplyMode
from msmodelslim.format.common.ir_builder.w8a8_mx_dynamic import (
    DEFAULT_MX_AXES,
    W8A8MXDynamicIrBuilder,
    W8A8MXDynamicStructureSpec,
)
from msmodelslim.ir.w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(2, 3)


class TestW8A8MXDynamicStructureSpec:
    def test_default_axes_follow_module_constant(self):
        spec = W8A8MXDynamicStructureSpec(weight_shape=(3, 2), weight_scale_shape=(3, 1))

        assert spec.w_axes == DEFAULT_MX_AXES == -1


class TestW8A8MXDynamicIrBuilder:
    def test_transform_shell_replaces_with_mx_fake_quant_linear(self):
        model = _Holder()
        structure = W8A8MXDynamicStructureSpec(weight_shape=(3, 2), weight_scale_shape=(3, 1))

        result = W8A8MXDynamicIrBuilder().transform_shell(model, "attn", nn.Linear(2, 3), structure=structure)

        assert result.mode is ApplyMode.REPLACE
        assert isinstance(result.module, W8A8MXDynamicPerBlockFakeQuantLinear)
        assert model.attn is result.module

    def test_transform_shell_raises_when_structure_type_mismatch(self):
        with pytest.raises(SchemaValidateError):
            W8A8MXDynamicIrBuilder().transform_shell(_Holder(), "attn", nn.Linear(2, 3), structure=object())
