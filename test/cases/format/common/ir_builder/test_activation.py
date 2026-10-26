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

Unit tests for msmodelslim.format.common.ir_builder.activation.
"""

import pytest
import torch
from torch import nn

from msmodelslim.format.common.ir_builder.activation import (
    Fa3DynamicParamSet,
    Fa3DynamicStructureSpec,
    Fa3PerBlockIrBuilder,
    Fa3PerHeadIrBuilder,
    Fa3PerHeadParamSet,
    Fa3PerHeadStructureSpec,
    Fa3PerTokenIrBuilder,
)
from msmodelslim.format.common.ir_builder.base import ApplyMode
from msmodelslim.ir.activation_dynamic import FakeQuantActivationPerBlock, FakeQuantActivationPerToken
from msmodelslim.ir.const import int8_per_head_sym, int8_per_token_sym, mxfp8_per_block_sym
from msmodelslim.ir.int8_activation_static import INT8FakeQuantActivationPerHead
from msmodelslim.ir.qal import QParam
from msmodelslim.utils.exception import SchemaValidateError


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Identity()


class TestFa3PerHeadIrBuilder:
    def test_transform_shell_inserts_per_head_module(self):
        model = _Holder()
        structure = Fa3PerHeadStructureSpec(input_scale_shape=(4,), scheme=int8_per_head_sym)

        result = Fa3PerHeadIrBuilder().transform_shell(model, "attn", nn.Identity(), structure=structure)

        assert result.mode is ApplyMode.INSERT
        assert isinstance(result.module, INT8FakeQuantActivationPerHead)
        assert model.attn is result.module
        assert result.module.x_q_scheme == int8_per_head_sym

    def test_transform_shell_raises_when_structure_type_mismatch(self):
        with pytest.raises(SchemaValidateError):
            Fa3PerHeadIrBuilder().transform_shell(
                _Holder(), "attn", nn.Identity(), structure=Fa3DynamicStructureSpec(scheme=int8_per_head_sym)
            )

    def test_bind_from_param_set_raises_when_ir_module_wrong_type(self):
        with pytest.raises(SchemaValidateError):
            Fa3PerHeadIrBuilder().bind_from_param_set(
                nn.Identity(), Fa3PerHeadParamSet(scale=torch.ones(4), scheme=int8_per_head_sym)
            )


class TestFa3PerTokenIrBuilder:
    def test_transform_shell_inserts_per_token_module(self):
        model = _Holder()

        result = Fa3PerTokenIrBuilder().transform_shell(
            model, "attn", nn.Identity(), structure=Fa3DynamicStructureSpec(scheme=int8_per_token_sym)
        )

        assert result.mode is ApplyMode.INSERT
        assert isinstance(result.module, FakeQuantActivationPerToken)
        assert model.attn is result.module
        assert result.module.x_q_scheme == int8_per_token_sym

    def test_bind_from_param_set_keeps_scheme(self):
        model = _Holder()
        builder = Fa3PerTokenIrBuilder()
        module = builder.transform_shell(
            model, "attn", nn.Identity(), structure=Fa3DynamicStructureSpec(scheme=int8_per_token_sym)
        ).module

        builder.bind_from_param_set(module, Fa3DynamicParamSet(scale=None, scheme=int8_per_token_sym))

        assert module.x_q_scheme == int8_per_token_sym

    def test_bind_from_param_set_raises_when_param_set_wrong_type(self):
        module = FakeQuantActivationPerToken(QParam(scheme=int8_per_token_sym))

        with pytest.raises(SchemaValidateError):
            Fa3PerTokenIrBuilder().bind_from_param_set(module, "not-a-param-set")


class TestFa3PerBlockIrBuilder:
    def test_transform_shell_inserts_per_block_module(self):
        model = _Holder()

        result = Fa3PerBlockIrBuilder().transform_shell(
            model, "attn", nn.Identity(), structure=Fa3DynamicStructureSpec(scheme=mxfp8_per_block_sym)
        )

        assert result.mode is ApplyMode.INSERT
        assert isinstance(result.module, FakeQuantActivationPerBlock)
        assert model.attn is result.module
        assert result.module.x_q_scheme == mxfp8_per_block_sym

    def test_bind_from_param_set_raises_when_ir_module_wrong_type(self):
        with pytest.raises(SchemaValidateError):
            Fa3PerBlockIrBuilder().bind_from_param_set(
                nn.Identity(), Fa3DynamicParamSet(scale=None, scheme=mxfp8_per_block_sym)
            )
