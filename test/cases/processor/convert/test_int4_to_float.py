#!/usr/bin/env python
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

msmodelslim.processor.convert.int4_to_float 模块的单元测试
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.processor.convert.int4_to_float import (
    Int4PackedToFloatProcessor,
    _dequant_per_group,
    _shape_to_tuple,
    _unpack_from_int32,
)
from msmodelslim.utils.exception import SchemaValidateError


def _context():
    return ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"), reader=MagicMock())


def _model_free_int4_linear():
    mod = ModelFreeLinear(
        full_name="layers.0.mlp.experts.0.up_proj",
        tensor_bindings={
            "weight_packed": TensorRef("weight_packed", "w_packed", "s0", "int32", (1, 1)),
            "weight_scale": TensorRef("weight_scale", "w_scale", "s0", "bf16", (1, 2)),
            "weight_shape": TensorRef("weight_shape", "w_shape", "s0", "int32", (2,)),
        },
        source_ir=SourceIR(kind=IRKind.INT4_PACKED),
    )
    mod.register_buffer("weight_packed", torch.tensor([[0x76543210]], dtype=torch.int32))
    mod.register_buffer("weight_scale", torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16))
    mod.register_buffer("weight_shape", torch.tensor([1, 8], dtype=torch.int32))
    mod.register_parameter("bias", nn.Parameter(torch.tensor([1.25], dtype=torch.float32), requires_grad=False))
    mod.lazy_initialized = True
    return mod


class TestInt4PackedToFloatProcessor:
    """测试 Int4PackedToFloatProcessor 类"""

    def test_transform_return_same_module_when_not_model_free_linear(self):
        linear = nn.Linear(2, 2)
        out = Int4PackedToFloatProcessor().transform(linear, _context())
        assert out is linear  # 校验非 ModelFreeLinear 原样返回

    def test_transform_return_same_module_when_required_buffers_missing(self):
        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={"weight_packed": TensorRef("weight_packed", "w", "s0", "int32", (1, 1))},
            source_ir=SourceIR(kind=IRKind.INT4_PACKED),
        )
        mod.register_buffer("weight_packed", torch.zeros((1, 1), dtype=torch.int32))
        mod.lazy_initialized = True

        out = Int4PackedToFloatProcessor().transform(mod, _context())
        assert out is mod  # 校验缺少 scale/shape 时跳过转换

    def test_transform_return_bf16_linear_when_int4_packed_weight_given(self):
        mod = _model_free_int4_linear()
        out = Int4PackedToFloatProcessor().transform(mod, _context())

        expected = torch.tensor([[-4.0, -3.5, -3.0, -2.5, -8.0, -6.0, -4.0, -2.0]], dtype=torch.float32)
        assert isinstance(out, nn.Linear)
        assert out.in_features == 8
        assert out.out_features == 1
        assert out.weight.dtype == torch.bfloat16
        assert out.bias.dtype == torch.bfloat16
        assert torch.allclose(out.weight.detach().to(torch.float32), expected, rtol=0, atol=0)
        assert torch.allclose(out.bias.detach().to(torch.float32), torch.tensor([1.25]), rtol=0, atol=0)

    def test_transform_lazy_init_when_model_free_linear_not_initialized(self):
        mod = _model_free_int4_linear()
        mod.lazy_initialized = False
        mod.lazy_init = MagicMock()

        out = Int4PackedToFloatProcessor().transform(mod, _context())

        mod.lazy_init.assert_called_once()
        assert isinstance(out, nn.Linear)


class TestInt4ToFloatFunctions:
    """测试 int4 解包与 per-group 反量化辅助函数"""

    def test_unpack_from_int32_return_signed_int4_when_packed_dim_one(self):
        out = _unpack_from_int32(torch.tensor([[0x76543210]], dtype=torch.int32), 4, torch.tensor([1, 8]), 1)
        expected = torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1]], dtype=torch.int8)
        assert torch.equal(out, expected)

    def test_unpack_from_int32_return_trimmed_shape_when_packed_dim_zero(self):
        out = _unpack_from_int32(torch.tensor([[0x76543210]], dtype=torch.int32), 4, torch.Size([6, 1]), 0)
        expected = torch.tensor([[-8], [-7], [-6], [-5], [-4], [-3]], dtype=torch.int8)
        assert torch.equal(out, expected)

    @pytest.mark.parametrize(
        "value,num_bits",
        [
            (torch.tensor([[1]], dtype=torch.int64), 4),
            (torch.tensor([[1]], dtype=torch.int32), 16),
        ],
    )
    def test_unpack_from_int32_raise_error_when_input_invalid(self, value, num_bits):
        with pytest.raises(SchemaValidateError):
            _unpack_from_int32(value, num_bits, torch.tensor([1, 8]), 1)

    def test_dequant_per_group_return_scaled_weight_when_shape_valid(self):
        weight = torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1]], dtype=torch.int8)
        scale = torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16)
        out = _dequant_per_group(weight, scale)
        expected = torch.tensor([[-4.0, -3.5, -3.0, -2.5, -8.0, -6.0, -4.0, -2.0]], dtype=torch.float32)
        assert torch.allclose(out, expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "weight,scale,match",
        [
            (torch.ones((1, 7), dtype=torch.int8), torch.ones((1, 2)), "not divisible"),
            (torch.ones((1, 8), dtype=torch.int8), torch.ones((2, 2)), "Mismatch"),
        ],
    )
    def test_dequant_per_group_raise_error_when_shape_invalid(self, weight, scale, match):
        with pytest.raises(SchemaValidateError, match=match):
            _dequant_per_group(weight, scale)

    def test_shape_to_tuple_return_tuple_when_tensor_size_or_tuple_given(self):
        assert _shape_to_tuple(torch.tensor([2, 4], dtype=torch.int32)) == (2, 4)
        assert _shape_to_tuple(torch.Size([2, 4])) == (2, 4)
        assert _shape_to_tuple((2, 4)) == (2, 4)
