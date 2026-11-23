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
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.core.quantization_schema.kernels.nvfp4_modelopt import unpack_fp4_e2m1, weight_dequant
from msmodelslim.core.quantization_algorithm.convert.nvfp4_to_float import Nvfp4ModelOptToFloatProcessor


def _context():
    return ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"), reader=MagicMock())


def _model_free_nvfp4_linear():
    mod = ModelFreeLinear(
        full_name="layers.0.self_attn.q_proj",
        tensor_bindings={
            "weight": TensorRef("weight", "w", "s0", "uint8", (1, 8)),
            "weight_scale": TensorRef("weight_scale", "s1", "s0", "float8_e4m3fn", (1, 1)),
            "weight_scale_2": TensorRef("weight_scale_2", "s2", "s0", "float32", ()),
        },
        source_ir=SourceIR(kind=IRKind.NVFP4_MODELOPT),
    )
    mod.register_buffer("weight", torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x00, 0x00, 0x00, 0x00]], dtype=torch.uint8))
    mod.register_buffer("weight_scale", torch.tensor([[2.0]], dtype=torch.float32))
    mod.register_buffer("weight_scale_2", torch.tensor(0.5, dtype=torch.float32))
    mod.lazy_initialized = True
    return mod


class TestNvfp4ModeloptKernels:
    def test_unpack_fp4_e2m1_return_float_values_when_input_is_valid_uint8(self):
        packed = torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8)
        out = unpack_fp4_e2m1(packed)
        expected = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]], dtype=torch.float32)
        assert torch.allclose(out, expected, rtol=0, atol=0)

    def test_unpack_fp4_e2m1_raise_type_error_when_input_is_not_uint8(self):
        with pytest.raises(TypeError):
            unpack_fp4_e2m1(torch.ones((1, 2), dtype=torch.int32))

    def test_weight_dequant_return_bf16_weights_when_input_shapes_match(self):
        packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x00, 0x00, 0x00, 0x00]], dtype=torch.uint8)
        scale = torch.tensor([[2.0]], dtype=torch.float32)
        scale2 = torch.tensor(0.5, dtype=torch.float32)
        out = weight_dequant(packed, scale, scale2, group_size=16)
        assert out.dtype == torch.bfloat16
        expected = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] + [0.0] * 8], dtype=torch.float32)
        assert torch.allclose(out.to(torch.float32), expected, rtol=0, atol=0)

    def test_weight_dequant_raise_value_error_when_group_dimension_mismatches(self):
        packed = torch.zeros((1, 8), dtype=torch.uint8)
        scale = torch.ones((1, 2), dtype=torch.float32)
        with pytest.raises(ValueError, match="does not match"):
            weight_dequant(packed, scale, torch.tensor(1.0), group_size=16)


class TestNvfp4ModelOptToFloatProcessor:
    def test_transform_return_original_module_when_input_is_not_model_free_linear(self):
        linear = nn.Linear(4, 4)
        out = Nvfp4ModelOptToFloatProcessor().transform(linear, _context())
        assert out is linear

    def test_transform_return_original_module_when_required_tensors_are_missing(self):
        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={"weight": TensorRef("weight", "w", "s0", "uint8", (1, 8))},
            source_ir=SourceIR(kind=IRKind.NVFP4_MODELOPT),
        )
        mod.register_buffer("weight", torch.zeros((1, 8), dtype=torch.uint8))
        mod.lazy_initialized = True
        out = Nvfp4ModelOptToFloatProcessor().transform(mod, _context())
        assert out is mod

    def test_transform_return_bf16_linear_when_nvfp4_tensors_are_complete(self):
        mod = _model_free_nvfp4_linear()
        out = Nvfp4ModelOptToFloatProcessor().transform(mod, _context())
        assert isinstance(out, nn.Linear)
        assert out.weight.dtype == torch.bfloat16
        assert out.weight.shape == torch.Size([1, 16])

    def test_transform_call_lazy_init_when_module_is_uninitialized(self):
        mod = _model_free_nvfp4_linear()
        mod.lazy_initialized = False
        mod.lazy_init = MagicMock()
        out = Nvfp4ModelOptToFloatProcessor().transform(mod, _context())
        mod.lazy_init.assert_called_once()
        assert isinstance(out, nn.Linear)
