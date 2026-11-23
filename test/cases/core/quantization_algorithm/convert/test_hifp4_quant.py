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
from msmodelslim.core.quantization_schema.w4a8_hifx_mx_dynamic import W4A8HiFxMXDynamicPerBlockFakeQuantLinear
from msmodelslim.core.quantization_algorithm.convert.hifp4_quant import HiFp4QuantProcessor, _hif4_quantize_weight


def _context():
    return ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"), reader=MagicMock())


def _model_free_float_linear(in_features: int = 64):
    mod = ModelFreeLinear(
        full_name="layers.0.mlp.up_proj",
        tensor_bindings={"weight": TensorRef("weight", "w", "s0", "bf16", (2, in_features))},
        source_ir=SourceIR(kind=IRKind.FLOAT),
    )
    mod.register_parameter(
        "weight", nn.Parameter(torch.randn(2, in_features, dtype=torch.bfloat16), requires_grad=False)
    )
    mod.register_parameter("bias", nn.Parameter(torch.randn(2, dtype=torch.bfloat16), requires_grad=False))
    mod.lazy_initialized = True
    return mod


class TestHifp4QuantFunctions:
    def test_hif4_quantize_weight_return_quantized_outputs_when_input_shape_is_valid(self):
        weight = torch.randn(2, 64, dtype=torch.bfloat16)
        q_weight, sf, sl2, sl3 = _hif4_quantize_weight(weight, group_size=64)
        assert q_weight.dtype == torch.bfloat16
        assert sf.shape == torch.Size([2, 1])
        assert sl2.shape == torch.Size([2, 1, 1, 1, 8])
        assert sl3.shape == torch.Size([2, 1, 2, 1, 8])

    def test_hif4_quantize_weight_raise_value_error_when_infeatures_not_divisible_by_group_size(self):
        with pytest.raises(ValueError, match="must be divisible"):
            _hif4_quantize_weight(torch.randn(2, 65, dtype=torch.bfloat16), group_size=64)


class TestHiFp4QuantProcessor:
    def test_transform_return_original_module_when_input_is_not_linear_or_model_free_linear(self):
        module = nn.ReLU()
        out = HiFp4QuantProcessor().transform(module, _context())
        assert out is module

    def test_transform_return_original_module_when_model_free_linear_weight_dim_is_not_2(self):
        mod = ModelFreeLinear(
            full_name="layers.0.norm",
            tensor_bindings={"weight": TensorRef("weight", "w", "s0", "bf16", (64,))},
            source_ir=SourceIR(kind=IRKind.FLOAT),
        )
        mod.register_parameter("weight", nn.Parameter(torch.ones(64, dtype=torch.bfloat16), requires_grad=False))
        mod.lazy_initialized = True
        out = HiFp4QuantProcessor().transform(mod, _context())
        assert out is mod

    def test_transform_return_hifp4_carrier_when_input_is_linear(self):
        linear = nn.Linear(64, 2, bias=True).to(torch.bfloat16)
        out = HiFp4QuantProcessor().transform(linear, _context())
        assert isinstance(out, W4A8HiFxMXDynamicPerBlockFakeQuantLinear)
        assert out.weight.shape == torch.Size([2, 64])
        assert out.weight_scale_lv2.shape[-3:] == torch.Size([1, 1, 8])

    def test_transform_call_lazy_init_and_return_hifp4_carrier_when_model_free_linear_is_uninitialized(self):
        mod = _model_free_float_linear(in_features=64)
        mod.lazy_initialized = False
        mod.lazy_init = MagicMock()
        out = HiFp4QuantProcessor().transform(mod, _context())
        mod.lazy_init.assert_called_once()
        assert isinstance(out, W4A8HiFxMXDynamicPerBlockFakeQuantLinear)
