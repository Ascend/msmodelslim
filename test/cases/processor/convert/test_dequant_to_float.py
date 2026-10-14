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

msmodelslim.processor.convert.dequant_to_float 模块的单元测试
"""

from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.processor.convert.dequant_to_float import DequantToFloatProcessor


class TestDequantToFloatProcessor:
    """测试 DequantToFloatProcessor 类"""

    def test_transform_return_same_module_when_not_model_free_linear(self):
        linear = nn.Linear(2, 2)
        context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"))
        out = DequantToFloatProcessor().transform(linear, context)
        assert out is linear  # 校验非 ModelFreeLinear 原样返回

    def test_transform_return_bf16_linear_when_fp8_weight_without_scale(self):
        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={"weight": TensorRef("weight", "w", "s0", "bf16", (4, 4))},
            source_ir=SourceIR(kind=IRKind.FP8_BLOCK),
        )
        mod.register_parameter("weight", nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16)))
        mod.lazy_initialized = True
        context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"), reader=MagicMock())
        out = DequantToFloatProcessor().transform(mod, context)
        assert isinstance(out, nn.Linear)  # 校验输出 Linear
        assert out.weight.dtype == torch.bfloat16  # 校验 bf16 权重

    def test_transform_return_bf16_linear_when_fp8_weight_and_scale_given(self):
        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={
                "weight": TensorRef("weight", "w", "s0", "float8_e4m3fn", (128, 128)),
                "weight_scale_inv": TensorRef("weight_scale_inv", "w_scale", "s0", "bf16", (1, 1)),
            },
            source_ir=SourceIR(kind=IRKind.FP8_BLOCK),
        )
        weight_fp8 = torch.empty(128, 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
        mod.register_parameter("weight", nn.Parameter(weight_fp8))
        mod.register_buffer("weight_scale_inv", torch.ones(1, 1))
        mod.lazy_initialized = True
        context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"), reader=MagicMock())
        with patch(
            "msmodelslim.processor.convert.dequant_to_float.weight_dequant",
            return_value=torch.ones(128, 128, dtype=torch.bfloat16),
        ):
            out = DequantToFloatProcessor().transform(mod, context)
        assert isinstance(out, nn.Linear)
        assert out.weight.dtype == torch.bfloat16
