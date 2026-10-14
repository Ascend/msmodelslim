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

msmodelslim.processor.convert.mxfp8_quant 模块的单元测试
"""

from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.processor.convert.mxfp8_quant import MxFp8QuantProcessor


class TestMxFp8QuantProcessor:
    """测试 MxFp8QuantProcessor 类"""

    def test_transform_deploy_mxfp8_module_when_model_free_linear_given(self):
        from msmodelslim.ir import W8A8MXDynamicPerBlockFakeQuantLinear

        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={"weight": TensorRef("weight", "w", "s0", "bf16", (16, 8))},
            source_ir=SourceIR(kind=IRKind.FLOAT),
        )
        mod.register_parameter("weight", nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16)))
        mod.lazy_initialized = True
        context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"))
        out = MxFp8QuantProcessor().transform(mod, context)
        assert isinstance(out, W8A8MXDynamicPerBlockFakeQuantLinear)  # 校验 MXFP8 部署模块
        assert out is not mod

    def test_transform_skip_when_model_free_linear_has_1d_norm_weight(self):
        mod = ModelFreeLinear(
            full_name="model.language_model.layers.0.linear_attn.norm",
            tensor_bindings={"weight": TensorRef("weight", "w", "s0", "bf16", (4096,))},
            source_ir=SourceIR(kind=IRKind.FLOAT),
        )
        mod.register_parameter("weight", nn.Parameter(torch.ones(4096, dtype=torch.bfloat16)))
        mod.lazy_initialized = True
        context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"))
        out = MxFp8QuantProcessor().transform(mod, context)
        assert out is mod  # 校验 1D norm 权重跳过量化

    def test_transform_delegate_to_quantizer_deploy_when_float_linear_given(self):
        linear = nn.Linear(8, 4, bias=False)
        deployed = nn.Linear(8, 4, bias=False)
        context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"))
        mock_quantizer = MagicMock()
        mock_quantizer.is_data_free.return_value = True
        mock_quantizer.deploy.return_value = deployed
        with patch("msmodelslim.processor.convert.mxfp8_quant.LinearQuantizer", return_value=mock_quantizer):
            out = MxFp8QuantProcessor().transform(linear, context)
        assert out is deployed  # 校验普通 Linear 走 quantizer.deploy
        mock_quantizer.setup.assert_called_once_with(linear)
