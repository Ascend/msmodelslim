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
# pylint: disable=consider-using-from-import,useless-parent-delegation

import unittest
from typing import Tuple, Optional
from unittest.mock import patch

import torch
import torch.nn as nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func,
)


class DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        return (x, None)


class SimpleModel(nn.Module):
    """简单的测试模型，包含一个线性层和一个TransformerDecoder层"""

    def __init__(self):
        super().__init__()
        self.decoder = DummyDecoderLayer()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        x = self.decoder(x)
        return x


class TestTransformersGenerated(unittest.TestCase):
    """测试_TransformersForwardBreak异常类和相关的生成器函数"""

    def setUp(self):
        """测试前的准备工作"""
        self.model = SimpleModel()

    def test_transformers_forward_break(self):
        """测试_TransformersForwardBreak异常类"""
        # 创建异常实例
        exception = TransformersForwardBreak()

        # 验证异常类型
        self.assertIsInstance(exception, Exception)

    def test_transformers_generated_visit_func(self):
        """测试transformers_generated_visit_func函数"""
        # 获取生成器
        generator = generated_decoder_layer_visit_func(self.model)

        # 获取第一个请求
        request = next(generator)

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)

    def test_transformers_generated_visit_func_with_custom_blocks(self):
        """测试transformers_generated_visit_func函数，使用自定义的transformer_blocks"""
        # 创建自定义的transformer_blocks
        transformer_blocks = [
            (name, module)
            for name, module in self.model.named_modules()
            if "decoder" in module.__class__.__name__.lower()
        ]

        # 获取生成器
        generator = generated_decoder_layer_visit_func(self.model, transformer_blocks)

        # 获取第一个请求
        request = next(generator)

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.barrier')
    def test_transformers_generated_forward_func(self, mock_barrier, mock_get_world_size, mock_get_rank):
        """测试_transformers_generated_forward_func函数"""
        # 设置模拟函数
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # 创建输入数据
        input_data = torch.randn(5, 10)

        # 获取生成器
        generator = transformers_generated_forward_func(self.model, input_data)

        # 获取第一个请求
        request = next(generator)

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)
        self.assertTrue(torch.equal(request.args[0], input_data))


if __name__ == '__main__':
    unittest.main()
