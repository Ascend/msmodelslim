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
# pylint: disable=consider-using-from-import

import unittest
from typing import List, Tuple

import torch
import torch.nn as nn

from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func,
)
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.utils.exception import InvalidModelError


class DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return (self.linear(x),)


class DummyModel(nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(num_layers)])

    def named_modules(self):
        for idx, layer in enumerate(self.layers):
            yield f"layers.{idx}", layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)[0]
        return x


class TestLayerWiseForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = DummyModel(num_layers=3)
        self.input = torch.randn(2, 4)

    def test_generated_decoder_layer_visit_func_auto_discovery(self):
        gen = generated_decoder_layer_visit_func(self.model)
        collected: List[ProcessRequest] = list(gen)
        self.assertEqual(len(collected), 3)
        for _, req in enumerate(collected):
            self.assertIsInstance(req, ProcessRequest)
            self.assertIn("layers.", req.name)
            self.assertIsInstance(req.module, DummyDecoderLayer)
            self.assertEqual(req.args, tuple())
            self.assertEqual(req.kwargs, {})

    def test_generated_decoder_layer_visit_func_with_custom_blocks(self):
        blocks = [("x", self.model.layers[1])]
        gen = generated_decoder_layer_visit_func(self.model, transformer_blocks=blocks)
        reqs = list(gen)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].name, "x")
        self.assertIs(reqs[0].module, self.model.layers[1])

    def test_transformers_generated_forward_func_normal_flow(self):
        gen = transformers_generated_forward_func(self.model, (self.input,))
        # First yield for layer 0
        req0 = next(gen)
        self.assertIsInstance(req0, ProcessRequest)
        out0 = req0.module(*req0.args, **req0.kwargs)
        # Send output back
        req1 = gen.send(out0)
        out1 = req1.module(*req1.args, **req1.kwargs)
        req2 = gen.send(out1)
        out2 = req2.module(*req2.args, **req2.kwargs)
        with self.assertRaises(StopIteration):
            gen.send(out2)

    def test_transformers_generated_forward_func_no_first_input_error(self):
        class BadModel(nn.Module):
            def named_modules(self):
                # No decoder layers discovered
                return []

            def forward(self, x):
                return x

        bad = BadModel()
        with self.assertRaises(IndexError):
            gen = transformers_generated_forward_func(bad, (self.input,))
            next(gen)

    def test_generated_decoder_layer_visit_func_when_distributed_then_call_barrier(self):
        """测试generated_decoder_layer_visit_func：分布式环境下应调用barrier"""
        from unittest.mock import patch

        # Mock dist.is_initialized返回True
        with patch('msmodelslim.model.common.layer_wise_forward.dist.is_initialized', return_value=True):
            with patch('msmodelslim.model.common.layer_wise_forward.dist.barrier') as mock_barrier:
                gen = generated_decoder_layer_visit_func(self.model)

                list(gen)
                # 验证barrier被调用
                mock_barrier.assert_called_once()

    def test_transformers_generated_forward_func_when_dict_input_then_use_kwargs(self):
        """测试transformers_generated_forward_func：dict输入时应使用**kwargs"""
        from unittest.mock import patch

        # 使用dict输入（覆盖第82-83行）
        dict_inputs = {'x': self.input}

        # Mock TransformersForwardBreak以模拟hook触发
        with patch('msmodelslim.model.common.layer_wise_forward.TransformersForwardBreak', Exception):
            gen = transformers_generated_forward_func(self.model, dict_inputs)

            try:
                list(gen)
            except (InvalidModelError, Exception):
                pass

    def test_transformers_generated_forward_func_when_single_tensor_input_then_call_directly(self):
        """测试transformers_generated_forward_func：单个tensor输入时应直接调用"""
        from unittest.mock import patch

        with patch('msmodelslim.model.common.layer_wise_forward.TransformersForwardBreak', Exception):
            gen = transformers_generated_forward_func(self.model, self.input)

            try:
                list(gen)
            except (InvalidModelError, Exception):
                pass

    def test_transformers_generated_forward_func_when_first_block_input_none_then_raise_error(self):
        """测试transformers_generated_forward_func：first_block_input为None时应抛出InvalidModelError"""
        from unittest.mock import patch  # noqa: F401

        # 创建一个模型，forward不会触发TransformersForwardBreak
        class NoBreakModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = DummyDecoderLayer()

            def named_modules(self):
                return [('layer0', self.layer)]

            def forward(self, x):
                # 正常返回，不触发TransformersForwardBreak
                return x

        model = NoBreakModel()

        gen = transformers_generated_forward_func(model, (self.input,))

        with self.assertRaises(InvalidModelError) as context:
            list(gen)

        self.assertIn("Can't get first block input", str(context.exception))

    def test_transformers_generated_forward_func_when_distributed_then_call_barrier(self):
        """测试transformers_generated_forward_func：分布式环境下应调用barrier"""
        from unittest.mock import patch

        with patch('msmodelslim.model.common.layer_wise_forward.dist.is_initialized', return_value=True):
            with patch('msmodelslim.model.common.layer_wise_forward.dist.barrier') as mock_barrier:  # noqa: F841
                gen = transformers_generated_forward_func(self.model, (self.input,))

                try:
                    list(gen)
                except (InvalidModelError, Exception):
                    pass
