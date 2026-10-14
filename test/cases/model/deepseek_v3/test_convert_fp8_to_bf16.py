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
# pylint: disable=consider-using-from-import,unspecified-encoding

import json
import os
import tempfile
import unittest
from typing import Dict
from unittest.mock import patch

import torch
import torch.nn as nn
from safetensors.torch import save_file

import msmodelslim.model.deepseek_v3.convert_fp8_to_bf16 as mod


class TestConvertFP8ToBF16(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_weight_dequant_when_given_weight_and_scale_then_return_dequantized_bfloat16(self):
        """测试weight_dequant方法：给定weight和scale时应返回反量化的bfloat16结果"""
        m, n = 256, 256
        weight = torch.randn(m, n, dtype=torch.float32)
        scale = torch.full((m // 128, n // 128), 0.5, dtype=torch.float32)

        expected = (
            weight.to(torch.float32) * scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:m, :n]
        ).to(torch.bfloat16)

        out = mod.weight_dequant(weight.clone(), scale)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (m, n))
        self.assertTrue(torch.allclose(out.float(), expected.float(), atol=0, rtol=0))

    def test_convert_module_fp8_to_bf16_when_called_then_apply_dequantization(self):
        """测试convert_module_fp8_to_bf16方法：调用时应对模型权重应用反量化"""
        linear = nn.Linear(256, 256, bias=False)
        linear.weight.data = torch.randn(256, 256, dtype=torch.bfloat16)
        original_dtype = linear.weight.dtype

        weight_map: Dict[str, str] = {'linear': 'chunk-00001-of-00001.safetensors'}
        scale = torch.full((2, 2), 0.25, dtype=torch.float32)

        original_get_inv_tensor = mod.get_inv_tensor
        try:

            def fake_get_inv_tensor(tensor_name, fp8_path, wm):
                self.assertEqual(tensor_name, 'linear')
                self.assertIs(wm, weight_map)
                return scale

            mod.get_inv_tensor = fake_get_inv_tensor  # type: ignore
            expected = mod.weight_dequant(linear.weight.data.clone(), scale)
            mod.convert_module_fp8_to_bf16('', nn.ModuleDict({'linear': linear}), 'IGNORED', weight_map)

            self.assertTrue(torch.allclose(linear.weight.data.float(), expected.float(), atol=0, rtol=0))
            self.assertEqual(linear.weight.dtype, original_dtype)
        finally:
            mod.get_inv_tensor = original_get_inv_tensor  # restore

    def test_auto_convert_module_fp8_to_bf16_when_weight_map_empty_then_skip_conversion(self):
        """测试auto_convert_module_fp8_to_bf16方法：weight_map为空时应跳过转换"""
        called = {'flag': False}

        def fake_get_inv_weight_map(_):
            return {}

        def fake_convert(*args, **kwargs):
            called['flag'] = True

        original_get_inv_weight_map = mod.get_inv_weight_map
        original_convert = mod.convert_module_fp8_to_bf16
        try:
            mod.get_inv_weight_map = fake_get_inv_weight_map  # type: ignore
            mod.convert_module_fp8_to_bf16 = fake_convert  # type: ignore
            mod.auto_convert_module_fp8_to_bf16("", nn.Linear(1, 1, bias=False), 'IGNORED')
            self.assertFalse(called['flag'])
        finally:
            mod.get_inv_weight_map = original_get_inv_weight_map
            mod.convert_module_fp8_to_bf16 = original_convert

    def test_auto_convert_module_fp8_to_bf16_when_weight_map_not_empty_then_call_convert(self):
        """测试auto_convert_module_fp8_to_bf16方法：weight_map不为空时应调用转换"""
        called = {'flag': False, 'flag2': False}

        def fake_get_inv_weight_map(_):
            called['flag2'] = True
            return {'root.linear': 'chunk.safetensors'}

        def fake_convert(name: str, model, model_path, weight_map):
            called['flag'] = True
            self.assertIn('linear', dict(model.named_modules()))

        original_get_inv_weight_map = mod.get_inv_weight_map
        original_convert = mod.convert_module_fp8_to_bf16
        try:
            mod.get_inv_weight_map = fake_get_inv_weight_map  # type: ignore
            mod.convert_module_fp8_to_bf16 = fake_convert  # type: ignore
            mod.auto_convert_module_fp8_to_bf16("", nn.ModuleDict({'linear': nn.Linear(1, 1, bias=False)}), 'IGNORED')
            self.assertTrue(called['flag'])
            self.assertTrue(called['flag2'])
        finally:
            mod.get_inv_weight_map = original_get_inv_weight_map
            mod.convert_module_fp8_to_bf16 = original_convert

    def test_auto_convert_module_fp8_to_bf16_when_keyerror_raised_then_handle_gracefully(self):
        """测试auto_convert_module_fp8_to_bf16方法：KeyError异常时应优雅处理"""

        def fake_get_inv_weight_map(_):
            return {'root.linear': 'chunk.safetensors'}

        def fake_convert(*args, **kwargs):
            raise KeyError('missing tensor')

        original_get_inv_weight_map = mod.get_inv_weight_map
        original_convert = mod.convert_module_fp8_to_bf16
        try:
            mod.get_inv_weight_map = fake_get_inv_weight_map  # type: ignore
            mod.convert_module_fp8_to_bf16 = fake_convert  # type: ignore
            mod.auto_convert_module_fp8_to_bf16("", nn.ModuleDict({'linear': nn.Linear(1, 1, bias=False)}), 'IGNORED')
        finally:
            mod.get_inv_weight_map = original_get_inv_weight_map
            mod.convert_module_fp8_to_bf16 = original_convert

    def test_get_inv_weight_map_when_called_then_load_and_filter_weight_map(self):
        """测试get_inv_weight_map函数：调用时应加载并过滤weight_map"""
        # 创建临时目录和index文件
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "model.safetensors.index.json")

            # 创建模拟的index.json内容
            index_content = {
                'weight_map': {
                    'model.layer1.weight': 'model-00001.safetensors',
                    'model.layer1.weight_scale_inv': 'model-00001.safetensors',
                    'model.layer2.weight': 'model-00002.safetensors',
                    'model.layer2.weight_scale_inv': 'model-00002.safetensors',
                }
            }

            with open(index_path, 'w') as f:
                json.dump(index_content, f)

            with patch('msmodelslim.model.deepseek_v3.convert_fp8_to_bf16.json_safe_load') as mock_json_load:
                mock_json_load.return_value = index_content

                # 清除lru_cache以确保函数重新执行
                mod.get_inv_weight_map.cache_clear()

                # 调用函数（覆盖第56-60行）
                result = mod.get_inv_weight_map(temp_dir)

                # 验证返回的weight_map只包含.weight_scale_inv的条目（已去掉后缀）
                self.assertEqual(len(result), 2)
                self.assertIn('model.layer1', result)
                self.assertIn('model.layer2', result)
                self.assertEqual(result['model.layer1'], 'model-00001.safetensors')
                self.assertEqual(result['model.layer2'], 'model-00002.safetensors')

    def test_get_inv_tensor_when_called_then_load_tensor_from_safetensors(self):
        """测试get_inv_tensor函数：调用时应从safetensors文件加载tensor"""
        # 创建临时目录和safetensors文件
        with tempfile.TemporaryDirectory() as temp_dir:
            safetensor_file = 'model-00001.safetensors'
            safetensor_path = os.path.join(temp_dir, safetensor_file)

            # 创建模拟的scale tensor
            scale_tensor = torch.randn(2, 2)
            tensors_dict = {'model.layer1.weight_scale_inv': scale_tensor}

            save_file(tensors_dict, safetensor_path)

            weight_map = {'model.layer1': safetensor_file}

            # Mock get_valid_read_path以返回原路径
            with patch('msmodelslim.model.deepseek_v3.convert_fp8_to_bf16.get_valid_read_path') as mock_get_path:
                mock_get_path.return_value = safetensor_path

                result = mod.get_inv_tensor('model.layer1', temp_dir, weight_map)

                self.assertEqual(result.shape, scale_tensor.shape)
                self.assertTrue(torch.allclose(result, scale_tensor))
