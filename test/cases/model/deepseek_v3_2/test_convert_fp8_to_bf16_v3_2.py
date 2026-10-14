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
import os
import tempfile
from unittest.mock import patch, Mock

import torch
import torch.nn as nn
from safetensors.torch import save_file

import msmodelslim.model.deepseek_v3_2.convert_fp8_to_bf16 as mod


class TestConvertFP8ToBF16(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_weight_dequant_returns_dequantized_bfloat16(self):
        """测试weight_dequant：返回反量化的bfloat16结果"""
        m, n = 256, 256
        weight = torch.randn(m, n, dtype=torch.float32)
        scale = torch.full((m // 128, n // 128), 0.5, dtype=torch.float32)

        # 计算预期结果
        scale_expanded = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:m, :n]
        expected = (weight * scale_expanded).to(torch.bfloat16)

        # 执行测试并验证
        out = mod.weight_dequant(weight, scale)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, (m, n))
        self.assertTrue(torch.allclose(out.float(), expected.float()))

    def test_convert_module_fp8_to_bf16_applies_dequantization(self):
        """测试convert_module_fp8_to_bf16：对模型权重应用反量化"""
        # 初始化测试模块
        parent_module = nn.Module()
        parent_module.linear = nn.Linear(256, 256, bias=False)
        original_weight = torch.randn(256, 256, dtype=torch.bfloat16).clone()
        parent_module.linear.weight.data = original_weight

        # 配置权重映射与scale
        weight_map = {'parent.linear': 'chunk-00001-of-00001.safetensors'}
        scale = torch.full((2, 2), 0.25, dtype=torch.float32)
        dequant_results = []

        # 定义副作用函数，替代also()方法
        def record_and_dequant(w, s):
            result = w * s.mean()
            dequant_results.append((w, s))
            return result

        # 替换依赖函数
        with (
            patch.object(mod, 'get_inv_tensor', return_value=scale),
            patch.object(mod, 'weight_dequant', side_effect=record_and_dequant),
        ):
            mod.convert_module_fp8_to_bf16("parent", parent_module, "IGNORED", weight_map)

            # 验证结果
            self.assertEqual(len(dequant_results), 1)
            input_weight, used_scale = dequant_results[0]
            self.assertTrue(torch.allclose(input_weight, original_weight))
            self.assertTrue(torch.allclose(used_scale, scale))
            self.assertEqual(parent_module.linear.weight.dtype, torch.bfloat16)

    def test_auto_convert_module_fp8_to_bf16_skips_when_weight_map_empty(self):
        """测试auto_convert_module_fp8_to_bf16：weight_map为空时跳过转换"""
        convert_mock = Mock()

        with (
            patch.object(mod, 'get_inv_weight_map', return_value={}),
            patch.object(mod, 'convert_module_fp8_to_bf16', convert_mock),
        ):
            mod.auto_convert_module_fp8_to_bf16('name', nn.Linear(1, 1), 'IGNORED')
            convert_mock.assert_not_called()

    def test_auto_convert_module_fp8_to_bf16_handles_keyerror(self):
        """测试auto_convert_module_fp8_to_bf16：优雅处理KeyError异常"""
        with (
            patch.object(mod, 'get_inv_weight_map', return_value={'root.linear': 'chunk.safetensors'}),
            patch.object(mod, 'convert_module_fp8_to_bf16', side_effect=KeyError('missing tensor')),
        ):
            # 执行测试：应无异常抛出
            mod.auto_convert_module_fp8_to_bf16('root', nn.ModuleDict({'linear': nn.Linear(1, 1)}), 'IGNORED')

    def test_get_inv_weight_map_loads_and_filters(self):
        """测试get_inv_weight_map：加载并过滤出含scale_inv的权重映射"""
        index_content = {
            'weight_map': {
                'model.layer1.weight': 'model-00001.safetensors',
                'model.layer1.weight_scale_inv': 'model-00001.safetensors',
                'model.layer2.weight': 'model-00002.safetensors',
                'model.layer2.weight_scale_inv': 'model-00002.safetensors',
            }
        }

        with patch.object(mod, 'json_safe_load', return_value=index_content):
            mod.get_inv_weight_map.cache_clear()
            result = mod.get_inv_weight_map('dummy_path')

            self.assertEqual(
                result, {'model.layer1': 'model-00001.safetensors', 'model.layer2': 'model-00002.safetensors'}
            )

    def test_get_inv_tensor_loads_from_safetensors(self):
        """测试get_inv_tensor：从safetensors文件加载scale tensor"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建模拟safetensors文件
            safetensor_file = 'model-00001.safetensors'
            safetensor_path = os.path.join(temp_dir, safetensor_file)
            scale_tensor = torch.randn(2, 2)
            save_file({'model.layer1.weight_scale_inv': scale_tensor}, safetensor_path)

            with patch.object(mod, 'get_valid_read_path', return_value=safetensor_path):
                result = mod.get_inv_tensor('model.layer1', temp_dir, {'model.layer1': safetensor_file})

                self.assertTrue(torch.allclose(result, scale_tensor))
