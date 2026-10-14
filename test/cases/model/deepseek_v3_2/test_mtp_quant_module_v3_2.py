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

import os
import unittest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn

from msmodelslim.model.deepseek_v3_2.mtp_quant_module import (
    remove_zero_and_shift,
    SharedHead,
    DeepseekV3RMSNorm,
    MTPLayer,
    get_mtp_layer,
    wrap_mtp_decoder,
)


class TestRemoveZeroAndShift(unittest.TestCase):
    def test_normal_case_with_single_zero(self):
        """测试每行含一个0的正常场景"""
        matrix = torch.tensor([[1, 0, 3, 4], [5, 6, 0, 8], [9, 10, 11, 0]])
        expected = torch.tensor([[1, 3, 4, 0], [5, 6, 8, 0], [9, 10, 11, 0]])
        torch.testing.assert_close(remove_zero_and_shift(matrix), expected)

    def test_zero_at_first_position(self):
        """测试0在每行首位的场景"""
        matrix = torch.tensor([[0, 2, 3, 4], [0, 6, 7, 8]])
        expected = torch.tensor([[2, 3, 4, 0], [6, 7, 8, 0]])
        torch.testing.assert_close(remove_zero_and_shift(matrix), expected)

    def test_multiple_zeros_in_row(self):
        """测试每行含多个0（仅移除第一个）"""
        matrix = torch.tensor([[1, 0, 3, 0], [0, 5, 0, 7]])
        expected = torch.tensor([[1, 3, 0, 0], [5, 0, 7, 0]])
        torch.testing.assert_close(remove_zero_and_shift(matrix), expected)


class TestDeepseekV3RMSNorm(unittest.TestCase):
    def test_initialization(self):
        """测试初始化参数正确性"""
        hidden_size, eps = 16, 1e-5
        norm = DeepseekV3RMSNorm(hidden_size, eps)

        self.assertIsInstance(norm.weight, nn.Parameter)
        self.assertEqual(norm.weight.shape, (hidden_size,))
        self.assertTrue(torch.allclose(norm.weight.data, torch.ones(hidden_size)))
        self.assertEqual(norm.variance_epsilon, eps)

    def test_forward_output_shape(self):
        """测试前向输出形状与输入一致"""
        norm = DeepseekV3RMSNorm(32)
        test_cases = [
            (torch.randn(10, 32), (10, 32)),
            (torch.randn(2, 10, 32), (2, 10, 32)),
            (torch.randn(5, 8, 10, 32), (5, 8, 10, 32)),
        ]

        for input_tensor, expected_shape in test_cases:
            self.assertEqual(norm(input_tensor).shape, expected_shape)

    def test_normalization_logic(self):
        """测试归一化计算逻辑"""
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
        norm = DeepseekV3RMSNorm(4, eps=0.0)
        output = norm(input_tensor)

        # 手动计算预期结果
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        expected = input_tensor * torch.rsqrt(variance)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_weight_application(self):
        """测试权重对归一化结果的作用"""
        norm = DeepseekV3RMSNorm(4)
        norm.weight.data = torch.tensor([0.5, 1.5, 2.0, 0.8])

        input_tensor = torch.ones(2, 4)
        output = norm(input_tensor)

        expected = norm.weight.data.repeat(2, 1)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_large_values_stabilization(self):
        """测试大数值输入的稳定性"""
        input_tensor = torch.tensor([[1000.0, 2000.0, 3000.0, 4000.0], [5000.0, 6000.0, 7000.0, 8000.0]])
        output = DeepseekV3RMSNorm(4)(input_tensor)
        self.assertTrue(torch.all(torch.abs(output) < 10.0))


class TestSharedHead(unittest.TestCase):
    def setUp(self):
        self.config = Mock(hidden_size=32, vocab_size=1000, rms_norm_eps=1e-6)
        self.shared_head = SharedHead(self.config)

    def test_forward_pass(self):
        """测试前向传播输出形状"""
        input_tensor = torch.randn(2, 5, self.config.hidden_size)
        output = self.shared_head(input_tensor)
        self.assertEqual(output.shape, (2, 5, self.config.vocab_size))


class TestMTPLayer(unittest.TestCase):
    def setUp(self):
        self.config = Mock(hidden_size=64, vocab_size=2000, rms_norm_eps=1e-6)

    def test_layer_initialization(self):
        """测试组件初始化与维度正确性"""
        mtp_layer = MTPLayer(self.config)

        # 验证组件类型
        self.assertIsInstance(mtp_layer.enorm, DeepseekV3RMSNorm)
        self.assertIsInstance(mtp_layer.hnorm, DeepseekV3RMSNorm)
        self.assertIsInstance(mtp_layer.shared_head, SharedHead)
        self.assertIsInstance(mtp_layer.eh_proj, nn.Linear)
        self.assertIsInstance(mtp_layer.embed_tokens, nn.Embedding)

        # 验证维度
        self.assertEqual(mtp_layer.eh_proj.in_features, self.config.hidden_size * 2)
        self.assertEqual(mtp_layer.eh_proj.out_features, self.config.hidden_size)
        self.assertEqual(mtp_layer.embed_tokens.num_embeddings, self.config.vocab_size)
        self.assertEqual(mtp_layer.embed_tokens.embedding_dim, self.config.hidden_size)


class TestGetMtpLayer(unittest.TestCase):
    def setUp(self):
        self.config = Mock(hidden_size=64, vocab_size=2000, rms_norm_eps=1e-6)
        self.model_path = "/path/to/model"
        self.expected_safetensor_path = os.path.join(self.model_path, "model-00163-of-000163.safetensors")

    @patch('msmodelslim.model.deepseek_v3_2.mtp_quant_module.load_file')
    @patch('msmodelslim.model.deepseek_v3_2.mtp_quant_module.os.path.join')
    @patch('msmodelslim.model.deepseek_v3_2.mtp_quant_module.get_logger')
    def test_successful_loading(self, mock_get_logger, mock_path_join, mock_load_file):
        """测试成功加载完整流程"""
        # Mock配置
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_path_join.return_value = self.expected_safetensor_path

        # 模拟权重
        hidden_size, vocab_size = self.config.hidden_size, self.config.vocab_size
        mock_load_file.return_value = {
            'model.layers.61.enorm.weight': torch.ones(hidden_size),
            'model.layers.61.hnorm.weight': torch.ones(hidden_size),
            'model.layers.61.shared_head.norm.weight': torch.ones(hidden_size),
            'model.layers.61.shared_head.head.weight': torch.randn(vocab_size, hidden_size),
            'model.layers.61.eh_proj.weight': torch.randn(hidden_size, hidden_size * 2),
            'model.layers.61.embed_tokens.weight': torch.randn(vocab_size, hidden_size),
            'model.layers.60.other.weight': torch.randn(10),
            'unrelated.key': torch.randn(5),
        }

        # 执行测试
        mtp_layer = get_mtp_layer(self.config, self.model_path)

        # 验证结果
        self.assertIsInstance(mtp_layer, MTPLayer)
        mock_path_join.assert_called_once_with(self.model_path, "model-00163-of-000163.safetensors")
        mock_load_file.assert_called_once()

        # 验证权重过滤
        expected_keys = {
            'enorm.weight',
            'hnorm.weight',
            'shared_head.norm.weight',
            'shared_head.head.weight',
            'eh_proj.weight',
            'embed_tokens.weight',
        }
        self.assertTrue(expected_keys.issubset(set(mtp_layer.state_dict().keys())))

        # 验证日志
        mock_logger.debug.assert_any_call('Start to load mtp')
        mock_logger.debug.assert_any_call('Success to load mtp')
        self.assertEqual(mock_logger.debug.call_count, 2)

    @patch('msmodelslim.model.deepseek_v3_2.mtp_quant_module.load_file')
    def test_weight_shape_mismatch(self, mock_load_file):
        """测试权重形状不匹配"""
        mock_load_file.return_value = {'model.layers.61.enorm.weight': torch.ones(3)}
        with self.assertRaises(RuntimeError) as cm:
            get_mtp_layer(self.config, self.model_path)
        self.assertIn("size mismatch", str(cm.exception))
        self.assertIn("enorm.weight", str(cm.exception))

    @patch('msmodelslim.model.deepseek_v3_2.mtp_quant_module.load_file')
    def test_file_not_found(self, mock_load_file):
        """测试文件不存在"""
        mock_load_file.side_effect = FileNotFoundError("File not found")
        with self.assertRaises(FileNotFoundError) as cm:
            get_mtp_layer(self.config, self.model_path)
        self.assertIn("File not found", str(cm.exception))

    @patch('msmodelslim.model.deepseek_v3_2.mtp_quant_module.load_file')
    def test_load_file_exception(self, mock_load_file):
        """测试其他加载异常"""
        mock_load_file.side_effect = PermissionError("Permission denied")
        with self.assertRaises(PermissionError):
            get_mtp_layer(self.config, self.model_path)

    @patch('msmodelslim.model.deepseek_v3_2.mtp_quant_module.load_file')
    @patch('msmodelslim.model.deepseek_v3_2.mtp_quant_module.os.path.join')
    def test_empty_weight_file(self, mock_path_join, mock_load_file):
        """测试空权重文件"""
        mock_load_file.return_value = {}
        with self.assertRaises(RuntimeError) as cm:
            get_mtp_layer(self.config, self.model_path)
        self.assertIn("Missing key(s) in state_dict", str(cm.exception))


# ------------------------------ 测试wrap_mtp_decoder函数 ------------------------------
class TestWrapMtpDecoder(unittest.TestCase):
    def test_property_replacement(self):
        """测试属性替换正确性"""
        mtp_decoder = Mock()
        mtp_layer = Mock()

        # 保存原始属性
        orig_enorm = mtp_decoder.enorm
        orig_hnorm = mtp_decoder.hnorm

        # 执行包装
        wrap_mtp_decoder(mtp_decoder, mtp_layer)

        # 验证替换
        self.assertEqual(mtp_decoder.enorm, mtp_layer.enorm)
        self.assertEqual(mtp_decoder.hnorm, mtp_layer.hnorm)
        self.assertEqual(mtp_decoder.shared_head, mtp_layer.shared_head)
        self.assertEqual(mtp_decoder.eh_proj, mtp_layer.eh_proj)
        self.assertEqual(mtp_decoder.embed_tokens, mtp_layer.embed_tokens)

        # 验证属性已变更
        self.assertNotEqual(mtp_decoder.enorm, orig_enorm)
        self.assertNotEqual(mtp_decoder.hnorm, orig_hnorm)


if __name__ == '__main__':
    unittest.main()
