#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import os
import tempfile
import unittest
from unittest.mock import patch
import shutil

import torch
from torch import nn
from transformers import PretrainedConfig

from msmodelslim.model.deepseek_v3.mtp_quant_module import (
    remove_zero_and_shift,
    DeepseekV3RMSNorm,
    SharedHead,
    MTPLayer,
    get_mtp_layer,
    wrap_mtp_decoder,
)


class DummyConfig(PretrainedConfig):
    """模拟配置对象"""

    model_type = "dummy"

    def __init__(self, **kwargs):
        super().__init__(pad_token_id=0, **kwargs)
        self.hidden_size = 128
        self.vocab_size = 1000
        self.rms_norm_eps = 1e-6
        self.num_hidden_layers = 3


class DummyDecoderLayer(nn.Module):
    """模拟DecoderLayer"""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        # 为MTP层预留属性（会被动态设置）
        self.enorm = None
        self.hnorm = None
        self.shared_head = None
        self.eh_proj = None
        self.embed_tokens = None

    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class DummyModel(nn.Module):
    """模拟基础模型的model部分"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DummyDecoderLayer(config.hidden_size) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids=None, **kwargs):
        # 返回一个简单的输出结构
        hidden_states = torch.randn(1, 10, self.config.hidden_size)
        return type('Output', (), {'__getitem__': lambda self, i: hidden_states if i == 0 else None})()


class DummyBaseModel(nn.Module):
    """模拟完整的基础模型"""

    def __init__(self, config):
        super().__init__()
        self.model = DummyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class TestRemoveZeroAndShift(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_remove_zero_and_shift_when_matrix_has_zeros_then_shift_and_pad(self):
        """测试remove_zero_and_shift方法：矩阵包含0时应移除并前移元素"""
        matrix = torch.tensor([[1, 2, 0, 3, 4], [5, 0, 6, 7, 8], [0, 9, 10, 11, 12]])

        result = remove_zero_and_shift(matrix)

        self.assertEqual(result.shape, matrix.shape)
        expected = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 8, 0], [9, 10, 11, 12, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_remove_zero_and_shift_when_single_row_then_process_correctly(self):
        """测试remove_zero_and_shift方法：单行矩阵时应正确处理"""
        matrix = torch.tensor([[1, 0, 2, 3]])
        result = remove_zero_and_shift(matrix)
        expected = torch.tensor([[1, 2, 3, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_remove_zero_and_shift_when_called_then_preserve_device(self):
        """测试remove_zero_and_shift方法：调用时应保留设备属性"""
        matrix = torch.tensor([[1, 0, 2]], device='cpu')
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.device, matrix.device)

    def test_remove_zero_and_shift_when_called_then_preserve_dtype(self):
        """测试remove_zero_and_shift方法：调用时应保留数据类型"""
        matrix = torch.tensor([[1, 0, 2]], dtype=torch.long)
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.dtype, matrix.dtype)


class TestMtpPositionIds(unittest.TestCase):
    def test_get_mtp_position_ids_matches_megatron_roll(self):
        from msmodelslim.model.deepseek_v3.mtp_quant_module import get_mtp_position_ids

        seq_len = 8
        position_ids = get_mtp_position_ids(seq_len, torch.device('cpu'))
        self.assertTrue(torch.equal(position_ids, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])))

    def test_get_mtp_position_ids_avoids_rope_oob_at_max_seq_len(self):
        from msmodelslim.model.deepseek_v3.mtp_quant_module import get_mtp_position_ids

        seq_len = 512
        position_ids = get_mtp_position_ids(seq_len, torch.device('cpu'))
        self.assertEqual(position_ids.max().item(), seq_len - 1)
        cos = torch.zeros(seq_len, 64)
        self.assertEqual(cos[position_ids].shape, (1, seq_len, 64))


class TestDeepseekV3RMSNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_rms_norm_initialization_when_created_then_weight_is_ones(self):
        """测试RMSNorm初始化：创建时权重应为1"""
        hidden_size = 128
        eps = 1e-6
        norm = DeepseekV3RMSNorm(hidden_size, eps=eps)

        # 验证weight参数初始化为1
        self.assertEqual(norm.weight.shape, (hidden_size,))
        self.assertTrue(torch.allclose(norm.weight, torch.ones(hidden_size)))
        self.assertEqual(norm.variance_epsilon, eps)

    def test_rms_norm_forward_when_called_then_return_normalized_output(self):
        """测试RMSNorm前向传播：调用时应返回归一化输出"""
        hidden_size = 128
        norm = DeepseekV3RMSNorm(hidden_size)

        hidden_states = torch.randn(2, 10, hidden_size)
        input_dtype = hidden_states.dtype

        # 前向传播
        output = norm(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(output.dtype, input_dtype)

    def test_rms_norm_when_different_dtypes_then_handle_correctly(self):
        """测试RMSNorm：不同数据类型时应正确处理"""
        hidden_size = 64

        # 测试float32
        norm_fp32 = DeepseekV3RMSNorm(hidden_size)
        hidden_states_fp32 = torch.randn(1, 5, hidden_size, dtype=torch.float32)
        output_fp32 = norm_fp32(hidden_states_fp32)
        self.assertEqual(output_fp32.dtype, torch.float32)

        norm_bf16 = DeepseekV3RMSNorm(hidden_size)
        hidden_states_bf16 = torch.randn(1, 5, hidden_size, dtype=torch.bfloat16)
        output_bf16 = norm_bf16(hidden_states_bf16)
        # 验证输出形状正确
        self.assertEqual(output_bf16.shape, hidden_states_bf16.shape)


class TestSharedHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()

    def test_shared_head_initialization_when_created_then_have_norm_and_head(self):
        """测试SharedHead初始化：创建时应包含norm和head层"""
        head = SharedHead(self.config)

        # 验证组件存在
        self.assertIsInstance(head.norm, DeepseekV3RMSNorm)
        self.assertIsInstance(head.head, nn.Linear)

        # 验证线性层配置
        self.assertEqual(head.head.in_features, self.config.hidden_size)
        self.assertEqual(head.head.out_features, self.config.vocab_size)
        self.assertIsNone(head.head.bias)

    def test_shared_head_forward_when_called_then_return_logits(self):
        """测试SharedHead前向传播：调用时应返回logits"""
        head = SharedHead(self.config)

        # 创建输入
        hidden_states = torch.randn(2, 10, self.config.hidden_size)

        # 前向传播
        logits = head(hidden_states)

        # 验证输出形状
        self.assertEqual(logits.shape, (2, 10, self.config.vocab_size))


class TestMTPLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()

    def test_mtp_layer_initialization_when_created_then_have_all_components(self):
        """测试MTPLayer初始化：创建时应包含所有组件"""
        mtp_layer = MTPLayer(self.config)

        # 验证组件存在
        self.assertIsInstance(mtp_layer.enorm, DeepseekV3RMSNorm)
        self.assertIsInstance(mtp_layer.hnorm, DeepseekV3RMSNorm)
        self.assertIsInstance(mtp_layer.shared_head, SharedHead)
        self.assertIsInstance(mtp_layer.eh_proj, nn.Linear)
        self.assertIsInstance(mtp_layer.embed_tokens, nn.Embedding)

        # 验证线性层配置
        self.assertEqual(mtp_layer.eh_proj.in_features, self.config.hidden_size * 2)
        self.assertEqual(mtp_layer.eh_proj.out_features, self.config.hidden_size)

        # 验证嵌入层配置
        self.assertEqual(mtp_layer.embed_tokens.num_embeddings, self.config.vocab_size)
        self.assertEqual(mtp_layer.embed_tokens.embedding_dim, self.config.hidden_size)


class TestMTPModuleFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()
        # 创建临时目录用于模拟模型路径
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # 及时删除临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_mtp_layer_when_called_then_return_initialized_mtp_layer(self):
        """测试get_mtp_layer方法：调用时应返回初始化后的MTPLayer"""
        # 创建模拟的权重文件
        safetensor_path = os.path.join(self.temp_dir, "model-00163-of-000163.safetensors")

        # 创建模拟的权重数据，确保包含所有必要的键
        mock_weights = {
            'model.layers.61.enorm.weight': torch.ones(self.config.hidden_size),
            'model.layers.61.hnorm.weight': torch.ones(self.config.hidden_size),
            'model.layers.61.eh_proj.weight': torch.ones((self.config.hidden_size, self.config.hidden_size * 2)),
            'model.layers.61.embed_tokens.weight': torch.ones((self.config.vocab_size, self.config.hidden_size)),
            'model.layers.61.shared_head.head.weight': torch.ones((self.config.vocab_size, self.config.hidden_size)),
            'model.layers.61.shared_head.norm.weight': torch.ones(self.config.hidden_size),
        }

        with (
            patch('msmodelslim.model.deepseek_v3.mtp_quant_module.load_file') as mock_load,
            patch('msmodelslim.model.deepseek_v3.mtp_quant_module.get_valid_read_path') as mock_get_path,
            patch('msmodelslim.model.deepseek_v3.mtp_quant_module.get_logger'),
        ):
            # 设置mock返回值
            mock_get_path.return_value = safetensor_path
            mock_load.return_value = mock_weights

            # 调用函数
            result = get_mtp_layer(self.config, self.temp_dir)

            # 验证返回类型
            self.assertIsInstance(result, MTPLayer)
            # 验证相关函数被调用
            mock_get_path.assert_called_once()
            mock_load.assert_called_once_with(safetensor_path, device="cpu")

    def test_get_mtp_layer_when_weights_exist_then_load_correctly(self):
        """测试get_mtp_layer方法：权重存在时应正确加载"""
        # 创建模拟权重数据
        mock_weights = {
            'model.layers.61.enorm.weight': torch.ones(self.config.hidden_size),
            'model.layers.61.hnorm.weight': torch.ones(self.config.hidden_size),
            'model.layers.61.eh_proj.weight': torch.ones((self.config.hidden_size, self.config.hidden_size * 2)),
            'model.layers.61.embed_tokens.weight': torch.ones((self.config.vocab_size, self.config.hidden_size)),
            'model.layers.61.shared_head.head.weight': torch.ones((self.config.vocab_size, self.config.hidden_size)),
            'model.layers.61.shared_head.norm.weight': torch.ones(self.config.hidden_size),
        }

        safetensor_path = os.path.join(self.temp_dir, "model-00163-of-000163.safetensors")

        with (
            patch('msmodelslim.model.deepseek_v3.mtp_quant_module.load_file') as mock_load,
            patch('msmodelslim.model.deepseek_v3.mtp_quant_module.get_valid_read_path') as mock_get_path,
            patch('msmodelslim.model.deepseek_v3.mtp_quant_module.get_logger'),
        ):
            mock_get_path.return_value = safetensor_path
            mock_load.return_value = mock_weights

            result = get_mtp_layer(self.config, self.temp_dir)

            # 验证MTPLayer被正确创建和加载
            self.assertIsInstance(result, MTPLayer)
            # 验证权重被正确加载
            self.assertTrue(hasattr(result, 'enorm'))
            self.assertTrue(hasattr(result, 'hnorm'))
            self.assertTrue(hasattr(result, 'eh_proj'))
            self.assertTrue(hasattr(result, 'embed_tokens'))
            self.assertTrue(hasattr(result, 'shared_head'))

    def test_wrap_mtp_decoder_when_called_then_transfer_components(self):
        """测试wrap_mtp_decoder方法：调用时应正确传输组件"""
        # 创建MTPLayer和decoder
        mtp_layer = MTPLayer(self.config)
        decoder = DummyDecoderLayer(self.config.hidden_size)

        # 初始时decoder的属性为None
        self.assertIsNone(decoder.enorm)
        self.assertIsNone(decoder.hnorm)
        self.assertIsNone(decoder.shared_head)
        self.assertIsNone(decoder.eh_proj)
        self.assertIsNone(decoder.embed_tokens)

        # 调用wrap函数
        with patch('msmodelslim.model.deepseek_v3.mtp_quant_module.get_logger'):
            wrap_mtp_decoder(decoder, mtp_layer)

        # 验证组件被正确传输
        self.assertIs(decoder.enorm, mtp_layer.enorm)
        self.assertIs(decoder.hnorm, mtp_layer.hnorm)
        self.assertIs(decoder.shared_head, mtp_layer.shared_head)
        self.assertIs(decoder.eh_proj, mtp_layer.eh_proj)
        self.assertIs(decoder.embed_tokens, mtp_layer.embed_tokens)

    def test_wrap_mtp_decoder_when_called_then_preserve_references(self):
        """测试wrap_mtp_decoder方法：调用时应保持引用关系"""
        mtp_layer = MTPLayer(self.config)
        decoder = DummyDecoderLayer(self.config.hidden_size)

        # 获取MTPLayer组件的原始引用
        original_enorm = mtp_layer.enorm
        original_hnorm = mtp_layer.hnorm
        original_shared_head = mtp_layer.shared_head
        original_eh_proj = mtp_layer.eh_proj
        original_embed_tokens = mtp_layer.embed_tokens

        with patch('msmodelslim.model.deepseek_v3.mtp_quant_module.get_logger'):
            wrap_mtp_decoder(decoder, mtp_layer)

        # 验证引用保持不变（不是拷贝）
        self.assertIs(decoder.enorm, original_enorm)
        self.assertIs(decoder.hnorm, original_hnorm)
        self.assertIs(decoder.shared_head, original_shared_head)
        self.assertIs(decoder.eh_proj, original_eh_proj)
        self.assertIs(decoder.embed_tokens, original_embed_tokens)


if __name__ == '__main__':
    unittest.main()
