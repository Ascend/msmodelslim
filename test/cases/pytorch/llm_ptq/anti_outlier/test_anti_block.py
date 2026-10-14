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

import sys  # noqa: F401
from typing import Optional, Tuple  # noqa: F401
import unittest
from unittest.mock import patch, MagicMock

import torch

from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block import (
    QuantQwen2VLVisionBlock,
    QuantQwen2VLDecoderLayer,
    QuantQwen25VLVisionBlock,
    LlavaQuantDecoder,
    LlavaClipVision,
    check_migration_import,
    QuantInternVisionEncoderLayer,
    QuantInternLM2DecoderLayer,
)

from resources.fake.qwen2_vl.fake import FakeQwenCreator
from resources.fake.qwen25_vl.fake import FakeQwen25Creator
from resources.fake.llava.fake import FakeLlavaCreator
from resources.fake.internvl_v2.fake import FakeInternVLCreator


# Mock migration_vit function
def mock_migration_vit(*args, **kwargs):
    return 1.0


def mock_migration(*args, **kwargs):
    return 1.0


class TestAntiBlock(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.seq_length = 10

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration', mock_migration)
    def test_quant_qwen2vl_decoder_layer(self):
        config = FakeQwenCreator.config
        fake_block = FakeQwenCreator.get_decoder_layer()
        block = QuantQwen2VLDecoderLayer(fake_block, config, "test_layer")
        hidden_states = torch.randn(self.batch_size, self.seq_length, config.hidden_size).to(self.device)
        attention_mask = torch.ones(self.batch_size, config.num_attention_heads, self.seq_length, self.seq_length).to(
            self.device
        )
        position_ids = torch.randint(0, 10000, (self.batch_size, self.seq_length)).to(self.device)
        past_key_value = None
        output_attentions = False
        use_cache = False
        cache_position = torch.arange(self.seq_length).to(self.device)

        position_embeddings = (
            torch.randn(3, 1, sum(config.rope_scaling['mrope_section']) * 2).to(self.device),
            torch.randn(3, 1, sum(config.rope_scaling['mrope_section']) * 2).to(self.device),
        )

        # 测试前向传播
        outputs = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, (self.batch_size, self.seq_length, config.hidden_size))

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration_vit', mock_migration_vit)
    def test_quant_qwen2vl_vision_block(self):
        config = FakeQwenCreator.config
        vision_config = FakeQwenCreator.vision_config
        fake_block = FakeQwenCreator.get_block()
        block = QuantQwen2VLVisionBlock(fake_block, config, "test_block")
        hidden_states = torch.randn(self.seq_length, vision_config.embed_dim).to(self.device)
        cu_seqlens = torch.tensor([0, self.seq_length]).to(self.device)
        rotary_pos_emb = torch.randn(self.seq_length, vision_config.embed_dim // 32).to(self.device)

        # 测试前向传播
        output = block(hidden_states, cu_seqlens, rotary_pos_emb)
        self.assertEqual(output.shape, (self.seq_length, vision_config.embed_dim))

    def test_migration_vit_not_available(self):
        """测试migration_vit不可用的情况"""
        with patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration_vit', None):
            config = FakeQwenCreator.config
            vision_config = FakeQwenCreator.vision_config
            fake_block = FakeQwenCreator.get_block()
            block = QuantQwen2VLVisionBlock(fake_block, config, "test_block")

            hidden_states = torch.randn(self.seq_length, vision_config.embed_dim).to(self.device)
            cu_seqlens = torch.tensor([0, self.seq_length]).to(self.device)
            rotary_pos_emb = torch.randn(self.seq_length, vision_config.embed_dim // 32).to(self.device)

            # 测试前向传播应该抛出ImportError
            with self.assertRaises(ImportError):
                block(hidden_states, cu_seqlens, rotary_pos_emb)

    def test_check_migration_import(self):
        """测试check_migration_import函数"""
        # 测试migration_vit存在的情况
        with patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration_vit', mock_migration_vit):
            self.assertTrue(check_migration_import(mock_migration_vit))

        # 测试migration_vit不存在的情况
        with patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration_vit', None):
            self.assertFalse(check_migration_import(None))

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration_vit', mock_migration_vit)
    def test_quant_qwen25vl_vision_block(self):
        config = FakeQwen25Creator.config
        vision_config = FakeQwen25Creator.vision_config
        fake_block = FakeQwen25Creator.get_block()
        block = QuantQwen25VLVisionBlock(fake_block, config, "test_block")
        hidden_states = torch.randn(self.seq_length, vision_config.hidden_size).to(self.device)
        cu_seqlens = torch.tensor([0, self.seq_length]).to(self.device)
        rotary_pos_emb = torch.randn(self.seq_length, vision_config.hidden_size // 32).to(self.device)

        # 测试前向传播
        output = block(hidden_states, cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        self.assertEqual(output.shape, (self.seq_length, vision_config.hidden_size))

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration', mock_migration)
    def test_quant_qwen25vl_decoder_layer(self):
        config = FakeQwen25Creator.config
        fake_block = FakeQwen25Creator.get_decoder_layer()
        block = QuantQwen2VLDecoderLayer(fake_block, config, "test_layer")
        hidden_states = torch.randn(self.batch_size, self.seq_length, config.hidden_size).to(self.device)
        attention_mask = torch.ones(self.batch_size, config.num_attention_heads, self.seq_length, self.seq_length).to(
            self.device
        )
        position_ids = torch.randint(0, 10000, (self.batch_size, self.seq_length)).to(self.device)
        past_key_value = None
        output_attentions = False
        use_cache = False
        cache_position = torch.arange(self.seq_length).to(self.device)

        position_embeddings = (
            torch.randn(3, 1, sum(config.rope_scaling['mrope_section']) * 2).to(self.device),
            torch.randn(3, 1, sum(config.rope_scaling['mrope_section']) * 2).to(self.device),
        )

        # 测试前向传播
        outputs = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, (self.batch_size, self.seq_length, config.hidden_size))

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration_vit', mock_migration_vit)
    def test_quant_internlm2vl_vision_block(self):
        config = FakeInternVLCreator.config
        vision_config = config.vision_config
        fake_block = FakeInternVLCreator.get_vision_block()
        block = QuantInternVisionEncoderLayer(fake_block, config, "test_block")
        hidden_states = torch.randn(self.batch_size, self.seq_length, vision_config.hidden_size).to(self.device)

        # 测试前向传播
        output = block(hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, vision_config.hidden_size))

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration', mock_migration)
    def test_quant_internlm2vl_decoder_layer(self):
        config = FakeInternVLCreator.config
        fake_block = FakeInternVLCreator.get_decoder_layer()
        block = QuantInternLM2DecoderLayer(fake_block, config, "test_layer")
        hidden_states = torch.randn(self.batch_size, self.seq_length, config.llm_config.hidden_size).to(self.device)
        attention_mask = torch.ones(
            self.batch_size, config.llm_config.num_attention_heads, self.seq_length, self.seq_length
        ).to(self.device)
        position_ids = torch.randint(0, 10000, (self.batch_size, self.seq_length)).to(self.device)
        past_key_value = None
        output_attentions = False
        use_cache = False

        fake_return_value = (hidden_states, attention_mask, past_key_value)

        with patch.object(block.attention, 'forward', MagicMock(return_value=fake_return_value)):
            # 测试前向传播
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, (self.batch_size, self.seq_length, config.llm_config.hidden_size))

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration', mock_migration)
    def test_quant_llava_vision_block(self):
        config = FakeLlavaCreator.text_config
        vision_config = FakeLlavaCreator.vision_config
        fake_block = FakeLlavaCreator.get_vision_block()
        block = LlavaClipVision(fake_block, config, "test_block")
        hidden_states = torch.randn(self.batch_size, self.seq_length, vision_config.hidden_size).to(self.device)
        attention_mask = torch.ones(self.batch_size, config.num_attention_heads, self.seq_length, self.seq_length).to(
            self.device
        )
        fake_return_value = (hidden_states, attention_mask)

        with patch.object(block.self_attn, 'forward', MagicMock(return_value=fake_return_value)):
            causal_attention_mask = torch.zeros(
                self.batch_size, config.num_attention_heads, self.seq_length, self.seq_length
            ).to(self.device)

            # 测试前向传播
            _ = block(hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.anti_block.migration', mock_migration)
    def test_quant_llava_decoder_layer(self):
        config = FakeLlavaCreator.text_config
        fake_block = FakeLlavaCreator.get_decoder_layer()
        block = LlavaQuantDecoder(fake_block, config, "test_layer")

        with patch.object(
            block.self_attn,
            'forward',
            MagicMock(return_value=(torch.randn(self.batch_size, self.seq_length, config.hidden_size), None, None)),
        ):
            hidden_states = torch.randn(self.batch_size, self.seq_length, config.hidden_size).to(self.device)
            attention_mask = torch.ones(
                self.batch_size, config.num_attention_heads, self.seq_length, self.seq_length
            ).to(self.device)
            position_ids = torch.randint(0, 10000, (self.batch_size, self.seq_length)).to(self.device)
            past_key_value = None
            output_attentions = False
            use_cache = False

            # 测试前向传播
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            self.assertIsInstance(outputs, tuple)
            self.assertEqual(outputs[0].shape, (self.batch_size, self.seq_length, config.hidden_size))


if __name__ == '__main__':
    unittest.main()
