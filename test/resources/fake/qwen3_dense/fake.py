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
# pylint: disable=consider-using-from-import,no-name-in-module

import os
import torch
import torch.nn as nn

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, Qwen3DecoderLayer, Qwen3RMSNorm


class FakeQwen3Creator:
    """
    用于生成一个随机的、非常小的Qwen3DecoderLayer，用于验证工具中某些流程的正确性
    """

    config = Qwen3Config.from_pretrained(os.path.join(os.path.dirname(__file__), "config.json"))

    @classmethod
    def get_decoder_layer(cls):
        """
        获取一个随机的、非常小的Qwen3DecoderLayer，用于验证工具中某些流程的正确性
        """
        layer = Qwen3DecoderLayer(config=cls.config, layer_idx=0)
        return layer

    @classmethod
    def get_rms_norm(cls):
        """
        获取一个随机的、非常小的Qwen3RMSNorm，用于验证工具中某些流程的正确性
        """
        norm = Qwen3RMSNorm(cls.config.hidden_size, cls.config.rms_norm_eps)
        return norm

    @classmethod
    def get_model(cls):
        """
        获取一个完整的Qwen3模型，包含前向传播功能
        """
        return FakeQwen3Model(cls.config)


class FakeQwen3Model(nn.Module):
    """
    一个简化的Qwen3模型，用于测试
    """

    def __init__(self, config):
        super().__init__()
        self.device = "cpu"
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.rotary_emb = Qwen3RotaryEmbedding(self.config)

        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        # 最终层
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重
        self.apply(self._init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None, **kwargs):
        """
        前向传播函数，返回logits

        Args:
            input_ids: 输入token ids，形状为 (batch_size, seq_len)
            position_ids: 位置id，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)

        Returns:
            logits: 输出logits，形状为 (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_length = input_ids.shape

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # 创建位置id（如果未提供）
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, **kwargs
            )[0]  # Qwen3DecoderLayer返回(output, past_key_value)

        # Final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def get_submodule(self, name):
        """获取子模块"""
        if name == "embed_tokens":
            return self.embed_tokens
        elif name == "lm_head":
            return self.lm_head
        elif name == "norm":
            return self.norm
        return None

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
