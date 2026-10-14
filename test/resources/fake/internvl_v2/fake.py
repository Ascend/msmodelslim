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

from typing import Optional, Tuple

import os
import json
import torch
import torch.nn as nn

from resources.fake.llava.fake import FakeLlavaCreator


class DictConfig:
    def __init__(self, config_dict: dict):
        self.config = config_dict

    def __getattr__(self, key):
        return self.config[key]


class MockInternVisionConfig(DictConfig):
    pass


class MockInternLMConfig(DictConfig):
    pass


class MockInternConfig(DictConfig):
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        super().__init__(config_dict)
        self.llm_config = MockInternLMConfig(self.config['llm_config'])
        self.vision_config = MockInternVisionConfig(self.config['vision_config'])


class MockInternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MockInternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class MockInternMLP(nn.Module):
    def __init__(self, config: MockInternVisionConfig):
        super().__init__()
        self.config = config
        self.act = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MockInternVisionEncoderLayer(nn.Module):
    def __init__(self, config: MockInternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = MockInternAttention(config)
        self.mlp = MockInternMLP(config)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        return hidden_states


class MockInternLM2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MockInternLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).'
            )

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
        )

        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

        self.rotary_emb = FakeLlavaCreator.get_decoder_layer().self_attn.rotary_emb


class MockInternLM2MLP(nn.Module):
    def __init__(self, config: MockInternLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class MockInternLM2DecoderLayer(nn.Module):
    def __init__(self, config: MockInternLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attention = MockInternLM2Attention(config)
        self.feed_forward = MockInternLM2MLP(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)


class FakeInternVLCreator:
    """
    用于生成一个随机的、非常小的InternVL模型组件，用于验证工具中某些流程的正确性
    """

    config = MockInternConfig()
    vision_config = config.vision_config
    llm_config = config.llm_config

    @classmethod
    def get_vision_block(cls):
        """
        获取一个随机的、非常小的InternVisionEncoderLayer，用于验证工具中某些流程的正确性
        """
        block = MockInternVisionEncoderLayer(config=cls.vision_config, drop_path_rate=0.0)
        return block

    @classmethod
    def get_decoder_layer(cls):
        """
        获取一个随机的、非常小的InternLM2DecoderLayer，用于验证工具中某些流程的正确性
        """
        layer = MockInternLM2DecoderLayer(config=cls.llm_config)
        return layer
