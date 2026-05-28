# coding=utf-8
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.

# pylint: disable=duplicate-code

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class Step3p5RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None, layer_idx=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.layer_idx = layer_idx
        if config.rope_parameters is not None:
            self.rope_type = config.rope_parameters.get("rope_type", config.rope_parameters.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        partial_rotary_factors = getattr(config, "partial_rotary_factors", None)
        if partial_rotary_factors is not None:
            config.partial_rotary_factor = partial_rotary_factors[self.layer_idx]
        else:
            config.partial_rotary_factor = 1.0

        self.rope_theta = config.rope_theta
        if isinstance(config.rope_theta, list):
            self.rope_theta = config.rope_theta.copy()
            config.rope_theta = self.rope_theta[self.layer_idx]

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        config.rope_theta = self.rope_theta


class Step3p5RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


class Step3p5MLP(nn.Module):
    def __init__(self, config, intermediate_size=None, swiglu_limit=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit


class Step3p5Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_groups

        layer_types = getattr(config, "layer_types", [])
        if layer_types:
            enable_sliding_window = layer_types[self.layer_idx] == "sliding_attention"
        else:
            enable_sliding_window = self.layer_idx % 2 == 0

        if hasattr(config, "yarn_only_types") and layer_types[self.layer_idx] not in config.yarn_only_types:
            config.rope_parameters = None
        else:
            config.rope_parameters = getattr(config, "rope_scaling", None)

        self.sliding_window = config.sliding_window
        if enable_sliding_window:
            self.num_attention_heads = config.attention_other_setting["num_attention_heads"]
            self.num_key_value_heads = config.attention_other_setting["num_attention_groups"]

        if self.sliding_window is not None and enable_sliding_window:
            self.sliding_window = self.sliding_window
        else:
            self.sliding_window = None
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.rotary_emb = Step3p5RotaryEmbedding(config, layer_idx=layer_idx)

        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.q_norm = Step3p5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_attention_heads, bias=False)

        self.use_rope = True
        use_rope_layers = getattr(config, "use_rope_layers", None)
        if use_rope_layers:
            self.use_rope = use_rope_layers[self.layer_idx]


class SharedHead(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.norm = Step3p5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class Step3p5MTPModule(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.enorm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.mlp = Step3p5MLP(config, config.intermediate_size)

        self.self_attn = Step3p5Attention(config, layer_idx)

        self.post_attention_layernorm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.transformer = nn.Module()
        self.transformer.shared_head = SharedHead(config)
