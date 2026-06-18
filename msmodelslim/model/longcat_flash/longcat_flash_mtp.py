# coding=utf-8
# Copyright 2025 Meituan and the HuggingFace Inc. team. All rights reserved
# Copyright 2018- The Hugging Face team. All rights reserved.
#
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

import torch
from torch import nn
from typing import Optional, Tuple

from transformers import PretrainedConfig
from transformers.activations import ACT2FN


class LongcatFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LongcatFlashRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, use_mla=False):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if use_mla:
        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LongcatFlashMLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) module for LongCat Flash.

    Uses low-rank compression for Q, K, V projections with partial RoPE.

    Args:
        config: Model configuration containing attention parameters
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        num_kv_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // num_kv_value_heads
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.attention_bias = getattr(config, "attention_bias", False)

        self.is_causal = True
        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=self.attention_bias)
        self.q_a_layernorm = LongcatFlashRMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=self.attention_bias,
        )
        self.kv_a_layernorm = LongcatFlashRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=self.attention_bias,
        )

        self.mla_scale_q_lora = None
        self.mla_scale_kv_lora = None
        if getattr(config, "mla_scale_q_lora", False):
            self.mla_scale_q_lora = (config.hidden_size / self.q_lora_rank) ** 0.5
        if getattr(config, "mla_scale_kv_lora", False):
            self.mla_scale_kv_lora = (config.hidden_size / self.kv_lora_rank) ** 0.5

        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (
            batch_size,
            seq_length,
            -1,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # apply q_lora scaling
        if self.mla_scale_q_lora is not None:
            q_pass = q_pass * self.mla_scale_q_lora
            q_rot = q_rot * self.mla_scale_q_lora

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(k_pass)

        # apply kv_lora scaling
        if self.mla_scale_kv_lora is not None:
            k_pass = k_pass * self.mla_scale_kv_lora

        k_pass = self.kv_b_proj(k_pass).view(key_shape).transpose(1, 2)
        k_pass, v_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(
            q_rot,
            k_rot,
            cos,
            sin,
            use_mla=True,
        )
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat([q_pass, q_rot], dim=-1)
        key_states = torch.cat([k_pass, k_rot], dim=-1)

        attn_output, attention_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            v_states,
            attention_mask,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attention_weights


class LongcatFlashMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.intermediate_size = config.ffn_hidden_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return down_proj


class LongcatFlashMTPLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.enorm = nn.ModuleDict({"m": LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)})
        self.hnorm = nn.ModuleDict({"m": LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)})

        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.input_layernorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = LongcatFlashMLA(config)

        self.transformer_layer = nn.ModuleDict(
            {
                "mlp": LongcatFlashMLP(config),
            }
        )

    def forward(
        self,
        previous_hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        input_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward method for LongcatFlash MTP Layer

        Args:
            input_embeds: Input embeddings from token embedding or previous layer
            previous_hidden_states: Hidden states from the main model's last layer
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask

        Returns:
            Hidden states after processing
        """
        # Apply normalization
        input_embeds = self.enorm["m"](input_embeds)
        previous_hidden_states = self.hnorm["m"](previous_hidden_states)

        hidden_states = self.eh_proj(torch.cat([input_embeds, previous_hidden_states], dim=-1))

        # Apply input layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + residual

        # Post attention norm and MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.transformer_layer["mlp"](hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class LongCatMultiTokenPredictor(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.num_mtp_layers = 1
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Embedding layer for input tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleDict({str(idx): LongcatFlashMTPLayer(config) for idx in range(self.num_mtp_layers)})

        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for LongCat Multi-Token Predictor
        """
        input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds
        for idx in range(self.num_mtp_layers):
            layer = self.layers[str(idx)]
            hidden_states = layer(
                input_embeds=hidden_states,
                previous_hidden_states=previous_hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


__all__ = [
    "LongCatMultiTokenPredictor",
]
