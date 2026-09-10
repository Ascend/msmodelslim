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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _normalize_indexer_type(indexer_type) -> str:
    if isinstance(indexer_type, str):
        if indexer_type in ("F", "f"):
            return "full"
        if indexer_type in ("S", "s"):
            return "shared"
        return indexer_type.lower()
    return "full"


def has_indexer(config, layer_idx: int) -> bool:
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is None:
        return False
    if layer_idx < 0 or layer_idx >= len(indexer_types):
        return False
    return _normalize_indexer_type(indexer_types[layer_idx]) != "shared"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight.to(torch.float32) * x
        x = x * torch.sigmoid(gate.to(torch.float32))
        return x.to(input_dtype)


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        flat = hidden_streams.flatten(start_dim=2).float()
        norm_flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.hc_eps)
        pre_w, post_w, comb_w = F.linear(norm_flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class ForgetGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.safe_gate_lower_bound = config.linear_lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(A_log)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
        return -decay_rate * g_softplus


class LinearAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.head_dim * self.num_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        # Q/K/V 各自独立 conv1d（与真实权重格式一致）
        self.q_conv1d = nn.Conv1d(
            in_channels=self.qkv_dim,
            out_channels=self.qkv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.qkv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.k_conv1d = nn.Conv1d(
            in_channels=self.qkv_dim,
            out_channels=self.qkv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.qkv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.v_conv1d = nn.Conv1d(
            in_channels=self.qkv_dim,
            out_channels=self.qkv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.qkv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.forget_gate = ForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = RMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.q_proj(hidden_states).transpose(1, 2)
        k = self.k_proj(hidden_states).transpose(1, 2)
        v = self.v_proj(hidden_states).transpose(1, 2)

        q = self.q_conv1d(q)[:, :, :seq_len]
        k = self.k_conv1d(k)[:, :, :seq_len]
        v = self.v_conv1d(v)[:, :, :seq_len]

        query = q.transpose(1, 2).view(hidden_shape)
        key = k.transpose(1, 2).view(hidden_shape)
        value = v.transpose(1, 2).view(hidden_shape)

        # g/beta 未用于下方简化注意力，保留调用以维持这些模块的前向执行（激活统计）
        _g = self.forget_gate(hidden_states)
        _beta = torch.sigmoid(self.b_proj(hidden_states))

        # 简化版 KDA attention，仅用于校准前向（见上：g/beta 不参与计算）
        scale = 1.0 / (self.head_dim**0.5)
        query = query * scale
        attn_weights = torch.matmul(query.float(), key.transpose(-2, -1).float())
        attn_weights = F.softmax(attn_weights, dim=-1).to(query.dtype)
        core_attn_out = torch.matmul(attn_weights, value)

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output


class Indexer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, self.hidden_size))

    def forward(self, hidden_states, q_resid, attention_mask=None, **kwargs):
        return torch.zeros(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.index_topk,
            dtype=torch.long,
            device=hidden_states.device,
        )


class DeepseekSparseAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.scaling = self.qk_head_dim**-0.5

        self.indexer = Indexer(config, layer_idx)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = hidden_states.shape[:2]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        key_states, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*key_states.shape[:-1], -1)
        key_states = torch.cat([key_states, k_rot], dim=-1)

        topk_indices = kwargs.get('prev_topk_indices', None)
        if topk_indices is None:
            topk_indices = self.indexer(hidden_states, q_resid)

        # 构建 topk 稀疏注意力掩码
        kv_length = key_states.shape[2]
        topk_valid = topk_indices.ge(0) & topk_indices.lt(kv_length)
        safe_indices = topk_indices.clamp(0, kv_length - 1)
        selected_counts = torch.zeros(
            topk_indices.shape[0],
            topk_indices.shape[1],
            kv_length,
            dtype=torch.int32,
            device=topk_indices.device,
        )
        selected_counts.scatter_add_(-1, safe_indices, topk_valid.to(torch.int32))
        mask = selected_counts.ne(0)
        min_dtype = torch.finfo(query_states.dtype).min
        attention_mask = torch.where(
            mask.unsqueeze(1),
            torch.tensor(0.0, device=query_states.device, dtype=query_states.dtype),
            min_dtype,
        )

        attn_weights = torch.matmul(query_states.float(), key_states.transpose(-2, -1).float()) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()).type_as(x)


class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.n_groups = config.n_group
        self.topk_groups = config.topk_group
        self.score_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        scores = scores.sigmoid()
        original_scores = scores
        scores = scores + self.e_score_correction_bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class ExpertMLP(nn.Module):
    """单个专家 MLP，per-expert nn.Linear 形态与真实权重格式一致。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()).type_as(x)


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Gate(config)
        self.num_experts = config.n_routed_experts
        # 直接以 nn.ModuleList 组织 ExpertMLP，权重路径与真实 checkpoint 一致：
        # mlp.experts.{i}.gate_proj.weight / up_proj.weight / down_proj.weight
        self.experts = nn.ModuleList(
            [ExpertMLP(config.hidden_size, config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_experts = ExpertMLP(config.hidden_size, config.n_shared_experts * config.moe_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = x
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        weights, indices = self.gate(x)

        # 按 gate 结果分发到命中的各专家 MLP
        final = torch.zeros_like(x)
        with torch.no_grad():
            mask = F.one_hot(indices, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx_tensor in hit:
            expert_idx = expert_idx_tensor[0].item()
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self.experts[expert_idx](x[token_idx])
            current = current * weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))

        x = final.view(*orig_shape)
        x = x + self.shared_experts(residuals)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size

        if self.block_type == "linear_attention":
            self.self_attn = LinearAttention(config, layer_idx)
        else:
            self.self_attn = DeepseekSparseAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dtype = hidden_states.dtype
        residual = hidden_states
        post, comb, collapsed = self.attn_hc(hidden_states)
        collapsed = self.input_layernorm(collapsed)

        if self.block_type == "linear_attention":
            attn_output = self.self_attn(collapsed, **kwargs)
            topk_indices = None
        else:
            attn_output = self.self_attn(collapsed, **kwargs)
            topk_indices = None

        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_states
        post, comb, collapsed = self.ffn_hc(hidden_states)
        collapsed = self.post_attention_layernorm(collapsed)
        ffn_output = self.mlp(collapsed)
        hidden_states = post.to(dtype).unsqueeze(-1) * ffn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        return hidden_states, topk_indices


class NextPredDecoderLayer(nn.Module):
    """nextn_predict (MTP) 解码层，对应 layer 45（num_nextn_predict_layers）。

    与常规 DecoderLayer 不同：
    - 不含 HC（HyperConnection）模块（权重 key 中无 hc_attn_*/hc_ffn_*）
    - 使用 DSA (DeepseekSparseAttention) + MoE
    - 通过 wrap_mtp_decoder 附加 enorm/hnorm/eh_proj/shared_head/embed_tokens
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block_type = (
            config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "deepseek_sparse_attention"
        )
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekSparseAttention(config, layer_idx)
        # MTP 层（idx >= num_hidden_layers）不在 mlp_layer_types 列表中，按 MoE（sparse）处理
        mlp_type = (
            config.mlp_layer_types[layer_idx]
            if config.mlp_layer_types and layer_idx < len(config.mlp_layer_types)
            else "sparse"
        )
        if mlp_type == "sparse":
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # MTP 层不含 HC streams，来自常规 DecoderLayer 的输出是 4D [B, S, head_dim, H]
        # 需要先 collapse 为 3D [B, S, H]（与 HyperHead.mean 一致）
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.mean(dim=2)

        residual = hidden_states
        collapsed = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(collapsed, **kwargs)
        hidden_states = attn_output + residual

        residual = hidden_states
        collapsed = self.post_attention_layernorm(hidden_states)
        ffn_output = self.mlp(collapsed)
        hidden_states = ffn_output + residual

        return hidden_states, None


class HyperHead(nn.Module):
    """HC 多流输出收敛：无权重 mean，与 transformers Glm5NextTextHyperHead 一致。"""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


class Glm5NextTextModel(nn.Module):
    """language_model 文本模型，与 transformers Glm5NextTextModel 结构一致。

    包含 embed_tokens、layers、norm、hc_head。
    """

    def __init__(self, config):
        super().__init__()
        self.head_dim = getattr(config, 'hc_mult', 1)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead()

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # 扩展出 HC 多流：[B, S, D] -> [B, S, head_dim, D]
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.head_dim, -1).contiguous()

        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        # 收敛 HC 多流并施加最终 norm，与 transformers Glm5NextTextModel.forward
        # 一致：hidden_states = self.norm(self.hc_head(hidden_states))
        hidden_states = self.norm(self.hc_head(hidden_states))
        return hidden_states


class Glm5NextModel(nn.Module):
    """主模型（VL），与 transformers Glm5NextModel 结构一致。

    包含 language_model（Glm5NextTextModel）；visual 视觉塔由适配器按需挂载。
    """

    def __init__(self, config):
        super().__init__()
        self.language_model = Glm5NextTextModel(config)


class Transformer(nn.Module):
    """顶层模型，与 transformers Glm5NextForConditionalGeneration 结构一致。

    包含 model（Glm5NextModel）与 lm_head。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Glm5NextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        hidden_states = self.model.language_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        return hidden_states
