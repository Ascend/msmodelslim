# -*- coding: UTF-8 -*-
# Copyright 2026 DeepSeek-AI and The Huggingace Inc. team. All rights reserved.
#
# This code is based on the official code of DeepSeek-V4-Flash:
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py
# It is modified to support the quantization of the model by msModelSlim.
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

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
# from kernel import act_quant, fp8_gemm, sparse_attn, hc_split_sinkhorn


world_size = 1
rank = 0
block_size = 128


def hc_split_sinkhorn(
    mixes: torch.Tensor,  # [b, s, mix_hc] => [b, s, (2 + hc) * hc]
    hc_scale: torch.Tensor,  # [3]
    hc_base: torch.Tensor,  # [(2 + hc) * hc]
    hc_mult: int = 4,  # hc
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    # Ensure all tensors are on the same device
    device = mixes.device
    hc_scale = hc_scale.to(device)
    hc_base = hc_base.to(device)

    b, s, _ = mixes.size()
    # get pre
    mixes_pre = mixes[:, :, :hc_mult]
    hc_scale_pre = hc_scale[0]
    hc_base_pre = hc_base[:hc_mult]
    pre = F.sigmoid(hc_scale_pre * mixes_pre + hc_base_pre)
    # get post
    mixes_post = mixes[:, :, hc_mult : 2 * hc_mult]
    hc_scale_post = hc_scale[1]
    hc_base_post = hc_base[hc_mult : 2 * hc_mult]
    post = 2 * F.sigmoid(hc_scale_post * mixes_post + hc_base_post)
    # get comb
    # step 1: init comb
    mixes_comb = mixes[:, :, 2 * hc_mult :]
    hc_scale_comb = hc_scale[2]
    hc_base_comb = hc_base[2 * hc_mult :]
    comb = (hc_scale_comb * mixes_comb + hc_base_comb).reshape(b, s, hc_mult, hc_mult)  # [b, s, hc, hc]
    comb = F.softmax(comb, dim=-1) + eps
    # step 2: do sinkhorn ops
    for _ in range(sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1).unsqueeze(-1) + eps)
        comb = comb / (comb.sum(dim=-2).unsqueeze(-2) + eps)

    return pre, post, comb


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    b, m, h, d = q.shape
    device = q.device
    topk_idxs = topk_idxs.to(device)

    # Prepare indices: clamp -1 to 0 for gathering, but keep mask
    mask = topk_idxs == -1
    safe_idxs = topk_idxs.clone()
    safe_idxs[mask] = 0

    # Gather KV: (b, m, topk, d)
    batch_indices = torch.arange(b, device=kv.device).view(b, 1, 1)
    kv_gathered = kv[batch_indices, safe_idxs, :]

    # Compute Scores (FP32)
    q_f32 = q.float()
    kv_f32 = kv_gathered.float()

    # (b, m, h, 1, d) @ (b, m, 1, topk, d)^T -> (b, m, h, topk)
    scores = torch.matmul(q_f32.unsqueeze(3), kv_f32.unsqueeze(2).transpose(-1, -2)).squeeze(3)
    scores = scores * softmax_scale
    scores = scores.masked_fill(mask.unsqueeze(2), float("-inf"))

    # Softmax logic with Sink
    scores_max = torch.max(scores, dim=-1).values  # (b, m, h)
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))
    exp_scores = exp_scores.masked_fill(mask.unsqueeze(2), 0.0)

    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.float().view(1, 1, h) - scores_max)
    total_denominator = sum_exp + sink_term

    # Weighted Sum
    numerator = torch.matmul(exp_scores.unsqueeze(3), kv_f32.unsqueeze(2)).squeeze(3)
    output = numerator / total_denominator.unsqueeze(-1)

    return output.to(q.dtype)


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_hash_layers (int): Number of hash MoE layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        head_dim (int): Dimension for attention.
        rope_head_dim (int): Dimension for rotary embeddings.
        o_groups (int): Number of groups of out projections.
        o_lora_rank (int): Dimension for out projections.
        window_size (int): Window size of sliding window attention.
        compress_ratios (list[int]): Each layer's compress_ratio.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        index_n_heads (int): Number of index heads.
        index_head_dim (int): Dimension for index head.
        index_topk (int): Top-k for index head.
        hc_mult (int): HC hidden size multiplier.
        sinkhorn_iters (int): Number of sinkhorn iterations for HC.
    """

    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = "ue8m0"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048  # 4096
    num_hidden_layers: int = 43
    n_hash_layers: int = 3
    n_heads: int = 64
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (
        1,
        1,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        1,
    )
    # yarn
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536  # 0
    rope_theta: float = 10000.0
    rope_factor: float = 4  # 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, (
            f"Vocabulary size must be divisible by world size (world_size={world_size})"
        )  # nosec
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None
) -> torch.Tensor:
    return F.linear(x, weight)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def hadamard_transform_ref(x: torch.Tensor, scale=1.0):
    from scipy.linalg import hadamard

    if hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, scale=hidden_size**-0.5)


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    def _get_window_topk_idxs():
        if start_pos >= window_size - 1:
            return torch.arange(window_size)
        elif start_pos > 0:
            return F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
        else:
            base = torch.arange(seqlen).unsqueeze(1)
            matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
            matrix = torch.where(matrix > base, -1, matrix)
            return matrix

    return _get_window_topk_idxs().unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    def _get_compress_topk_idxs():
        if start_pos > 0:
            return torch.arange(0, start_pos // ratio) + offset
        else:
            matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
            mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            matrix = torch.where(mask, -1, matrix + offset)
            return matrix

    return _get_compress_topk_idxs().unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        # wkv and wgate in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        # The first half of dimensions for overlapping compression and second half for normal compression.
        self.wkv = nn.Linear(self.dim, coff * self.head_dim, dtype=torch.float32, bias=False)
        self.wgate = nn.Linear(self.dim, coff * self.head_dim, dtype=torch.float32, bias=False)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache = None
        # If overlap is enabled, state[:, :ratio] for overlapping compression and state[:, ratio:] for normal compression.
        self.register_buffer(
            "kv_state",
            torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32
            ),
            persistent=False,
        )

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor):
        assert self.kv_cache is not None  # nosec
        bsz, seqlen, _ = x.size()
        ratio, overlap, d = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            freqs_cis = freqs_cis[:cutoff:ratio]
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset : offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat(
                        [self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, dtype=torch.bfloat16, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, offset: int):
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        self.compressor(x, start_pos, freqs_cis)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            device = index_score.device
            mask = (
                torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
                >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
            )
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=topk_idxs.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    """Multi-Query Attention (MQA) Layer."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        # self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
        )
        # self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.wo_b = nn.Linear(self.n_groups * args.o_lora_rank, self.dim, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = args.scale_fmt

        if self.compress_ratio > 1:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.window_size + args.max_seq_len // self.compress_ratio, self.head_dim),
            persistent=False,
        )
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            args.original_seq_len,
            args.compress_rope_theta if self.compress_ratio > 1 else args.rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio > 1 and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)
        if self.compress_ratio > 1:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, freqs_cis, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        # if start_pos == 0:
        if True:
            self.kv_cache[:bsz, : min(win, seqlen)] = kv[:, -win:]
            if self.compress_ratio > 1:
                if (kv_compress := self.compressor(x, start_pos, freqs_cis)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio > 1:
                self.compressor(x, start_pos, freqs_cis)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        # self.wo_a.weight [8192, 4096]; self.n_local_groups=8; self.o_lora_rank=1024
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            input_ids (torch.Tensor): Token IDs tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int, swiglu_limit=0):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        # x = F.silu(self.w1(x).float()) * self.w3(x).float()
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, (
            f"Number of experts must be divisible by world size (world_size={world_size})"
        )  # nosec
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        self.experts = nn.ModuleList(
            [
                Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        assert args.n_shared_experts == 1  # nosec
        # no swiglu_limit for shared experts
        self.shared_experts = Expert(args.dim, args.moe_inter_dim)

    def run_gate(self, x, input_ids: Optional[torch.Tensor] = None):
        return self.gate(x, input_ids)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.run_gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        if world_size > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        torch.set_default_dtype(origin_dtype)

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
        device = x.device
        hc_fn = hc_fn.to(device)
        hc_scale = hc_scale.to(device)
        hc_base = hc_base.to(device)
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.
        # self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.float32)
        self.hc_eps = args.hc_eps
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))
        torch.set_default_dtype(origin_dtype)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False, dtype=torch.float32)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        device = x.device
        hc_fn = hc_fn.to(device)
        hc_scale = hc_scale.to(device)
        hc_base = hc_base.to(device)
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        h = self.norm(h)

        # logging.info(f"h : {h}")
        logits = self.head(h[:, -1].float())

        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits
        # return h


class DeepSeekModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = Transformer(args)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False, dtype=torch.float32)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        h = self.model(input_ids, start_pos)
        logging.info(f"h : {h}")
        logits = self.lm_head(h[:, -1].float())

        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("npu")
    torch.manual_seed(0)
    args = ModelArgs(n_hash_layers=0)
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = DeepSeekModel(args)

    print(model(x).size())
    for i in range(128, 150):
        print(i, model(x[:, 0:1], i).size())
