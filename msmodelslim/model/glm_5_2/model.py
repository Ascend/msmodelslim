#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.
Copyright (c) 2025 GLM5 & ZhipuAI team
Copyright (c) 2023 DeepSeek

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

# modified from DeepSeek-V3.2-Exp/inference/model.py
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from msmodelslim.utils.distributed import DistHelper

WORLD_SIZE = 1
RANK = 0
USE_DP_MODE = True  # True: DP+EP mode, False: TP+EP mode
BLOCK_SIZE = 128


def _normalize_indexer_type(indexer_type) -> str:
    if isinstance(indexer_type, str):
        if indexer_type in ("F", "f"):
            return "full"
        if indexer_type in ("S", "s"):
            return "shared"
        return indexer_type.lower()
    return "full"


def get_indexer_type(args, layer_id: int) -> str:
    indexer_types = getattr(args, "indexer_types", None)
    num_layers = getattr(args, "num_hidden_layers", layer_id + 1)

    if indexer_types is None:
        pattern = getattr(args, "index_topk_pattern", None)
        if pattern is not None:
            indexer_types = [_normalize_indexer_type(item) for item in pattern]
        else:
            freq = getattr(args, "index_topk_freq", 1)
            indexer_types = ["full" if (max(i - 1, 0) % freq) == 0 else "shared" for i in range(num_layers)]
        if hasattr(args, "indexer_types"):
            args.indexer_types = indexer_types

    # MTP层有indexer
    if layer_id == 78:
        return "full"
    if layer_id < 0 or layer_id >= len(indexer_types):
        return "shared"
    return _normalize_indexer_type(indexer_types[layer_id])


def has_indexer(args, layer_id: int) -> bool:
    return get_indexer_type(args, layer_id) != "shared"


def fp8_index(q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor):
    logits = (torch.matmul(q.transpose(1, 2).to(torch.float32), k.permute(0, 2, 3, 1).to(torch.float32))).to(
        torch.float32
    )
    logits = torch.relu(logits)
    logits = logits * q_s.transpose(1, 2)
    logits = torch.sum(logits, dim=1)
    return logits


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
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
        index_head_dim (int): Dimension for index head.
        index_topk (int): Top-k for index head.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    first_k_dense_replace: int = 3
    num_attention_heads: int = 64
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    scoring_func: Literal["softmax", "sigmoid"] = "sigmoid"
    routed_scaling_factor: float = 2.5
    # mla
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    # index
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[str] = None
    index_topk_freq: int = 1
    # extra
    num_key_value_heads: int = 64
    rms_norm_eps: float = 1e-06


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.
    TP模式：切分词表；DP模式：不切分词表

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # TP模式下切分词表，DP模式下不切分
        if not USE_DP_MODE:
            self.part_vocab_size = vocab_size // WORLD_SIZE
            self.vocab_start_idx = RANK * self.part_vocab_size
        else:
            self.part_vocab_size = vocab_size
            self.vocab_start_idx = 0
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
        # TP模式下需要处理词表切分
        if WORLD_SIZE > 1 and not USE_DP_MODE:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if WORLD_SIZE > 1 and not USE_DP_MODE:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None
) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.
        scale_fmt (Optional[str]): The format of scaling factors.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
          is used for computation.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
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
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        dtype = x.dtype
        if residual is None:
            x = x.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype)
        else:
            x = residual = x.float() + residual.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype), residual.to(dtype)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_, max_, dim_):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min_ (float): Minimum value for the ramp function.
            max_ (float): Maximum value for the ramp function.
            dim_ (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min_ == max_:
            max_ += 0.001
        linear_func = (torch.arange(dim_, dtype=torch.float32) - min_) / (max_ - min_)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


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


class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim: int = args.hidden_size
        self.n_heads: int = args.index_n_heads
        # TP模式下切分注意力头，DP模式下不切分
        self.n_local_heads = args.index_n_heads if USE_DP_MODE else (args.index_n_heads // WORLD_SIZE)
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, dtype=torch.get_default_dtype(), bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = args.scale_fmt

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_scale = torch.ones(*q.size()[:-1], q.size(-1) // 128, dtype=torch.float32).npu()
        weights = self.weights_proj(x) * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        k = k.view(bsz, -1, 1, self.head_dim)
        index_score = fp8_index(q.contiguous(), weights, k)
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        topk_indices_ = topk_indices.clone()
        return topk_indices_


def weight_dequant(weight, scale):
    shape = weight.shape
    weight = (
        weight.view(shape[0] // BLOCK_SIZE, BLOCK_SIZE, shape[1] // BLOCK_SIZE, BLOCK_SIZE)
        .transpose(1, 2)
        .contiguous()
        .view(-1, BLOCK_SIZE * BLOCK_SIZE)
    )
    weight = (
        (weight.float() * scale.view(-1, 1).float())
        .to(torch.get_default_dtype())
        .view(shape[0] // BLOCK_SIZE, shape[1] // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        .transpose(1, 2)
        .contiguous()
        .view(shape)
    )
    return weight


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """

    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        # TP模式下切分注意力头，DP模式下不切分
        self.n_local_heads = args.num_attention_heads if USE_DP_MODE else (args.num_attention_heads // WORLD_SIZE)
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.q_a_proj = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.has_indexer = has_indexer(args, layer_id)
        indexer_types = getattr(args, "indexer_types", None)
        total_indexer_layers = (
            len(indexer_types) if indexer_types is not None else getattr(args, "num_hidden_layers", layer_id + 1)
        )
        self.next_skip_topk = layer_id + 1 < total_indexer_layers and not has_indexer(args, layer_id + 1)
        self.indexer = Indexer(args) if self.has_indexer else None
        self.last_topk_indices = None

        self.register_buffer(
            "kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False
        )
        self.register_buffer(
            "pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False
        )
        self.dequant_wkv_b = None

    def _get_topk_indices(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        prev_topk_indices: Optional[torch.Tensor],
    ):
        if self.indexer is not None:
            return self.indexer(x, qr, start_pos, freqs_cis, mask)
        if prev_topk_indices is None:
            raise RuntimeError(f"Layer {self.layer_id} reuses indexer topk, but previous topk indices are missing.")
        return prev_topk_indices

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        if mask is not None:  # MHA prefill
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.kv_b_proj(kv)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * self.softmax_scale

            # indexer
            topk_indices = self._get_topk_indices(x, qr, start_pos, freqs_cis, mask, prev_topk_indices)
            index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
            index_mask += mask
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1, dtype=torch.float32)
            x = torch.einsum("bsht,bthd->bshd", scores.type_as(x), v)
        else:  # MHA decode
            if self.dequant_wkv_b is None and self.kv_b_proj.scale is not None:
                self.dequant_wkv_b = weight_dequant(self.kv_b_proj.weight, self.kv_b_proj.scale)
            wkv_b = self.kv_b_proj.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim])
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope.float(), self.kv_cache[:bsz, :end_pos].float())
                + torch.einsum("bshr,btr->bsht", q_pe.float(), self.pe_cache[:bsz, :end_pos].float())
            ) * self.softmax_scale

            # indexer
            topk_indices = self._get_topk_indices(x, qr, start_pos, freqs_cis, mask, prev_topk_indices)
            index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1, dtype=torch.float32)
            x = torch.einsum("bsht,btc->bshc", scores.type_as(x), self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        x = self.o_proj(x.flatten(2))
        self.last_topk_indices = topk_indices if self.next_skip_topk else None
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int, reduce_output: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj((F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()).type_as(x))


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

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.hidden_size
        self.topk = args.num_experts_per_tok
        self.n_groups = args.n_group
        self.topk_groups = args.topk_group
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = (
            nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 6144 else None
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.e_score_correction_bias is not None:
            scores = scores + self.e_score_correction_bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.e_score_correction_bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
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

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj((F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()).type_as(x))


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

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.hidden_size
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // WORLD_SIZE
        self.n_activated_experts = args.num_experts_per_tok
        self.experts_start_idx = RANK * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [
                Expert(args.hidden_size, args.moe_intermediate_size)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = MLP(
            args.hidden_size, args.n_shared_experts * args.moe_intermediate_size, reduce_output=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DP模式：需要gather不同rank的输入（因为每个rank有完整模型但处理不同数据）
        # TP模式：不需要gather（因为不同rank处理不同的张量分片和专家，输入已经按专家路由）
        if WORLD_SIZE > 1 and USE_DP_MODE:
            seq_len_this_rank = x.size(-2)  # 当前rank的seq_len

            # 同步所有rank的seq_len
            seq_len_tensor = torch.tensor([seq_len_this_rank], dtype=torch.long, device=x.device)
            seq_len_list = [torch.zeros_like(seq_len_tensor) for _ in range(WORLD_SIZE)]
            dist.all_gather(seq_len_list, seq_len_tensor)
            seq_lens = [s.item() for s in seq_len_list]

            # 计算当前rank的起始位置（前面所有rank的token数量之和）
            start_pos = sum(seq_lens[:RANK])
            end_pos = start_pos + seq_len_this_rank

            # gather所有rank的输入
            x = torch.cat(DistHelper.gather_variable_shapes(x), dim=1)
        else:
            start_pos = 0
            end_pos = x.size(-2)

        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # 处理本地专家（专家在所有rank间切分，无论DP+EP还是TP+EP模式）
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        y += self.shared_experts(x)

        # 专家并行需要all_reduce汇总结果
        if WORLD_SIZE > 1:
            dist.all_reduce(y)
            # 筛选出本rank的token
            if USE_DP_MODE:
                return y.type_as(x).view(shape)[:, start_pos:end_pos, :]
            else:
                return y.type_as(x).view(shape)

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
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = MLA(args, layer_id=layer_id)
        self.mlp = MLP(args.hidden_size, args.intermediate_size) if layer_id < args.first_k_dense_replace else MoE(args)
        self.input_layernorm = RMSNorm(args.hidden_size)
        self.post_attention_layernorm = RMSNorm(args.hidden_size)
        self.last_topk_indices = None

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        prev_topk_indices: Optional[torch.Tensor] = None,
        topk_collector: Optional[Any] = None,
    ) -> torch.Tensor:
        if residual is None:
            x, residual = self.input_layernorm(x), x
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(x, start_pos, freqs_cis, mask, prev_topk_indices=prev_topk_indices)
        self.last_topk_indices = self.self_attn.last_topk_indices
        if topk_collector is not None:
            topk_collector.value = self.last_topk_indices
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class GLMMOEModel(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: ModelArgs):
        global WORLD_SIZE, RANK
        WORLD_SIZE = dist.get_world_size() if dist.is_initialized() else 1
        RANK = dist.get_rank() if dist.is_initialized() else 0

        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed_tokens = ParallelEmbedding(args.vocab_size, args.hidden_size)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.hidden_size)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1) if seqlen > 1 else None
        h, residual = self.embed_tokens(tokens), None
        topk_indices = None
        for layer in self.layers:
            h, residual = layer(
                h,
                residual,
                start_pos=start_pos,
                freqs_cis=freqs_cis,
                mask=mask,
                prev_topk_indices=topk_indices,
            )
            topk_indices = layer.last_topk_indices
        return self.norm(h, residual)


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = GLMMOEModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, dtype=torch.float32, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        h, _ = self.model(tokens, start_pos)
        logits = self.lm_head(h[:, -1].float())
        # TP模式：需要all_gather logits（因为词表被切分）
        # DP模式：不需要all_gather（每个rank有完整词表）
        if WORLD_SIZE > 1 and not USE_DP_MODE:
            all_logits = [torch.empty_like(logits) for _ in range(WORLD_SIZE)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits
