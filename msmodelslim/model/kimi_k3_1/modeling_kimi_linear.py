#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=C0302  # mirrors the released 1108-line modelling_kimi_linear.py, kept in one module

# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# This file is modified from the released Kimi-K3 ``modeling_kimi_linear.py`` to support the Kimi-K3.1 model.

# The multi-head latent attention, MoE gating and sparse MoE block in this file are
# adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py). They have been
# extensively modified and extended for the Kimi-Linear architecture.
#
# Licensing Information:
# - Code adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the Kimi K3 License (see the LICENSE file in this repository).
#
# Apache License, Version 2.0:
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

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn

#: KDA is copied verbatim from the released Kimi-K3 ``modeling_kimi_linear.py``: the same
#: ``fla`` ops, the same ``ShortConvolution`` / ``FusedRMSNormGated`` modules and the same
#: unpad helpers.  ``fla`` is a hard requirement, exactly like Kimi-K3 (it raises at import
#: time there too); ``kimi_k3/runtime_patches`` then swaps the Triton ``chunk_kda`` for the
#: naive PyTorch path on Ascend, so K3.1 executes byte-identical kernel code.
from fla.modules import FusedRMSNormGated, ShortConvolution

#: ``fused_recurrent_kda`` is unused by the prefill-only path, but Kimi-K3's runtime patch
#: detects the modeling module by both names, so it must stay imported.
from fla.ops.kda import chunk_kda, fused_recurrent_kda  # noqa: F401
from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
from fla.utils import tensor_cache

from msmodelslim.model.common.utils import resolve_expert_ep_range
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.logging import get_logger

#: Upper bound of the ``[B, H, q_chunk, kv]`` fp32 score buffer of one DSA indexer chunk.
#: Only a memory knob: it never changes which keys are selected.
_INDEX_SCORE_BUDGET_BYTES = 256 << 20

#: Default ``k_norm`` epsilon of the DSA indexer.  Overridable per snapshot through
#: ``dsa_config.index_k_norm_eps``.
DEFAULT_K_NORM_EPS = 1e-6


def get_dsa_config(config: Any) -> Dict[str, Any]:
    """Return the ``dsa_config`` block of a Kimi-K3.1 text config (empty when absent)."""
    dsa = getattr(config, "dsa_config", None)
    return dsa if isinstance(dsa, dict) else {}


def dsa_layer_ids(config: Any) -> frozenset:
    """0-based indices of the layers that own a DSA indexer.

    ``dsa_config["dsa_layers"]`` follows ``linear_attn_config`` and is **1-based**, while
    parameter names are 0-based: ``dsa_layers = [4, 8, ...]`` -> ``layers.3, layers.7, ...``.
    """
    raw = get_dsa_config(config).get("dsa_layers") or ()
    return frozenset(int(layer) - 1 for layer in raw)


def has_dsa_indexer(config: Any, layer_idx: int) -> bool:
    """Whether ``layer_idx`` (0-based) carries a DSA indexer."""
    if get_dsa_config(config).get("index_topk") is None:
        return False
    return layer_idx in dsa_layer_ids(config)


def is_kda_layer(config: Any, layer_idx: int) -> bool:
    """``True`` for KDA (linear-attention) layers; ``linear_attn_config`` is 1-based."""
    linear = getattr(config, "linear_attn_config", None) or {}
    return (layer_idx + 1) in (linear.get("kda_layers") or [])


def is_dense_mlp(config: Any, layer_idx: int) -> bool:
    return layer_idx < int(getattr(config, "first_k_dense_replace", 0) or 0)


def use_attn_residuals(config: Any) -> bool:
    return getattr(config, "attn_res_block_size", None) is not None


def kda_use_full_rank_gate(config: Any) -> bool:
    linear = getattr(config, "linear_attn_config", None) or {}
    return bool(linear.get("use_full_rank_gate", False))


def hidden_act_betas(config: Any) -> Tuple[float, Optional[float]]:
    beta = getattr(config, "activation_situ_beta", None) or 1.0
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return float(beta), None if linear_beta is None else float(linear_beta)


class SituAndMul(nn.Module):
    """SiTU gated activation: ``situ(gate) * linear(up)``.

    ``gate``/``up`` are the two halves of the concatenated ``w1``/``w3`` output, so the
    first half is the gate branch (``--x`` for ``x <= 0`` in the released config).
    """

    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        half = x.shape[-1] // 2
        gate = x[..., :half].float()
        up = x[..., half:].float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(dtype)


class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiRMSNorm,
) -> torch.Tensor:
    """AttnRes: softmax-weighted mix of the block snapshots plus the running prefix sum.

    ``prefix_sum`` is ``[tokens, hidden]`` and ``block_residual`` is
    ``[tokens, num_blocks, hidden]``; both are mixed in fp32 and cast back afterwards.
    With no stored block snapshot the mix degenerates to ``prefix_sum`` itself.
    """
    if block_residual is None or block_residual.shape[1] == 0:
        return prefix_sum
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = values.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    keys = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (keys * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1).to(values.dtype)


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather the non-padding rows -- copied from the released Kimi-K3 modeling file."""
    return x[indices]


@tensor_cache
def get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """``(indices, cu_seqlens, max_seqlen)`` of a ``[B, S]`` mask (Kimi-K3 helper)."""
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states: torch.Tensor, indices: torch.LongTensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Scatter un-padded rows back into ``[B, S, ...]`` -- copied from Kimi-K3."""
    out = hidden_states.new_zeros((batch_size * seq_len, *hidden_states.shape[1:]))
    out[indices] = hidden_states
    return out.view(batch_size, seq_len, *hidden_states.shape[1:])


class KimiDeltaAttention(nn.Module):
    """KDA layer: short causal conv -> gated delta rule -> gated RMSNorm -> o_proj."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        linear = config.linear_attn_config
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = int(linear["head_dim"])
        self.num_heads = int(linear["num_heads"])
        self.conv_size = int(linear["short_conv_kernel_size"])
        self.gate_lower_bound = linear.get("gate_lower_bound", None)
        self.rms_norm_eps = config.rms_norm_eps

        projection_size = self.head_dim * self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        for name in ("q", "k", "v"):
            # Verbatim Kimi-K3: ``fla`` ``ShortConvolution`` with ``activation='silu'``.
            setattr(
                self,
                f"{name}_conv1d",
                ShortConvolution(hidden_size=projection_size, kernel_size=self.conv_size, activation="silu"),
            )
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        if kda_use_full_rank_gate(config):
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        # Verbatim Kimi-K3: ``fla`` fused gated RMSNorm, same ``o_norm.weight`` parameter.
        self.o_norm = FusedRMSNormGated(self.head_dim, eps=self.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Kimi-K3's ``KimiDeltaAttention.forward``, prefill only (no ``KimiDynamicCache``).

        Calibration never decodes, so ``cache_params``/``mode`` are pinned to the chunk
        kernel and the conv final state is dropped; everything else -- the unpad step, the
        ``chunk_kda`` keyword set and the gated ``o_norm`` -- is copied verbatim.
        """
        del attention_mask, kwargs
        batch_size, q_len, _ = hidden_states.shape

        indices = None
        cu_seqlens = None
        if padding_mask is not None:
            if padding_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here.",
                )
            # Verbatim Kimi-K3: unpad once, then run the kernels on the packed token stream.
            indices, cu_seqlens, _ = get_unpad_data(padding_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        q, _ = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        k, _ = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        v, _ = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        beta = self.b_proj(hidden_states).float()

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )

        if kda_use_full_rank_gate(self.config):
            gate = self.g_proj(hidden_states)
        else:
            gate = self.g_b_proj(self.g_a_proj(hidden_states))
        gate = rearrange(gate, '... (h d) -> ... h d', d=self.head_dim)
        o = self.o_norm(o, gate)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if indices is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o


class KimiDSAIndexer(nn.Module):
    """DeepSeek-style sparse attention (DSA) indexer of the 24 NoPE MLA layers.

    ``index_topk`` keys are kept per query::

        q     = wq_b(q_a_layernorm(q_a_proj(x)))          # [B, S, H * D]
        k     = k_norm(wk(x))                             # [B, S, D]
        score = relu(q @ k^T / sqrt(D))
        index = sum_h weights_proj(x)[h] * score[h]       # [B, S, S]
        keep  = top-k(index)                              # [B, S, topk]

    K3.1's ``dsa_config`` pins every free choice of the released indexer: no k-pool
    compression, ``index_rope_dim = 0`` (Kimi-K3 is a NoPE MLA model), ``indexer_use_fp8 =
    False`` and ``index_topk_freq = 1`` (each full-attention layer owns its indexer, no
    cross-layer sharing).  Anything else is rejected instead of silently mis-modelled.

    There is deliberately **no** always-include-self / tail rule: the selection is exactly
    ``top-k`` over the causal+padding-filtered candidates, matching
    ``GlmMoeDsaIndexer`` (GLM-5.2 / GLM-5.3).  The only key that can be lost to a ``relu``
    tie is the diagonal at ``kv_len > index_topk``, which is the released behaviour too.
    """

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        dsa = get_dsa_config(config)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.n_heads = int(dsa.get("index_num_heads", 32))
        self.head_dim = int(dsa.get("index_head_dim", 128))
        self.index_topk = int(dsa.get("index_topk", 2048))
        self.rope_head_dim = int(dsa.get("index_rope_dim", 0) or 0)
        self.use_fp8 = bool(dsa.get("indexer_use_fp8", False))
        self.softmax_scale = float(self.head_dim) ** -0.5
        # ``index_skip_topk_offset`` only participates in GLM's *layer schedule*
        # (``"full" if max(i - offset + 1, 0) % freq == 0 else "shared"``); it never alters
        # the per-token top-k.  With K3.1's ``index_topk_freq = 1`` every layer is "full",
        # so the field is inert here and no key is force-kept beyond the top-k budget --
        # exactly like ``GlmMoeDsaIndexer`` (GLM-5.2 / GLM-5.3), which has no
        # always-include-self / tail mechanism.
        self.skip_topk_offset = int(dsa.get("index_skip_topk_offset", 1) or 0)
        if self.skip_topk_offset > 1:
            get_logger().warning(
                "dsa_config.index_skip_topk_offset=%d only shifts GLM's full/shared layer "
                "schedule; Kimi-K3.1 keeps index_topk_freq=1, so it has no effect.",
                self.skip_topk_offset,
            )

        if self.q_lora_rank is None:
            raise ValueError("Kimi-K3.1 DSA indexer requires the low-rank q path (q_lora_rank).")
        if self.rope_head_dim:
            raise NotImplementedError("Kimi-K3.1 dsa_config must keep index_rope_dim=0.")
        if self.use_fp8:
            raise NotImplementedError("Kimi-K3.1 dsa_config.indexer_use_fp8 must be False.")
        if int(dsa.get("index_topk_freq", 1)) != 1:
            raise NotImplementedError("Kimi-K3.1 dsa_config.index_topk_freq > 1 would need cross-layer top-k sharing.")

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        norm_type = str(dsa.get("index_k_norm_type", "layernorm")).lower()
        # ``k_norm`` of the released indexer uses ``rms_norm_eps`` (1e-6) rather than
        # ``nn.LayerNorm``'s 1e-5 default.  The value stays overridable per snapshot via
        # ``dsa_config.index_k_norm_eps`` (or a top-level ``index_k_norm_eps``).
        eps_override = dsa.get("index_k_norm_eps", getattr(config, "index_k_norm_eps", None))
        self.k_norm_eps = DEFAULT_K_NORM_EPS if eps_override is None else float(eps_override)
        if norm_type == "layernorm":
            self.k_norm = nn.LayerNorm(self.head_dim, eps=self.k_norm_eps)
        elif norm_type == "rmsnorm":
            self.k_norm = KimiRMSNorm(self.head_dim, eps=self.k_norm_eps)
        else:
            raise NotImplementedError(f"Unsupported dsa_config.index_k_norm_type: {norm_type}")

    @staticmethod
    def _visibility(
        attention_mask: Optional[torch.Tensor],
        batch: int,
        seq_len: int,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.dim() == 4:
            mask = attention_mask[:, 0]
            return mask if mask.dtype == torch.bool else mask == 0
        positions = torch.arange(kv_len, device=device)
        visible = positions[None, :] <= positions[:, None]
        visible = visible[None].expand(batch, -1, -1)
        if attention_mask is not None and attention_mask.dim() == 2:
            visible = visible & attention_mask[:, -kv_len:].to(device=device, dtype=torch.bool)[:, None, :]
        return visible

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a ``[B, 1, S, kv]`` boolean keep-mask (``True`` = attend)."""
        batch, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        keys = self.k_norm(self.wk(hidden_states)).float()
        kv_len = keys.shape[1]
        visible = self._visibility(attention_mask, batch, seq_len, kv_len, device)

        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
        weights = weights * (self.n_heads**-0.5)
        queries = self.wq_b(q_resid).float()

        select_k = min(self.index_topk, kv_len)
        selected = torch.full((batch, seq_len, select_k), -1, dtype=torch.long, device=device)

        per_query_bytes = max(1, batch * self.n_heads * kv_len * 4)
        chunk = max(1, min(seq_len, _INDEX_SCORE_BUDGET_BYTES // per_query_bytes))
        finfo_min = torch.finfo(torch.float32).min
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            q_chunk = queries[:, start:end].view(batch, end - start, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q_chunk, keys.transpose(-1, -2).unsqueeze(1))
            scores = F.relu(scores * self.softmax_scale)
            index_scores = torch.einsum("blh,bhlk->blk", weights[:, start:end], scores)
            index_scores = index_scores.masked_fill(~visible[:, start:end], finfo_min)
            top = index_scores.topk(select_k, dim=-1).indices
            keep = visible[:, start:end].gather(-1, top)
            selected[:, start:end] = top.masked_fill(~keep, -1)

        valid = selected.ge(0) & selected.lt(kv_len)
        counts = torch.zeros(batch, seq_len, kv_len, dtype=torch.int32, device=device)
        counts.scatter_add_(-1, selected.clamp(0, kv_len - 1), valid.to(torch.int32))
        return counts.ne(0).unsqueeze(1)


def _mask_to_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``True -> 0`` / ``False -> dtype_min``; fully masked rows degrade to uniform."""
    return torch.where(
        mask,
        torch.zeros((), dtype=dtype, device=mask.device),
        torch.full((), torch.finfo(dtype).min, dtype=dtype, device=mask.device),
    )


class KimiDSAAttention(nn.Module):
    """NoPE MLA full attention (Kimi-K3 layout) with an optional DSA indexer."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.use_output_gate = bool(getattr(config, "mla_use_output_gate", False))
        self.scaling = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)
        self.indexer = KimiDSAIndexer(config, layer_idx) if has_dsa_indexer(config, layer_idx) else None

    def _absorbed_attention(
        self,
        q_nope: torch.Tensor,
        q_tail: torch.Tensor,
        latent: torch.Tensor,
        k_tail: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """FA3 / Ascend MLA-absorb formulation with ``fa_q``/``fa_k``/``fa_v`` taps.

        ``k_nope = latent @ q_absorb^T`` and ``v = latent @ out_absorb^T`` are folded into
        the attention itself, so only the 512-wide latent is materialised per key.  This is
        the exact algebra the Ascend absorb-MLA kernel runs; the placeholder modules let the
        quantizer calibrate those tensors.
        """
        kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
        out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]
        q_nope = torch.matmul(q_nope, q_absorb)

        if hasattr(self, "fa_q"):
            q_nope = self.fa_q(q_nope)
        if hasattr(self, "fa_k"):
            latent = self.fa_k(latent.unsqueeze(1)).squeeze(1)
        if hasattr(self, "fa_v"):
            _ = self.fa_v(latent.unsqueeze(1)).squeeze(1)

        scores = torch.matmul(q_tail, k_tail.mT) + torch.matmul(q_nope, latent.mT)
        scores = scores * self.scaling
        if mask is not None:
            scores = scores + _mask_to_additive(mask, scores.dtype)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_nope.dtype)
        out = torch.matmul(weights, latent)
        return torch.matmul(out, out_absorb.mT)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del padding_mask, kwargs
        batch, seq_len, _ = hidden_states.shape
        query_shape = (batch, seq_len, self.num_heads, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_resid = None
            query_states = self.q_proj(hidden_states)
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(query_shape).transpose(1, 2)
        q_nope, q_tail = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        latent, k_tail = torch.split(compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent = self.kv_a_layernorm(latent)
        # ``mla_use_nope``: the 64 "rope" channels are projected but never rotated, so the
        # single MQA copy is broadcast over the heads at forward time.
        k_tail = k_tail.view(batch, 1, seq_len, self.qk_rope_head_dim).expand(
            batch, self.num_heads, seq_len, self.qk_rope_head_dim
        )

        mask = attention_mask
        if self.indexer is not None:
            if q_resid is None:  # pragma: no cover - rejected in KimiDSAIndexer.__init__
                raise ValueError("Kimi-K3.1 DSA indexer requires q_resid.")
            keep = self.indexer(hidden_states, q_resid, mask)
            mask = keep if mask is None else (mask & keep)

        if hasattr(self, "fa_q"):
            out = self._absorbed_attention(q_nope, q_tail, latent, k_tail, mask)
        else:
            expanded = (
                self.kv_b_proj(latent)
                .view(batch, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )
            k_nope, value_states = torch.split(expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            query_states = torch.cat((q_nope, q_tail), dim=-1)
            key_states = torch.cat((k_nope, k_tail), dim=-1)
            out = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None if mask is None else _mask_to_additive(mask, query_states.dtype),
                scale=self.scaling,
            )
        out = out.transpose(1, 2).reshape(batch, seq_len, -1).contiguous()
        if self.use_output_gate:
            out = out * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(out)


class KimiMLP(nn.Module):
    """Dense SiTU MLP of layer 0 and of the shared experts."""

    def __init__(self, config: Any, hidden_size: Optional[int] = None, intermediate_size: Optional[int] = None):
        super().__init__()
        in_dim = config.hidden_size if hidden_size is None else hidden_size
        ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(in_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, in_dim, bias=False)
        beta, linear_beta = hidden_act_betas(config)
        self.act_fn = SituAndMul(beta=beta, linear_beta=linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))


class KimiBlockSparseMLP(nn.Module):
    """One routed expert (``w1`` gate, ``w3`` up, ``w2`` down)."""

    def __init__(self, config: Any, hidden_size: Optional[int] = None, intermediate_size: Optional[int] = None):
        super().__init__()
        in_dim = config.hidden_size if hidden_size is None else hidden_size
        ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.w1 = nn.Linear(in_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, in_dim, bias=False)
        self.w3 = nn.Linear(in_dim, ffn_dim, bias=False)
        beta, linear_beta = hidden_act_betas(config)
        self.act_fn = SituAndMul(beta=beta, linear_beta=linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.w1(x), self.w3(x)], dim=-1)
        return self.w2(self.act_fn(gate_up))


class KimiMoEGate(nn.Module):
    """Aux-loss-free (``noaux_tc``) sigmoid router of Kimi-K3 / K3.1."""

    def __init__(self, config: Any):
        super().__init__()
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.score_func = config.moe_router_activation_func
        self.num_expert_group = int(getattr(config, "num_expert_group", 1) or 1)
        self.topk_group = int(getattr(config, "topk_group", 1) or 1)
        self.moe_renormalize = config.moe_renormalize
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        logits = F.linear(flat, self.weight.float())
        if self.score_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.score_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:  # pragma: no cover - config assert already restricts the values
            raise NotImplementedError(f"Unsupported moe_router_activation_func: {self.score_func}")

        scores_for_choice = scores + self.e_score_correction_bias
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.view(flat.shape[0], self.num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(flat.shape[0], self.num_expert_group, self.num_experts // self.num_expert_group)
                .reshape(flat.shape[0], -1)
            )
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

        topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.moe_renormalize:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor


class KimiSparseMoeBlock(nn.Module):
    """Latent MoE with DP+EP sharding.

    Weights are laid out exactly like the checkpoint (``experts.{i}.w1/w2/w3``,
    ``routed_expert_down_proj`` / ``routed_expert_norm`` / ``routed_expert_up_proj``).
    Under ``torch.distributed`` with ``world_size > 1`` every rank keeps only its own
    contiguous expert slice; activations are all-gathered, the partial expert sums are
    all-reduced and the result is sliced back, mirroring ``kimi_k3/ep_patches.py``.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        self.latent_moe_use_norm = bool(getattr(config, "latent_moe_use_norm", False))

        self.ep_size, ep_rank, self.experts_start_idx, self.experts_end_idx = resolve_expert_ep_range(self.num_experts)
        if self.ep_size > 1:
            get_logger().info(
                "Kimi-K3.1 MoE EP: rank=%d/%d experts=[%d, %d) of %d",
                ep_rank,
                self.ep_size,
                self.experts_start_idx,
                self.experts_end_idx,
                self.num_experts,
            )

        self.experts = nn.ModuleList(
            [
                KimiBlockSparseMLP(
                    config,
                    hidden_size=self.moe_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                )
                if self.experts_start_idx <= idx < self.experts_end_idx
                else None
                for idx in range(self.num_experts)
            ]
        )
        self.gate = KimiMoEGate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = KimiMLP(
                config,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            )
        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(config.hidden_size, self.moe_hidden_size, bias=False)
            self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size, config.hidden_size, bias=False)
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)

    def _local_expert_forward(
        self, tokens: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros(tokens.shape[0], tokens.shape[-1], dtype=torch.float32, device=tokens.device)
        for expert_idx in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[expert_idx]
            if expert is None:  # pragma: no cover - only under malformed EP state
                continue
            token_idx, slot_idx = torch.where(topk_idx == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(tokens[token_idx])
            out[token_idx] += expert_out.float() * topk_weight[token_idx, slot_idx].float().unsqueeze(-1)
        return out.type_as(tokens)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        use_dp_ep = self.ep_size > 1
        if use_dp_ep:
            seq_len = hidden_states.size(1)
            lengths = [
                torch.zeros(1, dtype=torch.long, device=hidden_states.device) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(lengths, torch.tensor([seq_len], dtype=torch.long, device=hidden_states.device))
            seq_lens = [int(item.item()) for item in lengths]
            rank = dist.get_rank()
            start_pos = sum(seq_lens[:rank])
            end_pos = start_pos + seq_len
            hidden_states = torch.cat(DistHelper.gather_variable_shapes(hidden_states), dim=1)
        else:
            start_pos, end_pos = 0, hidden_states.size(1)

        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.use_latent_moe:
            flat = self.routed_expert_down_proj(flat)

        out = self._local_expert_forward(flat, topk_idx, topk_weight)
        if use_dp_ep:
            # Every rank owns a disjoint expert slice; the MoE output is their sum.
            dist.all_reduce(out)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                out = self.routed_expert_norm(out)
            out = self.routed_expert_up_proj(out)
        out = out.view(*orig_shape)

        if self.config.num_shared_experts is not None:
            out = out + self.shared_experts(identity)
        if use_dp_ep:
            out = out[:, start_pos:end_pos, :]
        return out


class KimiDecoderLayer(nn.Module):
    """Kimi-K3.1 transformer block (AttnRes residuals + KDA or NoPE-MLA/DSA)."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_linear_attn = is_kda_layer(config, layer_idx)
        if self.is_linear_attn:
            self.self_attn = KimiDeltaAttention(config, layer_idx=layer_idx)
        else:
            self.self_attn = KimiDSAAttention(config, layer_idx=layer_idx)

        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if is_dense_mlp(config, layer_idx):
            self.mlp = KimiMLP(config)
        else:
            self.block_sparse_moe = KimiSparseMoeBlock(config)

        self.use_attn_residuals = use_attn_residuals(config)
        if self.use_attn_residuals:
            self.attn_res_block_size = int(config.attn_res_block_size)
            self.self_attention_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def _mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "block_sparse_moe"):
            return self.block_sparse_moe(hidden_states)
        return self.mlp(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        block_residual: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_attn_residuals:
            residual = hidden_states
            hidden_states = self.self_attn(
                self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                padding_mask=padding_mask,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self._mlp(self.post_attention_layernorm(hidden_states))
            return residual + hidden_states, block_residual

        batch, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states
        flat = prefix_sum.reshape(-1, hidden_size)

        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                flat, block_residual, self.self_attention_res_proj, self.self_attention_res_norm
            ).view(batch, seq_len, hidden_size)
        if self.layer_idx % self.attn_res_block_size == 0:
            if block_residual is None:
                block_residual = flat.new_zeros(flat.shape[0], 0, hidden_size)
            block_residual = torch.cat([block_residual, flat.unsqueeze(1)], dim=1)
            prefix_sum = None

        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            padding_mask=padding_mask,
        )
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        hidden_states = _apply_attn_res(
            prefix_sum.reshape(-1, hidden_size), block_residual, self.mlp_res_proj, self.mlp_res_norm
        ).view(batch, seq_len, hidden_size)
        hidden_states = self._mlp(self.post_attention_layernorm(hidden_states))
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        return prefix_sum, block_residual


class KimiLinearModel(nn.Module):
    """Text backbone: embeddings, 93 decoder layers, final AttnRes and ``norm``."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [KimiDecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_attn_residuals = use_attn_residuals(config)
        if self.use_attn_residuals:
            self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    @staticmethod
    def build_causal_mask(
        seq_len: int,
        batch_size: int = 1,
        padding_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Boolean ``[B, 1, S, S]`` keep-mask folding causality and left padding."""
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        mask = causal.view(1, 1, seq_len, seq_len)
        if padding_mask is not None:
            pad = padding_mask.reshape(batch_size, 1, 1, seq_len).to(device=device, dtype=torch.bool)
            mask = mask & pad
        return mask.expand(batch_size, 1, seq_len, seq_len)

    @staticmethod
    def is_linear_attn_layer(config: Any, layer_idx: int) -> bool:
        return is_kda_layer(config, layer_idx)

    def empty_block_residual(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_attn_residuals:
            return None
        return hidden_states.new_zeros(hidden_states.shape[0] * hidden_states.shape[1], 0, hidden_states.shape[2])

    def run_decoder_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        block_residual: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one block with the mask flavour its attention class expects."""
        if self.is_linear_attn_layer(self.config, layer.layer_idx):
            return layer(hidden_states, padding_mask=padding_mask, block_residual=block_residual)
        return layer(hidden_states, attention_mask=attention_mask, block_residual=block_residual)

    def finalize(self, hidden_states: torch.Tensor, block_residual: Optional[torch.Tensor]) -> torch.Tensor:
        """Global AttnRes mix followed by the final RMSNorm (no ``lm_head``)."""
        if self.use_attn_residuals and block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
            ).view_as(hidden_states)
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, seq_len, _ = inputs_embeds.shape
        pad_mask = attention_mask
        causal_mask = self.build_causal_mask(seq_len, batch, pad_mask, inputs_embeds.device)
        hidden_states = inputs_embeds
        block_residual = self.empty_block_residual(hidden_states)
        for layer in self.layers:
            hidden_states, block_residual = self.run_decoder_layer(
                layer, hidden_states, block_residual, causal_mask, pad_mask
            )
        return self.finalize(hidden_states, block_residual)


class KimiLinearForCausalLM(nn.Module):
    """``language_model`` subtree: backbone + ``lm_head`` (untied)."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.model = KimiLinearModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        hidden_states = self.model(*args, **kwargs)
        return self.lm_head(hidden_states)
