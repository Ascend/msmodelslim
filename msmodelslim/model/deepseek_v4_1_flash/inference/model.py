# Adapted from https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
# pylint: disable=W0621,C0302,W0201,W0237,W0221,W0631,R0901,E1102,E1136
# The DeepSeek reference graph reuses module-level names (x/args) as locals in
# every forward, and pylint cannot resolve torch's dynamic tensor/module types.
import hashlib
import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from .engram import EngramLayout, NgramHashState
from .image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
from .kernel import (
    act_quant,
    fp4_act_quant,
    fp4_gemm,
    fp8_gemm,
    hc_split_sinkhorn,
    sparse_attn,
)
from .vision import Aligner, ViT
from torch import nn

from msmodelslim.utils.logging import get_logger

# Set once by Transformer.__init__; one model per process, so layers just read them.
#
# Tensor parallelism (embedding / attention projections / head / engram) is
# deliberately left OFF: the quantization pipeline shards only the routed
# experts, so ``world_size`` stays 1 and every ``Parallel*`` class below is a
# no-op that neither splits weights nor issues a collective.
world_size = 1
rank = 0

# Expert parallelism: how many ranks divide one layer's routed experts.
# ``Transformer.__init__`` fills these from the process group.
ep_size = 1
ep_rank = 0
default_dtype = torch.float8_e4m3fn  # storage dtype for Linear weights, from ModelArgs.dtype
fp8_block_size = 32  # one fp8 scale per 32x32 weight block / 32 activations
fp4_block_size = 32  # one fp4 scale per 32 elements along K
scale_fmt = "ue8m0"
scale_dtype = torch.float8_e8m0fnu


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch's default dtype, restoring it even if the body raises."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
    """Field names are exactly the config JSON keys. The defaults are a small model that
    `python model.py` can run, not the released shapes -- though the scale-independent
    values (norm_eps, score_func, hc_*, engram_*) do match it.
    """

    # runtime limits rather than model shape: they size the KV caches
    max_batch_size: int = 4
    max_seq_len: int = 4096
    temperature: float = 1
    dtype: Literal["bf16", "fp8"] = "fp8"
    expert_dtype: Literal["fp4"] | None = "fp4"
    vocab_size: int = 129280
    dim: int = 1024
    moe_inter_dim: int = 1024
    n_layers: int = 5
    n_mtp_layers: int = 1  # extra draft layers appended after the backbone, indices n_layers..
    n_heads: int = 16
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # attention: latent q/kv projections, plus a LoRA-factorised output projection over o_groups
    q_lora_rank: int = 256
    head_dim: int = 128
    rope_head_dim: int = 32
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 256
    # sparse attention: every layer attends over a sliding window, and may add compressed KV on top
    window_size: int = 128
    # one entry per layer, MTP layers included: 0 = sliding window only, r = KV compressed r-to-1
    compress_ratios: tuple[int, ...] = (0, 2, 2, 1, 1, 0)
    # layers sharing a ratio also share one compressed KV and one indexer, produced by the first
    kv_source_layers: tuple[int, ...] = (1, 3)
    index_source_layers: tuple[int, ...] = (1, 3)
    # rope, with YaRN extrapolation when original_seq_len > 0. Compressed KV rotates at its own
    # theta because one latent stands for compress_ratio tokens, so its positions are further apart.
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # the indexer: a small extra attention that scores compressed positions, so each query can keep
    # just `index_topk` of them. Names match DeepSeek-V3.2-Exp, where this mechanism first appeared.
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 64
    # candidate pre-filtering: candidate_source_layer < 0 turns it off and the other two are unused
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # hyper-connections: the residual stream is carried as hc_mult parallel copies
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # engram: n-gram hash lookups added into the residual stream at a few layers
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()  # table rows per engram layer
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0  # bucket size each (n-gram size, head) starts searching primes from
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2  # token that fills n-gram slots with no history; matches training
    # size of the compressed tokenizer vocab; every hash multiplier is derived from it
    engram_compressed_vocab_size: int = 0
    # vision (VL); vision_n_layers == 0 disables the vision path
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 544 * 544
    vision_max_wh_ratio: int | None = None
    # raw id of <｜deepseek_image｜>; every position of an image span carries this id in input_ids
    image_token_id: int = 129264
    # dspark draft head. The draft is reached through the same forward entry
    # (pass ``main_hidden``) and driven by ``generate``; these shape it.
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 0
    dspark_n_activated_experts: int = 0

    @property
    def vision_enabled(self) -> bool:
        return self.vision_n_layers > 0

    def get_moe_config(self, layer_id: int) -> tuple[int, int]:
        """Return the routed/activated expert counts for a given layer."""
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_n_activated_experts or self.n_activated_experts,
        )


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each rank holds vocab_size // world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, (  # nosec
            f"Vocabulary size must be divisible by world size (world_size={world_size})"
        )
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            # ids off this rank read row 0 then get zeroed, so the all_reduce sums one real row
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        # The weight may be a CPU-resident shared leaf (the adapter's
        # get_offload_target_map keeps ``*.embed`` off the runner device) while
        # the ids come from that device -- F.embedding needs them together.  The
        # result goes back to the caller's device, which is where the rest of
        # the graph runs.
        device = x.device
        y = F.embedding(x.to(self.weight.device), self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y.to(device)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Pick a GEMM from the weight dtype. Quantized weights need a quantized activation, and both
    fp4 and fp8 weights take an fp8 one -- for fp4 the kernel handles the mixed precision.
    """
    assert bias is None  # nosec

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, fp8_block_size, scale_fmt, scale_dtype)
        return fp4_gemm(
            x,
            s,
            weight,
            weight.scale,
            scale_dtype,
            act_block_size=fp8_block_size,
        )
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, fp8_block_size, scale_fmt, scale_dtype)
        return fp8_gemm(
            x,
            s,
            weight,
            weight.scale,
            scale_dtype,
            block_size=fp8_block_size,
        )
    else:
        return F.linear(x, weight)


class Linear(nn.Linear):
    """bf16, fp8 or fp4 weights. Quantized ones get a `scale`, also attached to `.weight` so that
    `linear()` can reach it.

    Subclasses ``torch.nn.Linear`` so the standard ModelSlim quantizers
    (LinearQuantizer / deployed quant modules) can address these weights: the
    decoded-BF16 calibration path therefore “just works” as an nn.Linear, while
    the FP8 / packed-FP4 storage keeps the native ``linear()`` dispatch.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        dtype = dtype or default_dtype
        # nn.Linear accepts only floating dtypes; keep fp8/packed construction native.
        base_dtype = dtype if dtype in (torch.bfloat16, torch.float16, torch.float32, None) else None
        super().__init__(in_features, out_features, bias=bias, dtype=base_dtype)
        self.in_features = in_features
        self.out_features = out_features
        if dtype == torch.float4_e2m1fn_x2:
            # two values per byte: [out, in] logically, [out, in//2] stored
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(out_features, in_features // fp4_block_size, dtype=torch.float8_e8m0fnu)
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(
                    (out_features + fp8_block_size - 1) // fp8_block_size,
                    (in_features + fp8_block_size - 1) // fp8_block_size,
                    dtype=torch.float8_e8m0fnu,
                )
            )
        else:
            if dtype is not None and self.weight.dtype != dtype:
                self.weight.data = self.weight.data.to(dtype)
            self.register_parameter("scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Splits the output dim across ranks; each rank's slice of the output is already complete."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert out_features % world_size == 0, (  # nosec
            f"Output features must be divisible by world size (world_size={world_size})"
        )
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Splits the reduction dim, so each rank holds a partial sum: hence the fp32 all_reduce, with
    the bias added only after it.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert in_features % world_size == 0, (  # nosec
            f"Input features must be divisible by world size (world_size={world_size})"
        )
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class ParallelEngramEmbedding(nn.Module):
    """The n-gram hash table, sharded over its rows. Stays fp8: rows are dequantized on lookup."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        assert num_embeddings % world_size == 0, (  # nosec
            f"Engram table size {num_embeddings} must be divisible by world_size={world_size}"
        )
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.part_num_embeddings = num_embeddings // world_size
        self.vocab_start_idx = rank * self.part_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.part_num_embeddings

        self.block_size = fp8_block_size
        # the table stays fp8 as stored: rows are dequantized with `scale` on lookup
        self.weight = nn.Parameter(torch.empty(self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn))
        self.scale = nn.Parameter(torch.empty(self.part_num_embeddings, dim // self.block_size, dtype=scale_dtype))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
        local_indices = indices - self.vocab_start_idx
        local_indices = local_indices.masked_fill(mask, 0)

        values = F.embedding(local_indices, self.weight)
        scales = F.embedding(local_indices, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        values = values.masked_fill(mask.unsqueeze(-1), 0)

        if world_size > 1:
            dist.all_reduce(values)
        return values


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared
    value. The gate is a normalized dot product of stream against key.
    """

    def __init__(self, args: ModelArgs, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6

        self.embed = ParallelEngramEmbedding(layout.num_embeddings[self.layer_hash_index], layout.head_dim)
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = Linear(n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1))
        self.eps = args.norm_eps
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def _apply(self, fn, recurse: bool = True):
        """Keep the float projections CPU-resident across device moves.

        The layer-wise runner never yields the engram module itself, yet a
        yielded block's blanket ``module.to(offload_device)`` recurses into this
        submodule and would drop ``wkv`` to meta.  ``wkv`` holds the
        QuaRot-rotated projection the export reads at save time, so it must
        survive; the forward restores a device copy on demand (``_wkv_cpu`` /
        ``_qk_cpu``).
        """
        for key, param in self._parameters.items():
            if param is not None:
                self._parameters[key] = fn(param)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        if recurse:
            for key, child in self._modules.items():
                if key == 'wkv':
                    continue  # keep the rotated projection on CPU
                self._modules[key] = child._apply(fn, recurse=True)
        return self

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        gate_basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L], False shuts
        the gate so those positions pass through untouched.

        Layer-wise staging notes: the giant table (source FP8 / exported I8,
        ~98GB) always lives on CPU — the lookup runs on the CPU copy of the
        ids and ships only the hit rows to the stream device. wkv / q / k may
        sit on meta after the decoder post-hook offloaded the whole block
        (between samples); CPU copies taken on first sight restore them.
        """
        if hash_ids.device != x.device:
            hash_ids = hash_ids.to(x.device)
        # ---- table lookup on CPU; rows to the stream device ----
        embed = self.embed(hash_ids.cpu()).to(x.device)
        # ---- wkv with meta-offload CPU cache ----
        wkv = self.wkv
        cached_w = getattr(self, '_wkv_cpu', None)
        if cached_w is None and not wkv.weight.is_meta:
            cached_w = wkv.weight.detach().to('cpu').clone()
            self._wkv_cpu = cached_w
            self._qk_cpu = {
                'q_weight': (self.q_weight.detach().to('cpu').clone() if not self.q_weight.is_meta else None),
                'k_weight': (self.k_weight.detach().to('cpu').clone() if not self.k_weight.is_meta else None),
            }
        if wkv.weight.is_meta or wkv.weight.device != x.device:
            if cached_w is None:
                raise RuntimeError('engram wkv is meta and no CPU cache exists')
            wkv = torch.nn.Linear(wkv.in_features, wkv.out_features, dtype=cached_w.dtype, device=x.device)
            with torch.no_grad():
                wkv.weight.copy_(cached_w.to(x.device))
        kv = wkv(embed.flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))

        def _small(name):
            p = getattr(self, name)
            if p.is_meta:
                cache = getattr(self, '_qk_cpu', None)
                if cache and cache.get(name) is not None:
                    return cache[name].to(x.device)
                raise RuntimeError(f'engram {name} is meta and no CPU cache exists')
            return p if p.device == x.device else p.to(x.device)

        weight = _small('q_weight').float() * _small('k_weight').float()
        h, eps = x.float(), self.eps
        # QuaRot cameo: with rotation on, the residual HS arriving here (x) is in
        # the rotated ambient basis.  The exported gate (q/k) and the wkv VALUE
        # block are written for that basis -- the value's right rotation is fused
        # into the exported weights, so it is added straight back (weighted by the
        # gate).  Only the gate's dot product is defined against the ORIGINAL
        # basis, so undo the HS rotation for it (inverse right-rotate: x @ Q.T);
        # the RETURN reuses the actual x.  The caller owns the switch: it passes
        # ``gate_basis`` exactly when the checkpoint was rotated.
        if gate_basis is not None:
            h = h @ gate_basis.to(h.device, h.dtype).T
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (x.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, one row per position.

    With original_seq_len > 0 this applies YaRN: dimensions whose wavelength already fits inside the
    training context keep their frequency, those far beyond it are divided by `factor`, and the
    `beta_fast`..`beta_slow` band in between is faded across with a linear ramp.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        # the dim whose wavelength completes `rotations` turns over the training context
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate `x` in place, taking adjacent element pairs as complex numbers. Accepts [b, s, d] and
    [b, s, h, d]; `inverse` conjugates the rotation, which is how the attention output gets the
    query's rotation removed again so the cache can stay in one shared rotated form.
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


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Which sliding-window cache slots each query attends to; -1 marks a slot holding nothing.

    The cache is a ring of `window_size` slots. Prefill needs one row per query, each seeing its own
    causal window. A decode step has a single query that sees the whole ring, listed oldest first.
    Order within a row does not matter to `sparse_attn`, which handles every slot independently.
    """
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)  # ring still filling
    # sparse_attn needs real [b, m, topk] int32 memory, hence the materializing expand
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate.

    Returns the latent before RoPE, or None while a group is still filling up -- so during decode it
    only yields every `compress_ratio` steps, holding the partial group in `kv_state`/`score_state`.
    Pre-RoPE is deliberate: the indexer needs the unrotated form, so Attention rotates afterwards.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        compress_ratio = args.compress_ratios[layer_id]
        head_dim = args.head_dim
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, args.norm_eps)
        # ratio 1 is a plain projection, so it stays in the checkpoint's bf16; the softmax pooling
        # above ratio 1 runs in fp32, so those weights are promoted to fp32 to match
        self.wkv = Linear(args.dim, head_dim, dtype=torch.float32 if compress_ratio > 1 else torch.bfloat16)
        if compress_ratio > 1:
            self.wgate = Linear(args.dim, head_dim, dtype=torch.float32)
            # tail of an incomplete group, carried across decode steps
            self.kv_state: torch.Tensor
            self.score_state: torch.Tensor
            state_shape = (args.max_batch_size, compress_ratio, head_dim)
            self.register_buffer("kv_state", torch.zeros(state_shape, dtype=torch.float32), persistent=False)
            self.register_buffer(
                "score_state", torch.full(state_shape, -torch.inf, dtype=torch.float32), persistent=False
            )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                kv, self.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
                score, self.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # one token per step: fill a slot, and pool only when the group just completed
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))


class Indexer(torch.nn.Module):
    """Keeps the `index_topk` best compressed positions per query.

    A small side attention: fp4 query heads against one shared key per compressed position, scores
    rectified then combined by `weights_proj`. With a candidate source this is the second of two
    levels; `select_candidate_blocks` is the first.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        # the index keys are derived from the compressor's latent, so only a layer that compresses
        # its own KV can produce them; every other indexer reads them from that layer's cache
        self.owns_k = layer_id in args.kv_source_layers
        self.compress_ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.index_head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.freqs_cis: torch.Tensor | None = None
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.index_head_dim, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)
            self.k_cache: torch.Tensor
            self.register_buffer(
                "k_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len // self.compress_ratio, args.index_head_dim),
                persistent=False,
            )

    def forward(self, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor, start_pos: int, offset: int):
        """`latent` is this layer's RoPE-free compressed latent, None when this layer does not
        compress or when its current group is still incomplete. An index-key owner turns it into
        index keys here, which has to happen before Attention overwrites that same storage with
        the RoPE'd, quantized values.
        """
        assert self.freqs_cis is not None  # nosec
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        # latent is None while a group is still filling up, so there is nothing to publish yet
        if self.owns_k and latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = self.k_norm(self.wk(latent))
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant(k, fp4_block_size, True)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
            shared_attn.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        fp4_act_quant(q, fp4_block_size, True)

        index_k = shared_attn.index_k[:bsz, : end_pos // ratio]
        if index_k.device != x.device:
            index_k = index_k.to(x.device)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1:
            dist.all_reduce(index_score)

        # how many compressed positions each query can see: a block becomes visible once the query
        # has passed its last token. One query per decode step, so there it is just a number.
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            shared_attn.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            # level two: score with our own weights, but only inside the source's candidate blocks
            index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)

        # top-k by score, re-sorted into position order; unreachable -> -1, rest shifted by offset
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query.

    `logits` is [..., n_positions] with positions the query cannot reach already at -inf, which is
    what makes a block score of -inf mean "not reachable yet". `compress_lens` is a plain int during
    decode, or broadcasts against logits' leading dims during prefill. Returns a bool mask shaped
    like `logits`, so the layers consuming it just mask and never think about blocks again.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so pin it in: it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Attention(nn.Module):
    """Latent attention over two KV sources at once, concatenated into one `sparse_attn` call: a
    sliding window of raw KV, plus -- when compress_ratio > 0 -- `index_topk` compressed positions
    reaching further back. Q and the output projection are both low-rank, the latter grouped.

    compress_ratio > 0 does not mean the layer compresses its own KV: only kv_source_layers do,
    the rest read that same cache.
    """

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
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        is_backbone = layer_id < args.n_layers
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor: Compressor | None = None
        self.indexer: Indexer | None = None
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)

        self.window_kv_cache: torch.Tensor
        self.register_buffer(
            "window_kv_cache",
            torch.zeros(args.max_batch_size, args.window_size, self.head_dim),
            persistent=False,
        )
        if self.is_kv_source:
            self.compress_kv_cache: torch.Tensor
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len // self.compress_ratio,
                    self.head_dim,
                ),
                persistent=False,
            )
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.freqs_cis: torch.Tensor
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _window_kv(self, x, freqs_cis, start_pos):
        """This layer's sliding-window K and the window positions every query may attend to. The K
        stays fp8, quantized over the whole post-RoPE vector, RoPE tail included.
        """
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)
        if start_pos == 0:  # prefill: attend over this chunk, seeding the ring buffer for decode
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            window_kv = kv
        else:  # decode: one token into the ring buffer, attend over the whole window
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
            if window_kv.device != x.device:
                window_kv = window_kv.to(x.device)
        topk = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if topk.device != x.device:
            topk = topk.to(x.device)
        return window_kv, topk

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_len):
        """Which compressed positions each query attends to. Index sources run their own indexer;
        the layers in between reuse the result their source published.
        """
        if not self.is_index_source:
            # Cross-layer state may live on a different device than this layer's
            # input (layer-wise staging moves the layer, not shared_attn).
            idxs = shared_attn.topk_idxs
            return idxs.to(x.device) if idxs is not None and idxs.device != x.device else idxs

        bsz, seqlen, _ = x.size()
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None  # nosec
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            idxs = self.indexer(x, qr, latent, start_pos, offset)
        shared_attn.topk_idxs = idxs
        return idxs

    def _compress_kv(self, x, qr, start_pos, offset):
        """The shared compressed KV and the compressed positions every query may attend to. This
        layer compresses its own KV only when it is a source; otherwise it just reads the cache.
        """
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            shared_attn.compress_kv = self.compress_kv_cache
        # the indexer needs the latent before RoPE, so it runs before the cache is written
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
        if latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs)
            # the compressed KV is fp4, like the indexer: quantize to the fp4 grid and cast back
            fp4_act_quant(latent, fp4_block_size, True)
            self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
        # read after the write, so this does not depend on the slice aliasing the cache
        compress_kv = shared_attn.compress_kv[:bsz, :compress_len]
        if compress_kv.device != x.device:
            compress_kv = compress_kv.to(x.device)
        return compress_kv, idxs

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor | None = None):
        # Switch to this batch's shared slot: under the layer-major calibration
        # schedule the source ran for every batch before this consumer runs.
        shared_attn.begin(input_ids, start_pos)
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # wo_a is block-diagonal over groups (each projects only its own heads), hence einsum not
        # Linear. convert.py dequantizes it to bf16; an fp8 grouped GEMM would halve the memory.
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """MoE gating. The correction bias steers expert selection only; the routing weights come from the
    unbiased scores. Image-span tokens use a separate bias (training `noaux_tc_for_vl`).
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.dim = args.dim
        self.topk = n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
        self.bias_vl = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32)) if args.vision_enabled else None

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [n, dim]; image_mask: [n] bool, True for tokens inside an image span."""
        scores = linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # the bias picks experts but does not scale them: weights come from the raw scores
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20  # not norm_eps, matches training
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """One SwiGLU FFN. The clamps come straight from training, where they keep fp8/fp4 activations in
    range: the up branch is clamped on both sides, the gate branch only from above.
    """

    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0.0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through. Experts are split
    across ranks, so `self.experts` is None for those another rank owns.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.layer_id = layer_id
        self.dim = args.dim
        assert n_routed_experts % ep_size == 0, f"Number of experts must be divisible by ep_size={ep_size}"  # nosec
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // ep_size
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = ep_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self.experts = nn.ModuleList(
            [
                Expert(
                    args.dim,
                    args.moe_inter_dim,
                    dtype=expert_dtype,
                    swiglu_limit=args.swiglu_limit,
                )
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        assert args.n_shared_experts == 1  # nosec
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        mask = None if image_mask is None else image_mask.flatten()
        # Expert parallelism splits the experts but NOT the data: the runner still
        # shards the calibration set across ranks, so two ranks hold different
        # token counts and an all_reduce over the local expert outputs would sum
        # mismatched shapes -- HCCL refuses it ("the arguments for collective
        # communication are inconsistent between ranks").  So gather every rank's
        # tokens, compute this rank's experts over the union, reduce, and keep
        # only this rank's slice.  Same shape as kimi_k3's ep_forward.
        use_ep = ep_size > 1 and dist.is_initialized() and dist.get_world_size() > 1
        if use_ep:
            from msmodelslim.utils.distributed import DistHelper

            mine = torch.tensor([x.size(0)], dtype=torch.long, device=x.device)
            gathered = [torch.zeros_like(mine) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, mine)
            sizes = [int(s.item()) for s in gathered]
            start = sum(sizes[: dist.get_rank()])
            end = start + sizes[dist.get_rank()]
            x = torch.cat(DistHelper.gather_variable_shapes(x), dim=0)
            if mask is not None:
                mask = torch.cat(DistHelper.gather_variable_shapes(mask), dim=0)
        weights, indices = self.gate(x, mask)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        if use_ep:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        if use_ep:
            y = y[start:end]
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """A block whose residual stream is `hc_mult` parallel copies (Hyper-Connections).

    Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer input) and
    `hc_post` (expand back out, mixing the residual in through `comb`). `hc_mixes` derives all three
    coefficient sets from the stream itself, `comb` made doubly stochastic by Sinkhorn.

    The coefficients a sublayer computes are used by the *next* one -- see `forward`.
    """

    attention_cls = Attention

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = self.attention_cls(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(args, layer_id, engram_layout)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc]. Returns the
        pre / post / comb coefficients, split out of one projection of the flattened stream.
        """
        # normalized over the whole flattened hc*d stream, one statistic per token
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
        """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
        x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]
        """
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        *attn_args,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`pre_mix` collapses the hc_mult copies down to one input for this block's attention. Each
        sub-block's own `hc_mixes` produces the mix for the *next* one, so attention uses what the
        previous layer's FFN produced and the FFN uses what this attention produced.

        image_mask: [b, s] bool, True inside image spans (selects the VL routing bias).
        """
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, *attn_args, input_ids=input_ids)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x, image_mask)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = vocab_size // world_size
        # bf16 in the checkpoint, kept as fp32 here so the logits come out in fp32 directly
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, full_logits=False):
        """x: [b, s, d]. Generation only needs the last position, so that is the default."""
        if not full_logits:
            x = x[:, -1]
        device = x.device
        x = x.float()
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        # Same CPU-resident-leaf case as ParallelEmbedding: `*.head` is pinned to
        # CPU, so bring the activations to the weight and return on the caller's
        # device.
        logits = F.linear(x.to(weight.device), weight).to(device)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


@lru_cache(1)
def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int):
    assert start_pos > 0  # nosec
    matrix = torch.cat(
        [
            torch.arange(min(window_size, start_pos + 1)),
            window_size + torch.arange(block_size),
        ]
    )
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(Attention):
    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor, **kwargs):
        # ``compress_ratio == 0``: this attention reads no shared slot, and the
        # draft walk passes input_ids through the block's **kwargs.
        assert self.compress_ratio == 0  # nosec
        bsz, seqlen, _ = main_x.size()
        win = self.window_size
        rd = self.rope_head_dim

        if start_pos == 0:
            main_kv = self.kv_norm(self.wkv(main_x))
            apply_rotary_emb(main_kv[..., -rd:], self.freqs_cis[0:seqlen])
            act_quant(main_kv, fp8_block_size, scale_fmt, scale_dtype, True)
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = main_kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            return x

        # A whole-prefix `main_x` ([B, S, D] with S > 1, the shape the eval's
        # re-prefill hands down): the runtime buffer store rebuilds non-persistent
        # buffers per decoder layer, so the window cache cannot be carried over
        # from a separate seed call.  Roll positions 0..S-2 in here, then decode
        # position S-1 -- what serving gets from (seed @0) + (decode @i), in one
        # call.  S == 1 keeps the serving path unchanged.
        if seqlen > 1:
            prefix = seqlen - 1
            prefix_kv = self.kv_norm(self.wkv(main_x[:, :prefix]))
            apply_rotary_emb(prefix_kv[..., -rd:], self.freqs_cis[0:prefix])
            act_quant(prefix_kv, fp8_block_size, scale_fmt, scale_dtype, True)
            if prefix <= win:
                self.window_kv_cache[:bsz, :prefix] = prefix_kv
            else:
                cutoff = prefix % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = prefix_kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            main_x = main_x[:, -1:]
            seqlen = 1

        bsz, block_size, _ = x.size()
        main_freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)
        act_quant(main_kv, fp8_block_size, scale_fmt, scale_dtype, True)

        freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)

        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos)
        if topk_idxs.device != q.device:
            topk_idxs = topk_idxs.to(q.device)
        self.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)
        window_kv = self.window_kv_cache[:bsz]
        if window_kv.device != q.device:
            window_kv = window_kv.to(q.device)
        kv = torch.cat([window_kv, kv], dim=1)
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, dspark_markov_rank: int):
        super().__init__()
        self.embed = ParallelEmbedding(vocab_size, dspark_markov_rank)
        self.head = ParallelHead(vocab_size, dspark_markov_rank)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        logits = self.head(embed, full_logits=True)
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # proj in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for fp32 confidence score.
        self.proj = Linear(input_dim, 1, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)


class DSparkBlock(Block):
    """DSpark stage stored under the mtp.* checkpoint namespace."""

    attention_cls = DSparkAttention

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        stage_id = layer_id - args.n_layers
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.temperature = args.temperature
        if stage_id == 0:
            assert len(args.dspark_target_layer_ids) > 0, "DSpark needs target layers"  # nosec
            self.main_proj = Linear(args.dim * len(args.dspark_target_layer_ids), args.dim)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)
        self.embed: ParallelEmbedding | None = None
        self.head: ParallelHead | None = None

    def forward(
        self,
        x: torch.Tensor = None,
        start_pos: int = 0,
        pre_mix: torch.Tensor = None,
        main_x: torch.Tensor = None,
        *,
        phase: str = 'layer',
        main_hidden: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        greedy: bool = False,
    ):
        """`phase` routes the draft entry/exit through this call as well.

        ``forward_embed`` / ``forward_head`` are plain methods on the block, so
        calling them directly runs before any per-block weight hook has fired --
        in layer-wise serving that means their parameters are still un-hydrated.
        Going through ``__call__`` puts them on the same hook path as the draft
        loop, which is what lets an engine manage these blocks like any other
        decoder layer.
        """
        if phase == 'embed':
            return self.forward_embed(main_hidden, input_ids)
        if phase == 'head':
            return self.forward_head(x, pre_mix, input_ids, greedy=greedy)
        if start_pos == 0:
            self.attn(x, start_pos, main_x)  # prefill only seeds the window KV cache
            return x, pre_mix
        return super().forward(x, start_pos, pre_mix, None, main_x)  # drafts are text: no VL bias

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        assert self.embed is not None  # nosec
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)
        draft_input_ids[:, 0] = input_ids
        x = self.embed(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1).to(main_x.device)
        return x, main_x

    def forward_head(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        input_ids: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.head is not None  # nosec
        x = self.hc_pre(x, pre_mix)
        # `head` is the draft's OWN pre-rotation copy (see _separate_draft_leaves),
        # so neither the QuaRot basis nor the backbone's folded final-norm diagonal
        # applies: this is the reference implementation's line verbatim.
        logits = self.head(self.norm(x), full_logits=True)
        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, i])
            logits[:, i].add_(logits_bias)
            markov_embeds.append(markov_embed)
            output_ids[:, i + 1] = logits[:, i].argmax(-1) if greedy else sample(logits[:, i], self.temperature)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(x, markov_embed)
        return output_ids, logits, confidence


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """initial one-hot mix"""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing.

    One slot per state is not enough here.  The layer-wise runner is
    layer-major over the whole dataset: a source layer runs for **every batch**
    before its consumers run for any of them, so a consumer that reads a single
    slot gets whichever batch the source happened to write last -- and the topk
    and KV lengths do not match (measured: 105 vs 71, aclnnCat 161002).  The
    slots are therefore keyed by ``(start_pos, hash of input_ids)``, which
    identifies the batch regardless of dataset order or how many times a
    processor walks it.

    Sources: compress_kv and index_k from kv_source_layers, topk_idxs from
    index_source_layers, candidates from candidate_source_layer.
    """

    _CAP = 512

    def __init__(self):
        self._store: "OrderedDict" = OrderedDict()
        self._key = None

    def reset(self) -> None:
        self._store = OrderedDict()
        self._key = None

    @staticmethod
    def _hash_ids(input_ids) -> str:
        if input_ids is None:
            return "none"
        t = input_ids.detach().to("cpu").contiguous()
        return hashlib.md5(t.numpy().tobytes()).hexdigest()  # nosec B324  (content key, not security)

    def begin(self, input_ids, start_pos: int) -> None:
        """Switch to this batch's slot; call it at the top of a layer forward."""
        self._key = (int(start_pos), self._hash_ids(input_ids))

    def _slot(self, create: bool = False):
        if create:
            slot = self._store.get(self._key)
            if slot is None:
                self._store[self._key] = slot = {}
                while len(self._store) > self._CAP:
                    self._store.popitem(last=False)
            else:
                self._store.move_to_end(self._key)
            return slot
        return self._store.get(self._key)

    def publish(self, name: str, value) -> None:
        self._slot(create=True)[name] = value

    def fetch(self, name: str):
        slot = self._slot()
        return None if slot is None else slot.get(name)

    # The callers keep reading and writing plain attributes; the properties keep
    # that spelling and route it through the per-batch slot.
    @property
    def compress_kv(self):
        return self.fetch("compress_kv")

    @compress_kv.setter
    def compress_kv(self, value):
        self.publish("compress_kv", value)

    @property
    def index_k(self):
        return self.fetch("index_k")

    @index_k.setter
    def index_k(self, value):
        self.publish("index_k", value)

    @property
    def topk_idxs(self):
        return self.fetch("topk_idxs")

    @topk_idxs.setter
    def topk_idxs(self, value):
        self.publish("topk_idxs", value)

    @property
    def candidates(self):
        return self.fetch("candidates")

    @candidates.setter
    def candidates(self, value):
        self.publish("candidates", value)


# Only ever one model per process, same as world_size / rank / default_dtype above.
shared_attn = SharedAttentionRuntime()


class RotationScheme(Enum):
    """Which sub-graphs the rotation reaches.

    Rotation is a family of schemes, not one switch: the backbone and the draft
    can each be rotated or not, and every combination wires the boundary between
    them differently.  Only the combinations that are meaningful are listed.

    ``NONE``
        Nothing is rotated.  The draft keeps sharing the backbone's embed and
        head, exactly as the reference implementation has it.

    ``BACKBONE_ONLY``  (the default, and what this pipeline currently runs)
        The layer stack is rotated; the draft stays in the ORIGINAL basis.  Only
        ``mtp.0.main_proj`` is rotated, and only to cancel the ``main_hidden``
        the backbone hands down.  Two consequences at the boundary:
          * the Engram gate reads the rotated residual stream but defines its
            q/k against the original basis, so it inverse-rotates first;
          * the draft needs its own embed -- the shared one now lives in the
            rotated basis.

    ``BACKBONE_AND_DRAFT``
        Both rotated with the same matrix.  The stream crossing disappears, so
        no gate inverse-rotation, no ``main_proj`` bridge, and the draft could
        keep the shared embed.

    Why the draft never shares the HEAD
    -----------------------------------
    ``get_ln_fuse_map`` folds the backbone's final norm into the head
    (``{'norm': ['head']}``: ``head -> (head * g) @ Q``, ``norm.weight -> 1``).
    A head can carry exactly one norm's diagonal, and the draft's final norm is
    a *different* norm -- with the backbone's folded in, the draft would compute
    ``head(norm_draft / g)``.  So the draft takes its own head in every rotated
    scheme; only ``NONE``  (nothing folded) can share it.
    """

    NONE = "none"
    BACKBONE_ONLY = "backbone_only"
    BACKBONE_AND_DRAFT = "backbone_and_draft"


@dataclass(frozen=True)
class RotationPlan:
    """The rotation scheme in force, plus the matrix it used.

    Set by the adapter -- once when the ``quarot`` processor runs, once when a
    loaded export carries ``optional/quarot.safetensors`` -- so both entry
    points agree on what was rotated.  Not user-configurable: which scheme is
    valid is a property of the export, not a preference.
    """

    scheme: RotationScheme = RotationScheme.BACKBONE_ONLY
    #: The global rotation over the hidden size, when a crossing needs it.
    global_rotation: torch.Tensor | None = None

    @property
    def backbone_rotated(self) -> bool:
        return self.scheme is not RotationScheme.NONE

    @property
    def engram_gate_basis(self) -> torch.Tensor | None:
        """Basis to inverse-rotate by before the Engram gate; ``None`` = no-op."""
        return self.global_rotation if self.backbone_rotated else None


class Transformer(nn.Module):
    """DeepSeek-V4.1: embed -> expand to hc_mult copies -> blocks -> collapse -> logits. Building
    this sets the globals at the top of the file. The tokenizer only feeds the engram token map.
    """

    def __init__(self, args: ModelArgs, tokenizer=None):
        global world_size, rank, ep_size, ep_rank, default_dtype
        # TP stays off (see the module-level note): only the routed experts are
        # split, so the export keeps its single-card layout.
        world_size, rank = 1, 0
        if dist.is_initialized():
            ep_size, ep_rank = dist.get_world_size(), dist.get_rank()
        else:
            ep_size, ep_rank = 1, 0
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.temperature = args.temperature
        #: Rotation bases this model's weights were built with, keyed by name
        #: (``ROTATION_GLOBAL`` for the hidden-size one).  Rotation is a family of
        #: schemes, not one matrix: the adapter already emits a Hadamard over the
        #: hidden size *and* a block-Hadamard over ``q_lora_rank``, and a future
        #: recipe may add more, rotate only part of the graph, or pick a different
        #: basis per sub-module.  So consumers ask for the name they need, and an
        #: empty mapping means "nothing was rotated".
        self.rotation_plan = RotationPlan()
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.engram_layout = EngramLayout.from_args(args)
        self.engram_hash = (
            NgramHashState(args, self.engram_layout, tokenizer) if self.engram_layout is not None else None
        )
        #: Kept so log lines can decode what was generated -- an id alone says
        #: nothing about which token it is.  None when the shell is built without
        #: one (the structural fallback and the ``__main__`` demo).
        self.tokenizer = tokenizer
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args, self.engram_layout))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = torch.nn.ModuleList()
        self.target_layer_ids = args.dspark_target_layer_ids
        if args.dspark_block_size:
            for layer_id in range(args.n_mtp_layers):
                self.mtp.append(DSparkBlock(args.n_layers + layer_id, args))
                self.mtp[-1].embed = self.embed
                self.mtp[-1].head = self.head
        self.hc_mult = args.hc_mult
        #: Whether :meth:`generate` also runs the DSpark draft.  Off unless the
        #: loader turns it on: the draft blocks sit outside the layer-wise
        #: lifecycle the engine manages, so running them is a deliberate choice,
        #: and the adapter owns that switch (its ``DSPARK`` module constant).
        self.dspark = False
        self.vision = None
        if args.vision_enabled:
            self.vision = ViT(args)
            self.aligner = Aligner(args)
            # learned embeddings for the image span delimiters
            self.image_start = nn.Parameter(torch.empty(args.dim))
            self.image_end = nn.Parameter(torch.empty(args.dim))
            self.image_newline = nn.Parameter(torch.empty(args.dim))

    @torch.inference_mode()
    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h: torch.Tensor):
        """Overwrite each image's token span in h with its ViT/aligner features. The IMAGE slots take
        the aligner rows in row-major order; the span delimiters take learned embeddings.
        """
        for i, sample in enumerate(images):
            for img in sample or ():
                types = img.types.to(h.device)
                span = h[i, img.start : img.start + types.numel()]
                span[types == IMAGE_START] = self.image_start.to(h.dtype)
                span[types == IMAGE_END] = self.image_end.to(h.dtype)
                span[types == IMAGE_NEW_LINE] = self.image_newline.to(h.dtype)
                embeds = self.encode_image(img.patches.to(h.device), img.n_vit_h, img.n_vit_w)
                span[types == IMAGE] = embeds.to(h.dtype)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        images=None,
        token_types: torch.Tensor | None = None,
        main_hidden: torch.Tensor | None = None,
        greedy: bool = False,
        *,
        rotate: bool | None = None,
        engram: bool | None = None,
        dspark: bool = False,
        full_logits: bool = False,
    ):
        """The model's forward, in the two shapes a caller needs.

        ``main_hidden is None`` -- **backbone**: embed -> blocks -> collapse ->
        head, returning ``(output_ids, logits, main_hidden)``.  ``input_ids`` is
        [b, s], every entry a real token id -- generate.py only ever passes
        positions it has already filled, so the padding it uses internally never
        reaches here.  ``token_types`` / ``images`` carry the VL inputs built by
        image_processor.prepare_vl_inputs; image spans must lie inside the first
        (start_pos 0) chunk.

        ``main_hidden is not None`` -- **DSpark draft**: ``input_ids`` are the
        backbone's tokens for this step and ``main_hidden`` the hidden it handed
        down.  ``start_pos == 0`` is the prefill call, which only seeds the draft
        window KV cache and returns ``None``; later steps return ``(output_ids,
        logits, confidence)``.

        Three switches, each independent of the shape:

        ``rotate``
            Whether the rotated basis is in force, overriding
            ``self.rotation_plan`` for this call.  ``None`` (the default) takes
            the plan the adapter set from the export.  The one place the basis
            enters the forward is the Engram gate, which is defined in the
            original basis and so inverse-rotates the stream first.

        ``engram``
            Whether to inject Engram contributions.  ``None`` (the default)
            injects whenever the model carries Engram tables -- the checkpoint's
            own layout decides.  ``False`` skips both the hashing and every
            per-layer gate, so the ~200 GiB tables are never read.

        ``dspark``
            Backbone shape only: walk the draft prefill in the same call, so a
            caller that wants the whole model reaches both halves through one
            entry.  A calibration schedule built from the backbone alone would
            leave every ``mtp.*`` linear unquantized.  The draft blocks are
            ordinary pause points either way.  Not combinable with
            ``full_logits`` -- that mode has no sampled token to seed the draft
            with.

        ``full_logits``
            Return per-position logits ``[b, s, vocab]`` instead of the last
            position's, and skip sampling: the caller is scoring positions it
            already knows (see :func:`generate`, which uses this to check a
            draft proposal against the backbone in the forward it needs anyway).
        """
        if main_hidden is not None:
            return self._draft(input_ids, main_hidden, start_pos, greedy)
        # `None` means "follow the checkpoint's layout"; an explicit True has
        # only those same tables to inject.
        inject_engram = self.engram_hash is not None and engram is not False
        # The Engram gate is the only consumer of the rotation basis here -- the
        # weights themselves were rotated when the export was written.
        gate_basis = (
            self.rotation_plan.engram_gate_basis
            if rotate is None
            else (self.rotation_plan.global_rotation if rotate else None)
        )
        image_mask = None if token_types is None else token_types >= 0  # TEXT is -1
        # image tokens take no part in an n-gram and get no engram contribution; text-only needs no mask
        engram_mask = None if image_mask is None else ~image_mask
        # hash buffers are CPU-resident, so hash the cpu copy of the ids
        engram_hashes = None
        if inject_engram:
            try:
                engram_hashes = self.engram_hash(input_ids.cpu(), start_pos, engram_mask)
            except Exception:
                # The hash state is rebuilt off the meta context by the adapter;
                # if it is still meta the failure is not obvious from the raise.
                get_logger().error(
                    'engram_hash failed; buffers: %s',
                    {n: (str(b.device), b.is_meta) for n, b in self.engram_hash.named_buffers()},
                )
                raise
        h = self.embed(input_ids)
        # embed may stay on CPU (shared-leaf offload target); the layer stack
        # computes on the runner device — align before the first block runs.
        if h.device != input_ids.device:
            h = h.to(input_ids.device)
        if images is not None:
            assert start_pos == 0, "image spans must be prefilled in a single chunk"  # nosec
            self.merge_image_embeddings(images, h)
        # Expand to hc_mult copies for Hyper-Connections (expand+contiguous:
        # torch 2.10 CPU repeat hits a spurious dim error on inference tensors)
        h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        if engram_hashes is not None and engram_hashes.device != h.device:
            engram_hashes = engram_hashes.to(h.device)
        main_hiddens = []
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        # A new forward means new batches: drop the previous walk's shared slots.
        shared_attn.reset()
        for i, layer in enumerate(self.layers):
            if inject_engram and layer.engram is not None:
                h = layer.engram(
                    h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask, gate_basis=gate_basis
                )
            # the MTP head reads the attention input of its target layers, not their output
            if i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix = layer(h, start_pos, pre_mix, image_mask, input_ids=input_ids)
        h = layer.hc_pre(h, pre_mix)
        _hn = self.norm(h)
        logits = self.head(_hn, full_logits=full_logits)
        # Every-position sampling would be wasted work, and the caller of
        # ``full_logits`` reads the logits itself.
        output_ids = None if full_logits else sample(logits, self.temperature)
        if main_hiddens:
            _ref = main_hiddens[-1].device
            main_hiddens = [m.to(_ref) for m in main_hiddens]
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        if dspark and output_ids is not None and main_hidden is not None and len(self.mtp) > 0:
            # Walk the draft prefill so its blocks reach their pause points.  It
            # returns nothing (start_pos 0 only seeds the window KV cache); the
            # draft that actually predicts is the per-step call.
            self._draft(output_ids, main_hidden, 0, greedy)
        return output_ids, logits, main_hidden

    @staticmethod
    def _accepted_prefix(logits: torch.Tensor, anchor: int, pending: torch.Tensor) -> int:
        """How many of a draft's tokens the backbone's own argmax agrees with.

        ``logits`` comes from a forward over the sequence *plus* the draft's own
        tokens, so position ``anchor + j`` is the backbone's prediction for what
        follows ``pending[:, j - 1]``.  Compare it with ``pending[:, j]`` and stop
        at the first disagreement -- that prefix is what speculative decoding
        would accept, and how often it runs to the end is how much of the draft
        (and therefore of the quantization) survived.
        """
        for j in range(pending.shape[1]):
            if int(logits[0, anchor + j].argmax()) != int(pending[0, j]):
                return j
        return pending.shape[1]

    def _text(self, ids: torch.Tensor) -> str:
        """Decode ids for a log line; '' when the shell has no tokenizer."""
        if self.tokenizer is None:
            return ''
        return repr(self.tokenizer.decode(ids.reshape(-1).tolist()))

    def _draft(self, input_ids: torch.Tensor, main_hidden: torch.Tensor, start_pos: int = 0, greedy: bool = False):
        """The DSpark half, reached through :meth:`forward` (``main_hidden=``).

        Runs on the caller's inference-mode context, so it carries no decorator
        of its own.  ``start_pos == 0`` is the prefill call: it seeds the draft
        window KV cache and returns ``None``, because there is nothing to draft
        until the backbone has committed a token.

        The embed/head phases call ``forward_embed`` / ``forward_head`` directly
        instead of ``block(phase=...)`` so each draft block enters its ``forward``
        exactly once (in the loop below); the layer-wise calibration runner then
        visits each block a single time and may offload it to meta afterwards.
        """
        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for layer in self.mtp:
            h, pre_mix = layer(h, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(x=h, pre_mix=pre_mix, input_ids=input_ids, greedy=greedy)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        eos_token_id=None,
        pad_token_id=None,
        streamer=None,
        dspark: bool | None = None,
        **kwargs,
    ):
        """Greedy or sampled generation -- the model's other entry point.

        ``attention_mask``, ``pad_token_id`` and ``**kwargs`` are accepted and
        ignored: the engine's ``PrefillLoop`` passes HF-shaped generation kwargs
        and this graph attends over everything it is handed.

        ``dspark`` also runs the DSpark draft on every step.  ``None`` (the
        default) takes the model's own ``dspark`` attribute, which the adapter
        sets from its module-level switch -- the engine's ``PrefillLoop`` calls
        this with kwargs of its own choosing and has no way to pass a flag
        through.
        """
        if dspark is None:
            dspark = self.dspark
        dspark = dspark and len(self.mtp) > 0
        generated = input_ids
        if streamer is not None:
            # HF's TextStreamer takes its FIRST put to be the prompt and skips it
            # (``skip_prompt``), so hand it the prompt up front -- otherwise it
            # eats the first generated token.
            streamer.put(input_ids)
        pending = None
        for step in range(max_new_tokens):
            prefix_len = generated.shape[1]
            if dspark and pending is not None:
                # Score last step's proposal in the forward we need anyway: causal
                # attention means appending the draft's tokens cannot move the
                # logits at the position we sample from, so next_logits below is
                # the token the un-extended forward would have produced.
                proposed = pending[:, :-1]
                if proposed.device != generated.device:
                    # The draft's own leaves can be CPU-resident (the adapter
                    # keeps `*.embed`/`*.head`/`mtp.*` off the runner device), so
                    # its argmax can come back on whichever device its head sits
                    # on.  Say so rather than fail on the cat below.
                    get_logger().warning(
                        'draft proposal on %s while the sequence is on %s; moving it over',
                        proposed.device,
                        generated.device,
                    )
                    proposed = proposed.to(generated.device)
                output_ids, logits, main_hidden = self(torch.cat([generated, proposed], dim=-1), full_logits=True)
                next_logits = logits[:, prefix_len - 1]
                # The draft conditions on what was really generated, never on its
                # own proposal -- otherwise the acceptance rate measures the wrong
                # thing.
                main_hidden = main_hidden[:, :prefix_len]
                accepted = self._accepted_prefix(logits, prefix_len - 1, pending)
            else:
                output_ids, logits, main_hidden = self(generated)
                # ParallelHead already returns last-position logits: [B, vocab]
                next_logits = logits if logits.dim() == 2 else logits[:, -1]
                accepted = None
            if do_sample:
                probs = (next_logits / max(temperature, 1e-5)).softmax(-1)
                next_id = probs.multinomial(1)
            else:
                next_id = next_logits.argmax(-1, keepdim=True)
            get_logger().info('generate step %d: token %d %s', step, int(next_id.reshape(-1)[-1]), self._text(next_id))
            if accepted is not None:
                get_logger().info('DSpark accept %d/%d', accepted, pending.shape[1])
            generated = torch.cat([generated, next_id], dim=-1)
            if streamer is not None:
                streamer.put(next_id)  # HF TextStreamer expects a tensor
            if dspark:
                # start_pos=0 is the prefill call that seeds the draft window KV
                # cache (it returns None); later steps report the drafted tokens.
                start_pos = 0 if step == 0 else generated.shape[1] - 1
                # Condition on the token this loop actually appended, never on
                # ``output_ids``: that is sample()'s Gumbel-max draw, a different
                # token that never enters the sequence.
                draft = self(next_id.squeeze(-1), start_pos=start_pos, main_hidden=main_hidden, greedy=not do_sample)
                pending = None
                if draft is not None:
                    draft_ids, draft_logits, _ = draft
                    # Column 0 is the anchor echoed back, not a proposal.
                    pending = draft_ids[:, 1:]
                    get_logger().info(
                        'DSpark draft @%d: %s | ids=%s max_logit=%.4f',
                        start_pos,
                        self._text(draft_ids),
                        draft_ids.tolist(),
                        float(draft_logits.max()),
                    )
            if eos_token_id is not None and (generated[:, -1] == eos_token_id).all():
                break
        if streamer is not None:
            streamer.end()
        return generated


def sample(logits, temperature: float = 1.0):
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(dspark_block_size=6, dspark_target_layer_ids=(3, 4))
    x = torch.randint(0, args.vocab_size, (2, 150))
    model = Transformer(args)

    output_ids, logits, main_hidden = model(x[:, :128])
    model(output_ids, start_pos=0, main_hidden=main_hidden)
    for i in range(128, 150):
        output_ids, logits, main_hidden = model(x[:, i : i + 1], i)
        result = model(output_ids, start_pos=i, main_hidden=main_hidden)
        assert result is not None  # nosec
        output_ids, logits, confidence = result
