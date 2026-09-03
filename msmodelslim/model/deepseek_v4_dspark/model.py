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

DeepSeek-V4 DSpark MTP 模型结构。

从官方 DSpark 推理代码移植，复用 deepseek_v4 的基础模块（Block / Attention / sparse_attn 等），
供量化校准前向与权重加载使用。
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..deepseek_v4.model import (
    USE_DP_MODE,
    Attention,
    Block,
    MoE,
    ModelArgs,
    ParallelEmbedding,
    RMSNorm,
    sparse_attn,
    world_size,
)


def _dist_comm_device(tensor: torch.Tensor) -> torch.device:
    """逐层 offload 后激活可能在 CPU，HCCL/NCCL 集合通信需落回 NPU。"""
    if tensor.device.type != "cpu":
        return tensor.device
    if dist.is_initialized() and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{torch.npu.current_device()}")
    return tensor.device


class DSparkMoE(MoE):
    """DSpark 校准路径：CPU 激活进入 MoE DP all_gather 前先对齐到通信设备。"""

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if world_size > 1 and USE_DP_MODE:
            comm_device = _dist_comm_device(x)
            if x.device != comm_device:
                x = x.to(comm_device)
            if input_ids is not None and input_ids.device != comm_device:
                input_ids = input_ids.to(comm_device)
        return super().forward(x, input_ids)


def enable_dspark_moe_comm(module: nn.Module) -> None:
    """将模块内的 MoE 切换为 DSparkMoE（保留已有权重，不重新初始化）。"""
    for child in module.modules():
        # 仅替换基类 MoE，避免重复包装已是 DSparkMoE 的实例
        if isinstance(child, MoE) and not isinstance(child, DSparkMoE):
            child.__class__ = DSparkMoE


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
    """DSpark MTP 注意力：prefill 用 main 分支写 KV cache，decode 用 draft x 做 Q。"""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.compress_ratio = 0
        self.compressor = None
        self.indexer = None

    def forward(self, x: torch.Tensor, start_pos: int, main_x: Optional[torch.Tensor] = None):
        if main_x is None:
            return super().forward(x, start_pos)

        assert self.compress_ratio == 0  # nosec
        bsz, seqlen, _ = main_x.size()
        win = self.window_size
        rd = self.rope_head_dim

        main_freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        from ..deepseek_v4.model import apply_rotary_emb

        apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = main_kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            return x

        bsz, block_size, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]

        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos).to(x.device)
        self.kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)
        kv = torch.cat([self.kv_cache[:bsz], kv], dim=1)
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


class ParallelHead(nn.Module):
    """与官方 DSpark 一致的 vocab 并行 head（末层 MTP logits 用）。"""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.part_vocab_size = vocab_size // world_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, full_logits: bool = False):
        if not full_logits:
            x = x[:, -1]
        logits = F.linear(x.float(), self.weight)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, dspark_markov_rank: int):
        super().__init__()
        self.markov_w1 = ParallelEmbedding(vocab_size, dspark_markov_rank)
        self.markov_w2 = ParallelHead(vocab_size, dspark_markov_rank)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.markov_w1(token_ids)
        logits = self.markov_w2(embed, full_logits=True)
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)


class DSparkBlock(Block):
    """DSpark MTP stage：mtp.0 含 main_proj/main_norm，末层含 norm/head/markov/confidence。"""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        enable_dspark_moe_comm(self)
        self.attn = DSparkAttention(layer_id, args)

        self.dim = args.dim
        stage_id = layer_id - args.num_hidden_layers
        self.block_size = getattr(args, "dspark_block_size", 0) or 1
        self.noise_token_id = getattr(args, "dspark_noise_token_id", 0)
        self.embed: Optional[ParallelEmbedding] = None
        self.head: Optional[nn.Module] = None

        if stage_id == 0:
            target_ids = getattr(args, "dspark_target_layer_ids", ()) or ()
            in_dim = args.dim * len(target_ids)
            self.main_proj = nn.Linear(in_dim, args.dim, bias=False)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)

        if stage_id == getattr(args, "n_mtp_layers", 0) - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            markov_rank = getattr(args, "dspark_markov_rank", 0) or 256
            self.markov_head = DSparkMarkovHead(args.vocab_size, markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + markov_rank)
            hc_dim = self.hc_mult * args.dim
            origin_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
            torch.set_default_dtype(origin_dtype)

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

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: Optional[torch.Tensor] = None,
        main_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if main_x is None:
            return super().forward(x, start_pos, input_ids)

        if start_pos > 0:
            residual = x
            x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
            x = self.attn_norm(x)
            x = self.attn(x, start_pos, main_x)
            x = self.hc_post(x, residual, post, comb)

            residual = x
            x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
            x = self.ffn_norm(x)
            x = self.ffn(x, input_ids)
            x = self.hc_post(x, residual, post, comb)
            return x

        return self.attn(x, start_pos, main_x)

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        """DSpark MTP 入口：main_hidden 拼接 + draft token embed。"""
        assert self.embed is not None, "DSpark MTP embed 未绑定到主模型 embed"  # nosec
        main_x = self.main_norm(self.main_proj(main_hidden))
        if input_ids.dim() == 2:
            input_ids = input_ids[:, -1]
        draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)
        draft_input_ids[:, 0] = input_ids
        x = self.embed(draft_input_ids)  # pylint: disable=not-callable
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def forward_head(self, x: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0):
        """末层 MTP head（decode 阶段）。"""
        assert self.head is not None  # nosec
        x = self.hc_head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        if hasattr(self.head, "forward") and isinstance(self.head, ParallelHead):
            logits = self.head(self.norm(x), full_logits=True)  # pylint: disable=not-callable
        else:
            normed = self.norm(x)
            # pylint: disable-next=not-callable
            logits = self.head(normed.float() if normed.dtype != torch.float32 else normed)

        if input_ids.dim() == 2:
            seed_ids = input_ids[:, -1]
        else:
            seed_ids = input_ids

        output_ids = input_ids.new_empty(seed_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = seed_ids
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, i])
            logits[:, i].add_(logits_bias)
            markov_embeds.append(markov_embed)
            output_ids[:, i + 1] = _sample(logits[:, i], temperature)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(x, markov_embed)
        return output_ids, logits, confidence


def _sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
