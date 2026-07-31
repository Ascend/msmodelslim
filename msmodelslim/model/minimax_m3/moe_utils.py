#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import gc
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import PrepareWeight

__all__ = [
    "UnstackedM3MoeExpert",
    "UnstackedM3MoeBlock",
]


class UnstackedM3MoeExpert(nn.Module):
    """A single MoE expert with standard nn.Linear layers (post-conversion).

    Matches the forward semantics of the saved Mixtral-style w1/w2/w3
    expert and the transformers MiniMaxM3VLExperts slice, but exposes
    gate_proj / up_proj / down_proj so the standard quant hooks apply.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.swiglu_limit)
        up = self.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj((up + 1.0) * gate * torch.sigmoid(gate * self.swiglu_alpha))


class UnstackedM3DenseMLP(nn.Module):
    """Unstacked dense MLP with separate gate_proj / up_proj (instead of fused gate_up_proj).

    Matches the forward semantics of `MiniMaxM3VLDenseMLP` but exposes
    individual nn.Linear modules so the standard quant hooks apply.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return self.down_proj((up + 1.0) * glu)


class UnstackedM3MoeBlock(nn.Module):
    """MiniMax-M3 MoE block with per-expert nn.Linear (post-conversion).

    Replaces a transformers `MiniMaxM3VLSparseMoeBlock` (which holds 3D
    expert weights as Parameters) by a ModuleList of standard nn.Linear
    experts, so the standard quantization pipeline can hook them.

    The router (`gate`, an `nn.Linear`) and the shared experts
    (`shared_experts`, an `UnstackedM3DenseMLP`) are also unstacked.
    """

    def __init__(
        self,
        config,
        original_moe_block: Optional[nn.Module] = None,
        copy_weights: bool = False,
    ):
        super().__init__()
        if hasattr(config, "text_config"):
            text_config = config.text_config
        else:
            text_config = config

        self.hidden_size = text_config.hidden_size
        self.intermediate_size = text_config.intermediate_size
        self.shared_intermediate_size = getattr(text_config, "shared_intermediate_size", self.intermediate_size)
        self.num_experts = text_config.num_local_experts
        self.top_k = text_config.num_experts_per_tok
        self.swiglu_alpha = getattr(text_config, "swiglu_alpha", 1.702)
        self.swiglu_limit = getattr(text_config, "swiglu_limit", 7.0)
        self.routed_scaling_factor = getattr(text_config, "routed_scaling_factor", 1.0)

        dtype = next(original_moe_block.parameters()).dtype if original_moe_block is not None else None

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False, dtype=dtype)
        self.experts = nn.ModuleList(
            [
                UnstackedM3MoeExpert(
                    self.hidden_size,
                    self.intermediate_size,
                    self.swiglu_alpha,
                    self.swiglu_limit,
                    dtype=dtype,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.shared_experts = UnstackedM3DenseMLP(
            self.hidden_size,
            self.shared_intermediate_size,
            self.swiglu_alpha,
            self.swiglu_limit,
            dtype=dtype,
        )
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

        if original_moe_block is not None and copy_weights:
            self._transform_weights_from_original(original_moe_block, in_place=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)

        shared_output = self.shared_experts(flat)

        router_logits = F.linear(flat.to(self.gate.weight.dtype), self.gate.weight)
        scores = torch.sigmoid(router_logits.float())
        choice_scores = scores + self.e_score_correction_bias.unsqueeze(0)
        _, top_k_index = torch.topk(choice_scores, self.top_k, dim=-1, sorted=False)
        top_k_weights = scores.gather(1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat)
        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(top_k_index == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[expert_idx](flat[token_idx])
            output[token_idx] += expert_out * top_k_weights[token_idx, slot_idx].to(expert_out.dtype).unsqueeze(-1)

        output = output * self.routed_scaling_factor
        output = output + shared_output
        output = output.reshape(batch_size, sequence_length, hidden_dim)
        return output

    def _transform_weights_from_original(
        self,
        original_moe_block: nn.Module,
        in_place: bool = True,
    ) -> None:
        """Convert 3D expert weights (transformers native) to per-expert nn.Linear.

        The original block is assumed to be a transformers
        `MiniMaxM3VLSparseMoeBlock` with `gate` (a `MiniMaxM3VLTopKRouter`),
        `experts` (a `MiniMaxM3VLExperts`), `shared_experts`, and
        `routed_scaling_factor`.
        """
        with torch.no_grad():
            with PrepareWeight(original_moe_block.gate):
                gate_weight = original_moe_block.gate.weight.data
                self.gate.weight = nn.Parameter(gate_weight.contiguous().cpu(), requires_grad=False)
                if (
                    hasattr(original_moe_block.gate, "e_score_correction_bias")
                    and original_moe_block.gate.e_score_correction_bias is not None
                ):
                    self.e_score_correction_bias = (
                        original_moe_block.gate.e_score_correction_bias.data.cpu().contiguous()
                    )

            with PrepareWeight(original_moe_block.experts):
                full_gate_up = original_moe_block.experts.gate_up_proj.data.cpu()
                full_down = original_moe_block.experts.down_proj.data.cpu()

            inter = self.intermediate_size
            for i in range(self.num_experts):
                gate_up_i = full_gate_up[i]
                down_i = full_down[i]
                self.experts[i].gate_proj.weight = nn.Parameter(gate_up_i[:inter, :].contiguous(), requires_grad=False)
                self.experts[i].up_proj.weight = nn.Parameter(gate_up_i[inter:, :].contiguous(), requires_grad=False)
                self.experts[i].down_proj.weight = nn.Parameter(down_i.contiguous(), requires_grad=False)

            if original_moe_block.shared_experts is not None:
                with PrepareWeight(original_moe_block.shared_experts):
                    shared_gate_up = original_moe_block.shared_experts.gate_up_proj.weight.data.cpu()
                    shared_down = original_moe_block.shared_experts.down_proj.weight.data.cpu()

                shared_inter = self.shared_intermediate_size
                self.shared_experts.gate_proj.weight = nn.Parameter(
                    shared_gate_up[:shared_inter, :].contiguous(), requires_grad=False
                )
                self.shared_experts.up_proj.weight = nn.Parameter(
                    shared_gate_up[shared_inter:, :].contiguous(), requires_grad=False
                )
                self.shared_experts.down_proj.weight = nn.Parameter(shared_down.contiguous(), requires_grad=False)

            self.routed_scaling_factor = original_moe_block.routed_scaling_factor

        if in_place:
            del original_moe_block
            gc.collect()
