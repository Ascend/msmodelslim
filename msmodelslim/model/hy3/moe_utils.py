#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# Note: MoE dispatch and forward logic in UnstackedHy3MoE are adapted from
# transformers.models.hy_v3.modeling_hy_v3 (HYV3Experts / HYV3MoE).

"""
Hy3 MoE utilities for quantization.

Convert fused 3D expert Parameters (gate_up_proj / down_proj) into per-expert
nn.Linear modules so modelslim_v1 linear_quant can quantize routed experts.
"""

import time
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from msmodelslim.utils.logging import get_logger

__all__ = [
    "Hy3ExpertMLP",
    "UnstackedHy3MoE",
    "convert_hy3_moe_to_unstacked",
]


class Hy3ExpertMLP(nn.Module):
    """Single expert MLP with separate gate_proj, up_proj, and down_proj layers."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnstackedHy3MoE(nn.Module):
    """Drop-in replacement for HY3MoE with per-expert nn.Linear modules.

    Module names match the floating-point checkpoint layout:
    ``router.gate``, ``expert_bias``, ``shared_mlp``.
    """

    def __init__(self, config, original_moe: nn.Module):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.enable_moe_fp32_combine = config.enable_moe_fp32_combine

        self.router = nn.Module()
        self.router.gate = original_moe.gate
        self.shared_mlp = original_moe.shared_experts

        # Keep fp32 slot for expert_bias (upstream preserve + reload from ckpt).
        # Unstack renames e_score_correction_bias -> expert_bias; dtype must stay
        # fp32 so a later load_state_dict / assign does not truncate.
        bias = original_moe.e_score_correction_bias
        self.expert_bias = nn.Parameter(bias.detach().to(torch.float32), requires_grad=False)

        num_experts = config.num_experts
        hidden_dim = config.hidden_size
        intermediate_dim = config.moe_intermediate_size
        act_fn = ACT2FN[config.hidden_act]
        dtype = original_moe.experts.gate_up_proj.dtype
        device = original_moe.experts.gate_up_proj.device

        with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
            self.experts = nn.ModuleList(
                [Hy3ExpertMLP(hidden_dim, intermediate_dim, act_fn) for _ in range(num_experts)]
            )
        self.experts.to(device=device, dtype=dtype)

    def _dispatch_to_experts(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        num_experts = len(self.experts)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor[0].item()
            if expert_idx >= num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_hidden_states = self.experts[expert_idx](hidden_states[token_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        _, top_k_weights, top_k_index = self.router.gate(hidden_states, self.expert_bias)
        routed_output = self._dispatch_to_experts(hidden_states, top_k_index, top_k_weights)

        if self.enable_moe_fp32_combine:
            hidden_states = (routed_output.float() + self.shared_mlp(hidden_states).float()).to(hidden_states.dtype)
        else:
            hidden_states = routed_output + self.shared_mlp(hidden_states)

        return hidden_states.reshape(batch_size, seq_len, hidden_dim)


def convert_hy3_moe_to_unstacked(original_moe: nn.Module, config) -> UnstackedHy3MoE:
    """Convert fused HY3Experts Parameters into per-expert nn.Linear modules."""
    t0 = time.time()
    device = original_moe.experts.gate_up_proj.device
    new_moe = UnstackedHy3MoE(config, original_moe)
    gate_up = original_moe.experts.gate_up_proj
    down = original_moe.experts.down_proj

    with torch.no_grad():
        # gate_up: [E, 2*I, H] -> gate/up: [E, I, H]
        gate_w, up_w = gate_up.chunk(2, dim=1)
        for expert_idx in range(config.num_experts):
            new_moe.experts[expert_idx].gate_proj.weight.copy_(gate_w[expert_idx])
            new_moe.experts[expert_idx].up_proj.weight.copy_(up_w[expert_idx])
            new_moe.experts[expert_idx].down_proj.weight.copy_(down[expert_idx])

    del original_moe.experts

    get_logger().debug(
        "Converted HY3MoE to unstacked experts on %s in %.2fs (num_experts=%s)",
        device,
        time.time() - t0,
        config.num_experts,
    )
    return new_moe
