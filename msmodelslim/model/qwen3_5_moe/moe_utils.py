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

import torch
from torch import nn
import torch.nn.functional as F
from msmodelslim.utils.logging import get_logger
from transformers.activations import ACT2FN

logger = get_logger(__name__)


"""
Qwen3-VL MoE Model Utilities for Quantization

This module provides utilities to convert Qwen3-VL MoE models with fused 3D expert weights
into a format compatible with standard quantization pipelines that expect nn.Linear layers.

Key Features:
    - Unstacks 3D expert weights (num_experts, hidden_size, expert_dim) into individual nn.Linear layers
    - Memory-efficient in-place weight transformation
    - Maintains functional equivalence with original MoE implementation
    - Enables standard W8A8 quantization without modifying core quantization logic
"""


class Qwen3_5MoeExpertMLP(nn.Module):
    """Single expert MLP with separate gate_proj, up_proj, and down_proj layers."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert MLP.

        Args:
            x: Input tensor of shape (batch_size, hidden_dim)

        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down


class Qwen3_5MoeTopKRouter(nn.Module):
    """Top-k router copied from transformers Qwen3.5 MoE implementation."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.hidden_size)))

    def forward(self, hidden_states: torch.Tensor):
        router_logits = F.linear(hidden_states, self.weight)  # (num_tokens, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (num_tokens, top_k)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3_5MoeSparseMoeBlockWithMLP(nn.Module):
    """
    Sparse MoE block compatible with transformers Qwen3.5 flow:
    gate -> topk experts -> shared expert gate -> sum.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate = Qwen3_5MoeTopKRouter(config)

        # Keep experts as a flat ModuleList to avoid nested experts.experts path.
        self.experts = nn.ModuleList(
            [Qwen3_5MoeExpertMLP(self.hidden_dim, self.intermediate_dim, self.act_fn) for _ in range(self.num_experts)]
        )

        self.shared_expert = Qwen3_5MoeExpertMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            self.act_fn,
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        # Routed experts forward (same logic as transformers Qwen3.5 MoE experts dispatch)
        expert_output = torch.zeros_like(hidden_states_reshaped)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, num_tokens)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            expert_output.index_add_(0, token_idx, current_hidden_states.to(expert_output.dtype))

        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output += shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


def convert_experts_to_mlp(
    original_moe_block,
    config,
) -> Qwen3_5MoeSparseMoeBlockWithMLP:
    """
    Convert a transformers Qwen3_5MoeSparseMoeBlock into an unpacked-MLP version.
    """
    new_moe_block = Qwen3_5MoeSparseMoeBlockWithMLP(config)

    with torch.no_grad():
        # Router
        new_moe_block.gate.weight.copy_(original_moe_block.gate.weight)

        # Routed experts: split gate_up_proj into gate_proj + up_proj
        for expert_idx in range(config.num_experts):
            gate_up_weight = original_moe_block.experts.gate_up_proj[expert_idx]
            gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)

            new_moe_block.experts[expert_idx].gate_proj.weight.copy_(gate_weight)
            new_moe_block.experts[expert_idx].up_proj.weight.copy_(up_weight)
            new_moe_block.experts[expert_idx].down_proj.weight.copy_(original_moe_block.experts.down_proj[expert_idx])

        # Shared expert and gate
        new_moe_block.shared_expert.gate_proj.weight.copy_(original_moe_block.shared_expert.gate_proj.weight)
        new_moe_block.shared_expert.up_proj.weight.copy_(original_moe_block.shared_expert.up_proj.weight)
        new_moe_block.shared_expert.down_proj.weight.copy_(original_moe_block.shared_expert.down_proj.weight)
        new_moe_block.shared_expert_gate.weight.copy_(original_moe_block.shared_expert_gate.weight)

    new_moe_block = new_moe_block.to(
        device=original_moe_block.gate.weight.device,
        dtype=original_moe_block.gate.weight.dtype,
    )
    return new_moe_block
