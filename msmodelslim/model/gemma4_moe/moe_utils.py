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

Gemma4 MoE utilities for quantization.

Converts Gemma4TextExperts fused 3D expert weights into per-expert nn.Linear layers
so standard W8A8 / subgraph quantization can target each projection.
"""

import gc
from typing import Optional

import torch
from torch import nn

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import PrepareWeight
from msmodelslim.utils.logging import get_logger

try:
    from transformers.activations import ACT2FN
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
except ImportError as e:
    get_logger().warning("Failed to import Gemma4 modules: %s", e)
    ACT2FN = None
    Gemma4TextExperts = None
    Gemma4TextConfig = None

__all__ = [
    'UnstackedGemma4TextExpertMLP',
    'UnstackedGemma4TextExperts',
]


def _scalar_expert_index(expert_idx) -> int:
    """Convert expert index from tensor/scalar to Python int for submodule lookup."""
    if isinstance(expert_idx, torch.Tensor):
        return int(expert_idx.item())
    return int(expert_idx)


class UnstackedGemma4TextExpertMLP(nn.Module):
    """Single Gemma4 MoE expert with standard nn.Linear projections."""

    def __init__(
        self,
        config: "Gemma4TextConfig",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=dtype)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnstackedGemma4TextExperts(nn.Module):
    """
    Drop-in replacement for Gemma4TextExperts using unstacked nn.Linear experts.

    Keeps the same forward signature: (hidden_states, top_k_index, top_k_weights).

    Per-expert MLPs are registered as submodules "0", "1", ... so export paths are
    ``...layers.{id}.experts.{e}.gate_proj`` (not ``...experts.experts.{e}.``).
    """

    def __init__(
        self,
        config: "Gemma4TextConfig",
        original_experts: "Gemma4TextExperts",
        copy_weights: bool = False,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        dtype = next(original_experts.parameters()).dtype
        for expert_idx in range(self.num_experts):
            self.add_module(
                str(expert_idx),
                UnstackedGemma4TextExpertMLP(config, dtype=dtype),
            )
        self.act_fn = ACT2FN[config.hidden_activation]

        if copy_weights:
            self._transform_weights_from_original(original_experts, in_place=False)

    def _get_expert(self, expert_idx) -> UnstackedGemma4TextExpertMLP:
        return self._modules[str(_scalar_expert_index(expert_idx))]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = _scalar_expert_index(expert_idx_tensor[0])
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self._get_expert(expert_idx)(current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def _transform_weights_from_original(
        self,
        original_experts: "Gemma4TextExperts",
        in_place: bool = True,
    ) -> None:
        """
        Transform fused 3D weights into per-expert Linear weights.

        Original shapes:
            gate_up_proj: (num_experts, 2 * intermediate_dim, hidden_dim)
            down_proj:    (num_experts, hidden_dim, intermediate_dim)
        """
        with torch.no_grad():
            with PrepareWeight(original_experts):
                gate_up_param = original_experts.gate_up_proj
                down_param = original_experts.down_proj
                full_gate_up_proj = gate_up_param.data.cpu()
                full_down_proj = down_param.data.cpu()

            for expert_idx in range(self.num_experts):
                gate_up_weight = full_gate_up_proj[expert_idx]
                down_weight = full_down_proj[expert_idx]
                gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)

                expert = self._get_expert(expert_idx)
                expert.gate_proj.weight = nn.Parameter(gate_weight.contiguous(), requires_grad=False)
                expert.up_proj.weight = nn.Parameter(up_weight.contiguous(), requires_grad=False)
                expert.down_proj.weight = nn.Parameter(down_weight.contiguous(), requires_grad=False)

            del full_gate_up_proj, full_down_proj

        if in_place:
            if hasattr(original_experts, "gate_up_proj"):
                del original_experts.gate_up_proj
            if hasattr(original_experts, "down_proj"):
                del original_experts.down_proj
            gc.collect()
