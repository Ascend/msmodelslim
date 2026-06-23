# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.

import gc

import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

__all__ = [
    "UnstackedGlm4MoeLiteExpertMLP",
    "UnstackedGlm4MoeLiteTopkRouter",
    "UnstackedGlm4MoeLiteMoE",
]


class UnstackedGlm4MoeLiteExpertMLP(nn.Module):
    """Single expert MLP using nn.Linear, matching the safetensors key layout."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnstackedGlm4MoeLiteTopkRouter(nn.Module):
    """
    Router compatible with Glm4MoeLiteTopkRouter, but stores e_score_correction_bias
    as nn.Parameter so the saver can persist it.

    modelslim_v1 saver only writes named_parameters(...), not buffers. If we keep
    e_score_correction_bias as register_buffer like transformers does, it participates
    in forward but disappears in exported weights. Promoting it to Parameter preserves
    the original key path:

        mlp.gate.weight
        mlp.gate.e_score_correction_bias
    """

    def __init__(self, config, original_gate: nn.Module):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = getattr(config, 'num_experts', getattr(config, 'n_routed_experts'))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = getattr(config, 'n_group', 1)
        self.topk_group = getattr(config, 'topk_group', 1)
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)

        self.weight = nn.Parameter(original_gate.weight.detach().clone())
        self.e_score_correction_bias = nn.Parameter(original_gate.e_score_correction_bias.detach().clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        return F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))


class UnstackedGlm4MoeLiteMoE(nn.Module):
    """
    Drop-in replacement for Glm4MoeLiteNaiveMoe, but with per-expert nn.Linear modules:

        mlp.experts.0.gate_proj.weight
        mlp.experts.0.up_proj.weight
        mlp.experts.0.down_proj.weight

    The original Glm4MoeLiteNaiveMoe stores expert weights as fused 3D tensors
    (gate_up_proj, down_proj). This class unstacks them into individual nn.Linear
    modules so that the quantization tool can access each expert via standard paths.
    """

    def __init__(self, config, original_moe: nn.Module):
        super().__init__()

        self.config = config
        self.n_routed_experts = getattr(config, 'num_experts', getattr(config, 'n_routed_experts'))
        self.n_group = getattr(config, 'n_group', 1)
        self.topk_group = getattr(config, 'topk_group', 1)
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

        self.gate = UnstackedGlm4MoeLiteTopkRouter(config, original_moe.gate)
        self.shared_experts = original_moe.shared_experts

        self.num_experts = getattr(config, 'num_local_experts', self.n_routed_experts)
        intermediate_size = getattr(config, 'moe_intermediate_size', config.intermediate_size)
        hidden_act = getattr(config, 'hidden_act', 'silu')
        dtype = next(original_moe.experts.parameters()).dtype

        self.experts = nn.ModuleList(
            [
                UnstackedGlm4MoeLiteExpertMLP(
                    config.hidden_size,
                    intermediate_size,
                    hidden_act,
                    dtype=dtype,
                )
                for _ in range(self.num_experts)
            ]
        )

        self._unstack_expert_weights(original_moe.experts)

        for attr in ("gate_up_proj", "down_proj"):
            if hasattr(original_moe.experts, attr):
                delattr(original_moe.experts, attr)
        gc.collect()

    def _unstack_expert_weights(self, original_experts: nn.Module):
        """Split fused 3D expert weights into individual per-expert nn.Linear weights."""
        gate_up_proj = original_experts.gate_up_proj
        down_proj = original_experts.down_proj

        intermediate_size = self.experts[0].gate_proj.out_features

        for i in range(self.num_experts):
            expert = self.experts[i]
            expert.gate_proj.weight.data.copy_(gate_up_proj[i, :intermediate_size, :])
            expert.up_proj.weight.data.copy_(gate_up_proj[i, intermediate_size:, :])
            expert.down_proj.weight.data.copy_(down_proj[i])

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def _dispatch_to_experts(
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
            expert_idx = expert_idx_tensor[0].item()
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_hidden_states = self.experts[expert_idx](hidden_states[token_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self._dispatch_to_experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states
