# coding=utf-8
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 Intel Corporation

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

# Note: adapted from: https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/main/modeling_step3p5.py


import torch
import torch.nn.functional as F
from torch import nn
from msmodelslim.utils.logging import get_logger
from transformers.activations import ACT2FN

logger = get_logger(__name__)


def sigmoid_routing_function(gating_output: torch.Tensor, topk: int, renormalize: bool):
    gating_output = gating_output.float()
    gate_prob = torch.sigmoid(gating_output)
    gate_prob = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
    topk_prob, indices = torch.topk(gate_prob, k=topk, dim=1)
    expert_topk_weight = topk_prob
    if renormalize:
        expert_topk_weight = expert_topk_weight / torch.sum(expert_topk_weight, dim=-1, keepdim=True)
    return expert_topk_weight, indices


class Step3p5MoeExpertMLP(nn.Module):
    def __init__(self, config, intermediate_size=None, swiglu_limit=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up)


class Step3p5MoEMLPWithUnpackExperts(nn.Module):
    """
    Sparse MoE block compatible with Step3p5MoeExpertMLP flow.
    """

    def __init__(self, config, swiglu_limit=None):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.use_moe_router_bias = config.use_moe_router_bias
        if self.use_moe_router_bias:
            self.router_bias = nn.Parameter(
                torch.zeros(config.moe_num_experts, dtype=torch.float32), requires_grad=False
            )
            self.custom_routing_function = self.router_bias_func
        elif config.moe_router_activation == "sigmoid":
            self.custom_routing_function = sigmoid_routing_function
        else:
            self.custom_routing_function = None

        self.need_fp32_gate = config.need_fp32_gate
        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

        # Keep experts as a flat ModuleList to avoid nested experts.experts path.
        self.experts = nn.ModuleList(
            [Step3p5MoeExpertMLP(config, self.moe_intermediate_size, self.limit) for _ in range(self.num_experts)]
        )

    def router_bias_func(self, gating_output: torch.Tensor, topk: int, renormalize: bool):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20)
        return expert_topk_weight, indices

    def forward(self, hidden_states):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.need_fp32_gate:
            router_logits = torch.matmul(hidden_states.to(torch.float32), self.gate.weight.t().to(torch.float32))
        else:
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

        if self.custom_routing_function:
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights = routing_weights * self.routed_scaling_factor

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            # current_hidden_states = (
            #     self.get_expert_output(current_state, expert_idx) *
            #     routing_weights[top_x, idx, None])
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


def convert_step35_moe_to_unpacked(
    original_moe_block,
    config,
    swiglu_limit=None,
) -> Step3p5MoEMLPWithUnpackExperts:
    """
    Convert a Step3p5MoEMLP into a Step3p5MoEMLPWithUnpackExperts version
    """
    # Create a new unpacked MoE block
    swiglu_limit = original_moe_block.limit if hasattr(original_moe_block, 'limit') else swiglu_limit
    new_moe_block = Step3p5MoEMLPWithUnpackExperts(config, swiglu_limit)

    with torch.no_grad():
        # Copy router (gate) weights
        new_moe_block.gate.weight.copy_(original_moe_block.gate.weight)

        # Copy router bias if exists
        if original_moe_block.use_moe_router_bias:
            new_moe_block.router_bias.copy_(original_moe_block.router_bias)

        # Copy expert weights
        for expert_idx in range(config.moe_num_experts):
            # Get weights from original MoELinear layers
            # Original: MoELinear.weight shape is (num_experts, out_features, in_features)

            gate_weight = original_moe_block.gate_proj.weight[expert_idx]  # (intermediate_size, hidden_size)
            new_moe_block.experts[expert_idx].gate_proj.weight.copy_(gate_weight)

            up_weight = original_moe_block.up_proj.weight[expert_idx]  # (intermediate_size, hidden_size)
            new_moe_block.experts[expert_idx].up_proj.weight.copy_(up_weight)

            down_weight = original_moe_block.down_proj.weight[expert_idx]  # (hidden_size, intermediate_size)
            new_moe_block.experts[expert_idx].down_proj.weight.copy_(down_weight)

    logger.info(
        "Converted Step3p5MoEMLP to Step3p5MoEMLPWithUnpackExperts: "
        "num_experts=%s, hidden_size=%s, moe_intermediate_size=%s",
        config.moe_num_experts,
        config.hidden_size,
        config.moe_intermediate_size,
    )

    return new_moe_block
