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
"""
# pylint: disable=logging-fstring-interpolation
# Qwen3-VL MoE Model Utilities for Quantization
#
# This module provides utilities to convert Qwen3-VL MoE models with fused 3D expert weights
# into a format compatible with standard quantization pipelines that expect nn.Linear layers.
#
# Key Features:
#     - Unstacks 3D expert weights (num_experts, hidden_size, expert_dim) into individual nn.Linear layers
#     - Memory-efficient in-place weight transformation
#     - Maintains functional equivalence with original MoE implementation
#     - Enables standard W8A8 quantization without modifying core quantization logic

import gc
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
from torch import nn
from msmodelslim.model.common.utils import resolve_expert_ep_range
from msmodelslim.utils.logging import get_logger
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import (  # pylint: disable=no-name-in-module
    PrepareWeight,
)

# ACT2FN lives in every supported transformers release, while the Qwen3-VL-MoE
# model package only exists from 4.57.0.  Keeping the two imports independent means
# an older transformers (e.g. the 4.51.0 pinned by CI) still provides ACT2FN, so the
# standalone MLP/EP helpers keep working and only the model-typed conversion paths
# report the missing dependency.
try:
    from transformers.activations import ACT2FN
except ImportError as e:  # pragma: no cover - transformers always ships this module
    get_logger().warning(f"Failed to import ACT2FN: {e}")
    ACT2FN = None

try:
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextSparseMoeBlock,
    )
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeTextConfig,
    )
except ImportError as e:
    get_logger().warning(f"Qwen3VLMoe modules unavailable (requires transformers>=4.57.0): {e}")
    Qwen3VLMoeTextSparseMoeBlock = None
    Qwen3VLMoeTextConfig = None

_EP_REQUIRED_MSG = (
    "Qwen3-VL-MoE model conversion requires transformers>=4.57.0 (transformers.models.qwen3_vl_moe is unavailable)"
)


__all__ = [
    'UnstackedQwen3VLMoeTextMLP',
    'UnstackedQwen3VLMoeSparseMoeBlock',
    'convert_qwen3_moe_to_linear',
]


def _accumulate_expert_range(
    experts: nn.ModuleList,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_start: int,
    expert_end: int,
) -> torch.Tensor:
    """Sum the routed contributions of experts ``[expert_start, expert_end)``.

    The accumulator stays in float32 and experts are added in ascending global id
    order, which makes the result a deterministic function of the inputs.  Casting
    to the activation dtype here would round every EP rank's partial sum to bf16
    before the cross-rank reduction (~2^-9 relative error per rank), a much larger
    error source than the single-card path that rounds exactly once.
    """
    hidden_dim = hidden_states.shape[-1]
    output = torch.zeros(
        hidden_states.shape[0],
        hidden_dim,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for expert_id in range(expert_start, expert_end):
        expert = experts[expert_id]
        if expert is None:
            continue
        token_idx, topk_idx = torch.where(selected_experts == expert_id)
        if token_idx.numel() == 0:
            continue
        expert_output = expert(hidden_states[token_idx]).to(torch.float32)
        weights = routing_weights[token_idx, topk_idx].unsqueeze(-1).to(torch.float32)
        output.index_add_(0, token_idx, expert_output * weights)
    return output


def _sum_in_rank_order(partials: List[torch.Tensor]) -> torch.Tensor:
    """Add the per-rank partial sums in ascending rank order."""
    acc = partials[0]
    for part in partials[1:]:
        acc = acc + part
    return acc


def _gather_partials(local_partial: torch.Tensor) -> List[torch.Tensor]:
    """Collect every rank's partial into rank order with one contiguous collective."""
    world_size = dist.get_world_size()
    local_partial = local_partial.contiguous()
    if hasattr(dist, "all_gather_into_tensor"):
        gathered = local_partial.new_empty((world_size, *local_partial.shape))
        dist.all_gather_into_tensor(gathered, local_partial)
        return list(gathered.unbind(0))
    gathered = [torch.empty_like(local_partial) for _ in range(world_size)]
    dist.all_gather(gathered, local_partial)
    return gathered


def _gather_hidden_states(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Gather variable sequence lengths and return this rank's output slice."""
    from msmodelslim.utils.distributed import DistHelper

    local_seq_len = hidden_states.size(1)
    seq_len_tensor = torch.tensor([local_seq_len], dtype=torch.long, device=hidden_states.device)
    seq_len_tensors = [torch.zeros_like(seq_len_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(seq_len_tensors, seq_len_tensor)
    seq_lens = [int(item.item()) for item in seq_len_tensors]
    rank = dist.get_rank()
    start = sum(seq_lens[:rank])
    end = start + local_seq_len
    gathered = torch.cat(DistHelper.gather_variable_shapes(hidden_states), dim=1)
    return gathered, start, end


class UnstackedQwen3VLMoeTextMLP(nn.Module):
    """
    Single expert MLP with standard nn.Linear layers.

    This replaces a single expert's 3D weight slice with explicit Linear layers,
    making it compatible with standard quantization logic.
    """

    def __init__(
        self,
        config: "Qwen3VLMoeTextConfig",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # Use standard nn.Linear instead of 3D Parameter
        self.gate_proj = nn.Linear(self.hidden_size, self.expert_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(self.hidden_size, self.expert_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(self.expert_dim, self.hidden_size, bias=False, dtype=dtype)

        if ACT2FN is None:  # pragma: no cover - defensive, transformers ships ACT2FN
            raise RuntimeError("transformers.activations.ACT2FN is unavailable")
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard MLP forward: down(act(gate) * up)"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnstackedQwen3VLMoeSparseMoeBlock(nn.Module):
    """
    Sparse MoE block with unstacked expert weights.

    Replaces Qwen3VLMoeTextSparseMoeBlock by converting the fused 3D expert weights
    into individual nn.Linear layers organized in a ModuleList.
    """

    def __init__(
        self,
        config: "Qwen3VLMoeTextConfig",
        original_moe_block: "Qwen3VLMoeTextSparseMoeBlock",
        copy_weights: bool = False,
        expert_slices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        # Expert-parallel callers pre-load only this rank's slice of the fused 3D
        # tensors, so the generic path below would re-read (and re-deserialise) all
        # 128 experts in every rank.
        self._expert_slices = expert_slices

        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.moe_intermediate_size
        self.expert_dim = self.intermediate_size

        # Get dtype from original module
        dtype = next(original_moe_block.parameters()).dtype

        # Create new gate (don't reference original to avoid meta device issues)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False, dtype=dtype)

        # Expert-parallel sharding: this rank owns the contiguous expert range
        # [experts_start_idx, experts_end_idx); every other slot stays None so the
        # module list keeps the global expert id as its index.
        ep_size, ep_rank, expert_start, expert_end = resolve_expert_ep_range(self.num_experts)
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_start_idx = expert_start
        self.experts_end_idx = expert_end

        # Create one MLP per local expert with standard Linear layers
        self.experts = nn.ModuleList(
            [
                UnstackedQwen3VLMoeTextMLP(config, dtype=dtype) if expert_start <= expert_id < expert_end else None
                for expert_id in range(self.num_experts)
            ]
        )
        if ep_size > 1:
            get_logger().info(
                "Qwen3-VL MoE EP enabled: rank=%s/%s experts=[%s, %s) of %s",
                ep_rank,
                ep_size,
                expert_start,
                expert_end,
                self.num_experts,
            )

        if copy_weights:
            self._transform_weights_from_original(original_moe_block, in_place=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with routing logic.

        Experts are evaluated sparsely (only the ones a token routes to) and their
        weighted contributions are accumulated in float32 in ascending global expert
        id order, which keeps the summation deterministic for a given device count.
        """
        # Record input device
        input_device = hidden_states.device
        param_device = self.gate.weight.device

        # Move input to param device if needed
        if input_device != param_device:
            hidden_states = hidden_states.to(param_device)

        if self.training:
            raise NotImplementedError(
                "Training mode for unstacked MoE not implemented. "
                "This conversion is intended for inference and quantization only."
            )

        batch_size, seq_len, _ = hidden_states.shape

        # EP must first gather the full token block so that every token reaches the
        # rank owning its experts.  Skipping this gather is not an option: the ranks
        # do not advance in lockstep, so their partial tokens differ in length and
        # ``all_gather_into_tensor`` would see mismatched shapes and hang.
        use_ep = dist.is_initialized() and dist.get_world_size() > 1
        if use_ep:
            hidden_states, slice_start, slice_end = _gather_hidden_states(hidden_states)
            batch_size, seq_len, _ = hidden_states.shape
        else:
            slice_start, slice_end = 0, seq_len

        hidden_states_flat = hidden_states.view(-1, self.hidden_size)

        # Router: compute routing weights for each token
        router_logits = self.gate(hidden_states_flat)
        routing_weights = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if use_ep:
            local_partial = _accumulate_expert_range(
                self.experts,
                hidden_states_flat,
                router_indices,
                routing_weights,
                self.experts_start_idx,
                self.experts_end_idx,
            )
            final_output = _sum_in_rank_order(_gather_partials(local_partial))
        else:
            # Single card keeps the original dense path so existing quantization
            # results do not drift; only the EP path uses the sparse accumulator.
            expert_outputs = torch.stack([expert(hidden_states_flat) for expert in self.experts], dim=0)
            routing_matrix = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
            final_output = torch.einsum("be,ebh->bh", routing_matrix, expert_outputs)

        # Reshape back and keep only this rank's slice of the gathered token block
        final_output = final_output.to(hidden_states.dtype)
        final_output = final_output.view(batch_size, seq_len, self.hidden_size)
        final_output = final_output[:, slice_start:slice_end, :]

        # Move back to input device if needed
        if final_output.device != input_device:
            final_output = final_output.to(input_device)

        return final_output

    def _transform_weights_from_original(
        self, original_moe_block: "Qwen3VLMoeTextSparseMoeBlock", in_place: bool = True
    ):
        """
        Transform 3D fused weights into individual Linear layer weights.

        Original weight shapes:
            - gate_up_proj: (num_experts, hidden_size, 2 * expert_dim)
            - down_proj:    (num_experts, expert_dim, hidden_size)

        Target Linear weight shapes:
            - gate_proj.weight: (expert_dim, hidden_size)  [transposed]
            - up_proj.weight:   (expert_dim, hidden_size)  [transposed]
            - down_proj.weight: (hidden_size, expert_dim)  [transposed]
        """
        with torch.no_grad():
            # 1) gate weights: safely load from meta/offload to CPU
            with PrepareWeight(original_moe_block.gate):
                gate_weight = original_moe_block.gate.weight.data
                self.gate.weight = nn.Parameter(
                    gate_weight.contiguous().cpu(),
                    requires_grad=False,
                )

            # 2) experts' 3D weights: pre-sliced copy if the caller provided one,
            # otherwise read the fused tensor (all experts) and slice below.
            if self._expert_slices is not None:
                full_gate_up_proj, full_down_proj = self._expert_slices
                local_offset = self.experts_start_idx
            else:
                with PrepareWeight(original_moe_block.experts):
                    gate_up_param = original_moe_block.experts.gate_up_proj
                    down_param = original_moe_block.experts.down_proj

                    full_gate_up_proj = gate_up_param.data.cpu()  # (E, H, 2D)
                    full_down_proj = down_param.data.cpu()  # (E, D, H)
                local_offset = 0

            # 3) Construct each local expert's Linear weights on CPU
            for expert_idx in range(self.experts_start_idx, self.experts_end_idx):
                slice_idx = expert_idx - local_offset
                gate_up_weight = full_gate_up_proj[slice_idx]  # (H, 2D)
                down_weight = full_down_proj[slice_idx]  # (D, H)

                gate_weight, up_weight = gate_up_weight.chunk(2, dim=-1)

                self.experts[expert_idx].gate_proj.weight = nn.Parameter(
                    gate_weight.t().contiguous(), requires_grad=False
                )
                self.experts[expert_idx].up_proj.weight = nn.Parameter(up_weight.t().contiguous(), requires_grad=False)
                self.experts[expert_idx].down_proj.weight = nn.Parameter(
                    down_weight.t().contiguous(), requires_grad=False
                )

            # Release full 3D CPU tensors
            del full_gate_up_proj, full_down_proj
            # The per-expert Linear weights now own the data; holding the pre-sliced
            # source would keep another ~1.2 GiB per layer alive.
            self._expert_slices = None

        if in_place:
            # Free original 3D weights immediately to save memory
            if hasattr(original_moe_block.experts, "gate_up_proj"):
                del original_moe_block.experts.gate_up_proj
            if hasattr(original_moe_block.experts, "down_proj"):
                del original_moe_block.experts.down_proj
            gc.collect()


def convert_qwen3_moe_to_linear(
    model, config: "Qwen3VLMoeTextConfig", in_place: bool = True, verbose: bool = True
) -> None:
    """
    Convert Qwen3-VL MoE model with fused 3D expert weights to standard nn.Linear layers.

    This function identifies MoE layers in the language model and replaces them with
    unstacked versions where each expert has explicit nn.Linear layers.

    Args:
        model: Qwen3VLMoeForConditionalGeneration instance
        config: Qwen3VLMoeTextConfig or Qwen3VLMoeConfig
        in_place: If True, move weights directly to save memory; if False, copy weights
        verbose: If True, log detailed progress information
    """
    if Qwen3VLMoeTextSparseMoeBlock is None:
        raise RuntimeError(_EP_REQUIRED_MSG)

    # Handle both Qwen3VLMoeConfig and Qwen3VLMoeTextConfig
    if hasattr(config, 'text_config'):
        text_config = config.text_config
    else:
        text_config = config

    # Identify target layers (those with MoE blocks)
    target_layers = []
    for layer_idx in range(text_config.num_hidden_layers):
        # Skip layers designated as MLP-only
        if layer_idx in text_config.mlp_only_layers:
            continue
        # Only process layers at sparse step intervals
        if (layer_idx + 1) % text_config.decoder_sparse_step == 0:
            target_layers.append(layer_idx)

    if verbose:
        get_logger().info(f"Converting {len(target_layers)} MoE layers to standard Linear layers: {target_layers}")
        get_logger().info(f"Memory mode: {'in-place (memory efficient)' if in_place else 'copy (safe)'}")

    language_model = model.model.language_model if hasattr(model, 'model') else model.language_model

    for layer_idx in target_layers:
        if verbose:
            get_logger().info(f"Processing layer {layer_idx}...")

        original_moe_block = language_model.layers[layer_idx].mlp

        # Verify it's actually a MoE block
        if not isinstance(original_moe_block, Qwen3VLMoeTextSparseMoeBlock):
            get_logger().warning(
                f"Layer {layer_idx} is not a Qwen3VLMoeTextSparseMoeBlock, skipping. Got: {type(original_moe_block)}"
            )
            continue

        # Create unstacked MoE block
        unstacked_moe_block = UnstackedQwen3VLMoeSparseMoeBlock(text_config, original_moe_block, copy_weights=False)

        # Transform weights
        unstacked_moe_block._transform_weights_from_original(original_moe_block, in_place=in_place)

        # Replace the original MLP
        language_model.layers[layer_idx].mlp = unstacked_moe_block

        del original_moe_block
        gc.collect()

        if verbose:
            get_logger().info(f"✓ Layer {layer_idx} converted")

    if verbose:
        get_logger().info(
            f"Successfully converted all {len(target_layers)} MoE layers. Model is now ready for standard quantization."
        )
