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
# pylint: disable=consider-using-from-import,logging-fstring-interpolation,pointless-string-statement

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

__all__ = [
    'UnstackedQwen3VLMoeTextMLP',
    'UnstackedQwen3VLMoeSparseMoeBlock',
    'convert_qwen3_moe_to_linear',
]

import gc  # noqa: E402
from typing import Optional  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from msmodelslim.utils.logging import get_logger  # noqa: E402
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import PrepareWeight  # noqa: E402

try:
    from transformers.activations import ACT2FN
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextSparseMoeBlock,
    )
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeTextConfig,
    )
except ImportError as e:
    get_logger().warning(f"Failed to import Qwen3VLMoe modules: {e}")
    ACT2FN = None
    Qwen3VLMoeTextSparseMoeBlock = None
    Qwen3VLMoeTextConfig = None


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
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.moe_intermediate_size
        self.expert_dim = self.intermediate_size

        # Get dtype from original module
        dtype = next(original_moe_block.parameters()).dtype

        # Create new gate (don't reference original to avoid meta device issues)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False, dtype=dtype)

        # Create num_experts individual MLPs with standard Linear layers
        self.experts = nn.ModuleList([UnstackedQwen3VLMoeTextMLP(config, dtype=dtype) for _ in range(self.num_experts)])

        if copy_weights:
            self._transform_weights_from_original(original_moe_block, in_place=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with routing logic.

        For inference, we compute all expert outputs and weight them by routing scores.
        """
        # Record input device
        input_device = hidden_states.device
        param_device = self.gate.weight.device

        # Move input to param device if needed
        if input_device != param_device:
            hidden_states = hidden_states.to(param_device)

        batch_size, seq_len, _ = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)

        if self.training:
            raise NotImplementedError(
                "Training mode for unstacked MoE not implemented. "
                "This conversion is intended for inference and quantization only."
            )

        # Router: compute routing weights for each token
        router_logits = self.gate(hidden_states_flat)
        routing_weights = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Compute all expert outputs
        expert_outputs = torch.stack([expert(hidden_states_flat) for expert in self.experts], dim=0)

        # Weight expert outputs by routing scores
        routing_matrix = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)

        # Weighted sum
        final_output = torch.einsum('be,ebh->bh', routing_matrix, expert_outputs)

        # Reshape back
        final_output = final_output.view(batch_size, seq_len, self.hidden_size)

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

            # 2) experts' 3D weights: safely load to CPU
            with PrepareWeight(original_moe_block.experts):
                gate_up_param = original_moe_block.experts.gate_up_proj
                down_param = original_moe_block.experts.down_proj

                full_gate_up_proj = gate_up_param.data.cpu()  # (E, H, 2D)
                full_down_proj = down_param.data.cpu()  # (E, D, H)

            # 3) Construct each expert's Linear weights on CPU
            for expert_idx in range(self.num_experts):
                gate_up_weight = full_gate_up_proj[expert_idx]  # (H, 2D)
                down_weight = full_down_proj[expert_idx]  # (D, H)

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
