# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Transformers-style Multi-Token Prediction (MTP) model for Qwen3.5.

This module implements MTP with unpacked linear layers (separate q/k/v projections)
and unpacked experts (individual nn.Linear per expert), compatible with both
dense (Qwen3_5TextConfig) and MoE (Qwen3_5MoeTextConfig) configurations.

Reuses existing transformers Qwen3.5 components where possible:
  - Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding, Qwen3_5Attention, Qwen3_5MLP
    from modeling_qwen3_5
  - Qwen3_5MoeTopKRouter from modeling_qwen3_5_moe

Only MTP-specific and unpacked-expert classes are defined here.

Reference: vLLM implementation at vllm/model_executor/models/qwen3_5_mtp.py
"""

import logging
from typing import Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from .moe_utils import Qwen3_5MoeSparseMoeBlockWithMLP

# --- Reuse existing Qwen3.5 components ---
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MTP Decoder Layer
# ---------------------------------------------------------------------------
class Qwen3_5MtpDecoderLayer(nn.Module):
    """
    MTP decoder layer: always uses full attention (never linear attention).
    Reuses Qwen3_5Attention and Qwen3_5RMSNorm from existing implementation.
    Supports both dense MLP (Qwen3_5MLP) and MoE MLP depending on config.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Reuse Qwen3_5Attention directly (already has separate q/k/v/o projections)
        self.self_attn = Qwen3_5Attention(config, layer_idx)

        # Use MoE if config has experts, otherwise dense MLP
        use_moe = hasattr(config, "num_experts") and getattr(config, "num_experts", 0) > 0
        if use_moe:
            self.mlp = Qwen3_5MoeSparseMoeBlockWithMLP(config)
        else:
            self.mlp = Qwen3_5MLP(config, config.intermediate_size)

        # Reuse Qwen3_5RMSNorm directly
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # MoE blocks may return tuples
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Multi-Token Predictor (core MTP module)
# ---------------------------------------------------------------------------
class Qwen3_5MultiTokenPredictor(nn.Module):
    """
    Core MTP module that fuses base model hidden states with input embeddings
    provided by the backbone model, then passes through decoder layer(s).

    Architecture:
        1. Normalize embeddings and base hidden states separately (Qwen3_5RMSNorm)
        2. Concatenate [norm_embedding, norm_hidden] -> shape (batch, seq, hidden*2)
        3. Project via fc to hidden_size
        4. Pass through MTP decoder layer(s) (full attention + MLP/MoE)
        5. Apply final norm (Qwen3_5RMSNorm)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        # Pre-fusion normalization (reuse Qwen3_5RMSNorm)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Fusion layer: projects concatenated [embedding, hidden] to hidden_size
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # MTP decoder layers (always full attention)
        self.layers = nn.ModuleList([Qwen3_5MtpDecoderLayer(config, layer_idx=i) for i in range(self.num_mtp_layers)])

        # Final normalization (reuse Qwen3_5RMSNorm)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embedding (reuse Qwen3_5TextRotaryEmbedding)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) - hidden states from the base model
            attention_mask: optional attention mask
            position_ids: optional position IDs (3D for mRoPE)
            past_key_values: optional cache for KV values
            inputs_embeds: backbone-provided embeddings, required
            cache_position: optional cache position indices

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        # Step 1: Get embeddings from caller/backbone
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided for Qwen3_5MultiTokenPredictor")

        # Step 2: Normalize embedding and hidden states separately
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)

        # Step 3: Concatenate and project
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)

        # Step 4: Prepare position embeddings
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            if cache_position is not None:
                position_ids = cache_position.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=hidden_states.device)
                position_ids = cache_position.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Step 5: Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        # Step 6: Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# MTP Causal LM wrapper
# ---------------------------------------------------------------------------
class Qwen3_5MtpForCausalLM(nn.Module):
    """
    Qwen3.5 Multi-Token Prediction wrapper.

    This wrapper only keeps the MTP predictor and returns MTP hidden states.
    Embedding table and LM head should be reused from the backbone model.

    Usage:
        config = Qwen3_5MoeTextConfig(...)  # or Qwen3_5TextConfig(...)
        model = Qwen3_5MtpForCausalLM(config)
        model.load_weights(checkpoint_weights)

        # Forward pass
        logits = model(input_ids, base_hidden_states)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mtp = Qwen3_5MultiTokenPredictor(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through MTP predictor.

        Args:
            hidden_states: (batch, seq_len, hidden_size) - base model hidden states
            attention_mask: optional attention mask
            position_ids: optional position IDs
            past_key_values: optional KV cache
            inputs_embeds: optional pre-computed embeddings
            cache_position: optional cache position indices

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        hidden_states = self.mtp(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        return hidden_states

    def load_weights(self, weights: dict[str, torch.Tensor] | list[tuple[str, torch.Tensor]]):
        """
        Load weights from a checkpoint, handling:
        1. 'mtp.' prefix remapping
        2. Packed expert weight unpacking (3D tensors -> individual expert nn.Linear)

        Args:
            weights: Either a dict of {name: tensor} or list of (name, tensor) pairs.
                     Weight names should be in the original checkpoint format, e.g.:
                     - 'mtp.layers.0.self_attn.q_proj.weight'
                     - 'mtp.layers.0.mlp.experts.gate_up_proj'
        """
        if isinstance(weights, dict):
            weight_items = weights.items()
        else:
            weight_items = weights

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        num_experts = getattr(self.config, "num_experts", 0)

        for name, loaded_weight in weight_items:
            # --- Remap weight names ---
            mapped_name = self._remap_weight_name(name)
            if mapped_name is None:
                continue

            # --- Handle packed expert weights ---
            if "experts.gate_up_proj" in mapped_name and num_experts > 0:
                self._load_packed_gate_up_experts(mapped_name, loaded_weight, params_dict, loaded_params)
                continue

            if "experts.down_proj" in mapped_name and num_experts > 0 and "shared_expert" not in mapped_name:
                # Check if this is the packed 3D expert down_proj (not shared_expert.down_proj)
                if loaded_weight.dim() == 3:
                    self._load_packed_down_experts(mapped_name, loaded_weight, params_dict, loaded_params)
                    continue

            # --- Skip rotary embedding buffers ---
            if "rotary_emb.inv_freq" in mapped_name:
                continue

            # --- Direct loading ---
            if mapped_name not in params_dict:
                logger.warning("Parameter %s (from %s) not found, skipping", mapped_name, name)
                continue

            param = params_dict[mapped_name]
            if param.shape != loaded_weight.shape:
                logger.warning(
                    "Shape mismatch for %s: expected %s, got %s, skipping",
                    mapped_name,
                    param.shape,
                    loaded_weight.shape,
                )
                continue

            param.data.copy_(loaded_weight)
            loaded_params.add(mapped_name)

        logger.info("Loaded %d parameters for MTP model", len(loaded_params))
        return loaded_params

    def _remap_weight_name(self, name: str) -> Optional[str]:
        """
        Remap checkpoint weight names to model parameter names.

        Checkpoint format -> Model format:
            mtp.fc.weight -> mtp.fc.weight
            mtp.pre_fc_norm_embedding.weight -> mtp.pre_fc_norm_embedding.weight
            mtp.pre_fc_norm_hidden.weight -> mtp.pre_fc_norm_hidden.weight
            mtp.layers.0.* -> mtp.layers.0.*
            mtp.norm.weight -> mtp.norm.weight
        """
        if name.startswith("mtp."):
            # MTP weights map directly
            return name
        else:
            # Skip weights that don't belong to MTP
            return None

    def _load_packed_gate_up_experts(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        loaded_params: set,
    ):
        """
        Unpack gate_up_proj from 3D tensor to individual expert gate_proj and up_proj.

        Checkpoint format:
            experts.gate_up_proj: shape [num_experts, 2 * moe_intermediate_size, hidden_size]

        Split into per-expert:
            experts.{i}.gate_proj.weight: shape [moe_intermediate_size, hidden_size]
            experts.{i}.up_proj.weight: shape [moe_intermediate_size, hidden_size]
        """
        num_experts = loaded_weight.shape[0]

        # Find the layer prefix: e.g. "mtp.layers.0.mlp.experts.experts."
        # The model path has double "experts" because SparseMoeBlock.experts is a
        # Qwen3_5MtpExperts which itself has a ModuleList named .experts
        prefix = name.replace("experts.gate_up_proj", "experts.experts.")

        for expert_idx in range(num_experts):
            gate_up_weight = loaded_weight[expert_idx]  # (2 * intermediate, hidden)
            gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)

            gate_name = f"{prefix}{expert_idx}.gate_proj.weight"
            up_name = f"{prefix}{expert_idx}.up_proj.weight"

            if gate_name in params_dict:
                params_dict[gate_name].data.copy_(gate_weight)
                loaded_params.add(gate_name)
            else:
                logger.warning("Parameter %s not found, skipping", gate_name)

            if up_name in params_dict:
                params_dict[up_name].data.copy_(up_weight)
                loaded_params.add(up_name)
            else:
                logger.warning("Parameter %s not found, skipping", up_name)

    def _load_packed_down_experts(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        loaded_params: set,
    ):
        """
        Unpack down_proj from 3D tensor to individual expert down_proj.

        Checkpoint format:
            experts.down_proj: shape [num_experts, hidden_size, moe_intermediate_size]

        Split into per-expert:
            experts.{i}.down_proj.weight: shape [hidden_size, moe_intermediate_size]
        """
        num_experts = loaded_weight.shape[0]

        # Find the layer prefix: e.g. "mtp.layers.0.mlp.experts.experts."
        prefix = name.replace("experts.down_proj", "experts.experts.")

        for expert_idx in range(num_experts):
            down_weight = loaded_weight[expert_idx]  # (hidden, intermediate)

            down_name = f"{prefix}{expert_idx}.down_proj.weight"

            if down_name in params_dict:
                params_dict[down_name].data.copy_(down_weight)
                loaded_params.add(down_name)
            else:
                logger.warning("Parameter %s not found, skipping", down_name)
