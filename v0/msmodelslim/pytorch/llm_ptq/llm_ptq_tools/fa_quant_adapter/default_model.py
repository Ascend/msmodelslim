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


import logging
import warnings
from importlib import import_module
from typing import List, Optional, Tuple

import torch
from transformers import Cache, PretrainedConfig

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import PrepareWeight
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.fa_quant_adapter import (
    AttentionType,
    ForwardFactory,
)


def _install_forward_adapter(
    module: torch.nn.Module,
    module_type: str,
    adapter_type: str,
    module_name: str,
    logger: logging.Logger,
):
    """安装forward适配器"""
    try:
        forward_adapter = ForwardFactory.get_forward_adapter(module_type, adapter_type)
        if hasattr(module, "forward"):
            module.forward = forward_adapter(module.forward).__get__(module, module.__class__)
        else:
            raise ValueError(f"Module {module_name} has no forward method to adapt")

        logger.info(f"Successfully installed FAQuantizer for module {module_name}")
    except Exception as e:
        logger.error(f"Failed to install FAQuantizer for module {module_name}: {str(e)}")
        raise


def install_for_default_model(
    model: torch.nn.Module,
    config: PretrainedConfig,
    logger: logging.Logger,
    skip_layers: List[str],
):
    """为默认模型安装量化器"""
    for name, module in model.named_modules():
        if "Attention" in module.__class__.__name__:
            if any(skip_name in name for skip_name in skip_layers):
                logger.info(f"Skipping FAQuantizer installation for module {name}")
                continue

            if hasattr(module, "fa_quantizer"):
                logger.warning(f"Module {name} already has FAQuantizer installed.")
                continue
            
            module.fa_quantizer = FAQuantizer(config, logger=logger)

            default_attn_type = {
                "deepseekv2": AttentionType.MLA,
                "deepseek_v2": AttentionType.MLA,
                "deepseekv3": AttentionType.MLA,
                "deepseek_v3": AttentionType.MLA,
            }

            attn_type = default_attn_type.get(config.model_type, AttentionType.MHA).value
            if attn_type:
                _install_forward_adapter(module, config.model_type, attn_type, name, logger)


@ForwardFactory.register("deepseekv3", "mla")
@ForwardFactory.register("deepseek_v3", "mla")
@ForwardFactory.register("deepseekv2", "mla")
@ForwardFactory.register("deepseek_v2", "mla")
def deepseekv2_mla_forward_adapter(original_forward):
    """DeepSeek V2/V3模型的MLA forward适配器"""

    deepseek_module = import_module(original_forward.__module__)
    apply_rotary_pos_emb = deepseek_module.apply_rotary_pos_emb

    def new_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37.\n"
                "Please make sure to use `attention_mask` instead."
            )
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv_seq_len = k_pe.shape[-2]

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            compressed_kv = compressed_kv.unsqueeze(1)
            k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
            compressed_kv = compressed_kv.squeeze(1)

        with PrepareWeight(self.kv_b_proj):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)

        q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
        out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

        q_nope = torch.matmul(q_nope, q_absorb)

        # ----------FA3-------------
        q_nope = self.fa_quantizer.quant(q_nope, qkv="q")
        compressed_kv = self.fa_quantizer.quant(compressed_kv.unsqueeze(1), qkv="k").squeeze(1)
        _ = self.fa_quantizer.quant(compressed_kv.unsqueeze(1), qkv="v").squeeze(1)
        # ----------FA3-------------

        attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT))
        attn_weights = attn_weights * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is None:
            raise ValueError("Attention mask cannot be None")
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_pe.dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
        attn_output = torch.matmul(attn_output, out_absorb.mT)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return new_forward