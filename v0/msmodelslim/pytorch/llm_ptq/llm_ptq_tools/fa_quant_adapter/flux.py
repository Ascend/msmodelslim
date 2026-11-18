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
from importlib import import_module
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.fa_quant_adapter import ForwardFactory


def get_flux_config(config: PretrainedConfig, logger: logging.Logger):
    sp_size = 1 if not config.is_tp else dist.get_world_size()

    config_dict = {
        'num_attention_heads': config.num_attention_heads // sp_size, 
        'hidden_size': config.attention_head_dim * config.num_attention_heads,
        'num_key_value_heads': config.num_attention_heads // sp_size,
    }
    return SimpleNamespace(**config_dict)


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
        
        processor_cls = module.processor.__class__
        original_call = module.processor.__call__
        new_call = forward_adapter(original_call).__get__(
            module.processor, processor_cls
        )
        processor_cls.__call__ = new_call
        
        logger.info(f"Successfully installed FAQuantizer for module {module_name}")
    except Exception as e:
        logger.error(f"Failed to install FAQuantizer for module {module_name}: {str(e)}")
        raise


def install_for_flux_model(
    model: torch.nn.Module,
    config: PretrainedConfig,
    logger: logging.Logger,
    skip_layers: List[str],
):
    """为Flux模型安装量化器"""
    processor_map = {
        "FluxTransformerBlock": "FluxAttnProcessor2_0",
        "FluxSingleTransformerBlock": "FluxSingleAttnProcessor2_0"
    }

    for name, module in model.named_modules():
        module_type = module.__class__.__name__
    
        if module_type not in processor_map:
            continue
            
        attn_module = module.attn
        attn_full_name = f"{name}.attn"
        
        if any(skip_name in attn_full_name for skip_name in skip_layers):
            logger.info(f"Skipping {attn_full_name}")
            continue

        if hasattr(attn_module, "fa_quantizer"):
            logger.warning(f"Module {attn_full_name} already has FAQuantizer installed.")
            continue
        
        flux_config = get_flux_config(config, logger)
        attn_module.fa_quantizer = FAQuantizer(flux_config, logger=logger)
        logger.info(f"Installed quantizer at {attn_full_name}")

        processor_type = processor_map.get(module_type, None)
        if processor_type:
            _install_forward_adapter(attn_module, module_type, processor_type, attn_full_name, logger)


# 全局变量用于保存第一次获取的函数
FLUX_APPLY_ROTARY_EMB_MINDSPEED = None
FLUX_APPLY_FA = None
FLUX_ATTENTION = None


# 注册 Flux 模型的适配器
@ForwardFactory.register("FluxTransformerBlock", "FluxAttnProcessor2_0")
def flux_attn_processor_adapter(original_call):
    """FluxAttnProcessor2_0 的量化适配器"""
    global FLUX_APPLY_ROTARY_EMB_MINDSPEED, FLUX_APPLY_FA, FLUX_ATTENTION

    # 如果已经初始化过，直接使用保存的函数
    if all([FLUX_APPLY_ROTARY_EMB_MINDSPEED, FLUX_APPLY_FA, FLUX_ATTENTION]):
        apply_rotary_emb_mindspeed = FLUX_APPLY_ROTARY_EMB_MINDSPEED
        apply_fa = FLUX_APPLY_FA
        Attention = FLUX_ATTENTION
    else:
        # 第一次调用，从模块中获取并保存
        flux_attn_processor_module = import_module(original_call.__module__)
        
        apply_rotary_emb_mindspeed = flux_attn_processor_module.apply_rotary_emb_mindspeed
        apply_fa = flux_attn_processor_module.apply_fa
        Attention = flux_attn_processor_module.Attention
        
        # 保存到全局变量
        FLUX_APPLY_ROTARY_EMB_MINDSPEED = apply_rotary_emb_mindspeed
        FLUX_APPLY_FA = apply_fa
        FLUX_ATTENTION = Attention
    
    def new_call(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        if attn.is_tp:
            attn_heads = attn.heads // attn.world_size
        else:
            attn_heads = attn.heads
        head_dim = inner_dim // attn_heads

        query = query.view(batch_size, -1, attn_heads, head_dim)
        key = key.view(batch_size, -1, attn_heads, head_dim)
        value = value.view(batch_size, -1, attn_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn_heads, head_dim
        )
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn_heads, head_dim
        )
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn_heads, head_dim
        )

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb_mindspeed(query, image_rotary_emb)
            key = apply_rotary_emb_mindspeed(key, image_rotary_emb)

        # --------------------fa3-----------------------------
        query = attn.fa_quantizer.quant(query, qkv="q")
        key = attn.fa_quantizer.quant(key, qkv="k")
        value = attn.fa_quantizer.quant(value, qkv="v")
        # --------------------fa3-----------------------------
        
        hidden_states = apply_fa(query, key, value, attention_mask)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if attn.is_tp:
            dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        if attn.is_tp:
            dist.all_reduce(encoder_hidden_states, op=dist.ReduceOp.SUM)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states
    return new_call


# 全局变量用于保存第一次获取的函数
FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED = None
FLUX_SINGLE_APPLY_FA = None
FLUX_SINGLE_ATTENTION = None


# 注册 Flux 模型的适配器
@ForwardFactory.register("FluxSingleTransformerBlock", "FluxSingleAttnProcessor2_0")
def flux_attn_single_processor_adapter(original_call):
    """FluxSingleAttnProcessor2_0 的量化适配器"""
    global FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED, FLUX_SINGLE_APPLY_FA, FLUX_SINGLE_ATTENTION

    # 如果已经初始化过，直接使用保存的函数
    if all([FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED, FLUX_SINGLE_APPLY_FA, FLUX_SINGLE_ATTENTION]):
        apply_rotary_emb_mindspeed = FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED
        apply_fa = FLUX_SINGLE_APPLY_FA
        Attention = FLUX_SINGLE_ATTENTION
    else:
        # 第一次调用，从模块中获取并保存
        flux_attn_processor_module = import_module(original_call.__module__)
        
        apply_rotary_emb_mindspeed = flux_attn_processor_module.apply_rotary_emb_mindspeed
        apply_fa = flux_attn_processor_module.apply_fa
        Attention = flux_attn_processor_module.Attention
        
        # 保存到全局变量
        FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED = apply_rotary_emb_mindspeed
        FLUX_SINGLE_APPLY_FA = apply_fa
        FLUX_SINGLE_ATTENTION = Attention

    def new_call(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        if attn.is_tp:
            attn_heads = attn.heads // attn.world_size
        else:
            attn_heads = attn.heads
        head_dim = inner_dim // attn_heads

        query = query.view(batch_size, -1, attn_heads, head_dim)

        key = key.view(batch_size, -1, attn_heads, head_dim)
        value = value.view(batch_size, -1, attn_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb_mindspeed(query, image_rotary_emb)
            key = apply_rotary_emb_mindspeed(key, image_rotary_emb)

        # --------------------fa3-----------------------------
        query = attn.fa_quantizer.quant(query, qkv="q")
        key = attn.fa_quantizer.quant(key, qkv="k")
        value = attn.fa_quantizer.quant(value, qkv="v")
        # --------------------fa3-----------------------------

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = apply_fa(query, key, value, attention_mask)
        hidden_states = hidden_states.to(query.dtype)
        B, S, H = hidden_states.shape
        if attn.is_tp:
            hidden_states_full = torch.empty(
                [attn.world_size, B, S, H], dtype=hidden_states.dtype, device=hidden_states.device
                )
            dist.all_gather_into_tensor(hidden_states_full, hidden_states)
            hidden_states = hidden_states_full.permute(1, 2, 0, 3).reshape([B, S, 2 * H])

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states
    return new_call