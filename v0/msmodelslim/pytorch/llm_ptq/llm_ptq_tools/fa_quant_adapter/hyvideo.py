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


def get_hyvideo_config(config: PretrainedConfig, logger: logging.Logger):
    if dist.is_initialized():
        sp_size = dist.get_world_size()
        logger.info(f"sp_size: {sp_size}")
    else:
        logger.info("sp_size = 1 (not in distributed environment)")
        sp_size = 1

    config_dict = {
        'num_attention_heads': config.heads_num // sp_size, 
        'hidden_size': config.hidden_size // sp_size,
        'num_key_value_heads': config.heads_num // sp_size,
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
        original_method = getattr(module, adapter_type)
        adapted_method = forward_adapter(original_method).__get__(module, module.__class__)
        setattr(module, adapter_type, adapted_method)
        logger.info(f"Successfully installed FAQuantizer for module {module_name}")
    except Exception as e:
        logger.error(f"Failed to install FAQuantizer for module {module_name}: {str(e)}")
        raise


def install_for_hyvideo_model(
    model: torch.nn.Module,
    config: PretrainedConfig,
    logger: logging.Logger,
    skip_layers: List[str],
):
    """为HYVideo模型安装量化器"""
    module_map = {
        "MMDoubleStreamBlock": "double_forward",
        "MMSingleStreamBlock": "single_forward"
    }

    for name, module in model.named_modules():
        module_type = module.__class__.__name__
    
        if module_type not in module_map:
            continue
            
        if any(skip_name in name for skip_name in skip_layers):
            logger.info(f"Skipping FAQuantizer installation for module {name}")
            continue

        if hasattr(module, "fa_quantizer"):
            logger.warning(f"Module {name} already has FAQuantizer installed.")
            continue

        hunyuan_config = get_hyvideo_config(config, logger)
        module.fa_quantizer = FAQuantizer(hunyuan_config, logger=logger)
        logger.info(f"Installed quantizer at {name}")

        forward_type = module_map.get(module_type, None)
        if forward_type:
            _install_forward_adapter(module, module_type, forward_type, name, logger)


# 注册 HYVideo 模型的适配器
@ForwardFactory.register("MMDoubleStreamBlock", "double_forward")
def hyvideo_mm_double_stream_block_double_forward_adapter(original_forward):
    """HYVideo 模型的double_forward适配器"""
        
    hyvideo_double_module = import_module(original_forward.__module__)
    modulate = hyvideo_double_module.modulate
    rearrange = hyvideo_double_module.rearrange
    apply_rotary_emb = hyvideo_double_module.apply_rotary_emb
    attention = hyvideo_double_module.attention
    parallel_attention = hyvideo_double_module.parallel_attention

    def new_double_forward(
            self,
            img, txt,
            img_mod1_shift,
            img_mod1_scale,
            txt_mod1_shift,
            txt_mod1_scale,
            freqs_cis,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv
        ):
        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            if not (img_qq.shape == img_q.shape and img_kk.shape == img_k.shape):
                raise ValueError(
                    f"Rotary embedding output shape mismatch. "
                    f"img_qq shape: {img_qq.shape}, img_q shape: {img_q.shape}, "
                    f"img_kk shape: {img_kk.shape}, img_k shape: {img_k.shape}"
                )
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        expected_cu_seqlens_q_length = 2 * img.shape[0] + 1
        if cu_seqlens_q.shape[0] != expected_cu_seqlens_q_length:
            raise ValueError(
                f"cu_seqlens_q shape mismatch: "
                f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
                f"expected first dimension length: {expected_cu_seqlens_q_length}"
            )

        # --------------------fa3-----------------------------
        q = self.fa_quantizer.quant(q, qkv="q")
        k = self.fa_quantizer.quant(k, qkv="k")
        v = self.fa_quantizer.quant(v, qkv="v")
        # --------------------fa3-----------------------------

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                mode="torch",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                scale=self.scale
            )
        
        return attn
    return new_double_forward


@ForwardFactory.register("MMSingleStreamBlock", "single_forward")
def hyvideo_mm_single_stream_block_single_forward_adapter(original_forward):
    """HYVideo 模型的 single_forward 适配器"""

    hyvideo_single_module = import_module(original_forward.__module__)
    rearrange = hyvideo_single_module.rearrange
    apply_rotary_emb = hyvideo_single_module.apply_rotary_emb
    attention = hyvideo_single_module.attention
    parallel_attention = hyvideo_single_module.parallel_attention

    def new_single_forward(
            self,
            qkv,
            freqs_cis,
            txt_len,
            x,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv
        ):
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            if not (img_qq.shape == img_q.shape and img_kk.shape == img_k.shape):
                raise ValueError(
                    f"Rotary embedding output shape mismatch. "
                    f"img_qq shape: {img_qq.shape}, img_q shape: {img_q.shape}, "
                    f"img_kk shape: {img_kk.shape}, img_k shape: {img_k.shape}"
                )
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        expected_cu_seqlens_q_length = 2 * x.shape[0] + 1
        if cu_seqlens_q.shape[0] != expected_cu_seqlens_q_length:
            raise ValueError(
                f"cu_seqlens_q shape mismatch. "
                f"cu_seqlens_q.shape: {cu_seqlens_q.shape}, x.shape[0]: {x.shape[0]}, "
                f"expected first dimension length: {expected_cu_seqlens_q_length}"
            )

        # --------------------fa3-----------------------------
        q = self.fa_quantizer.quant(q, qkv="q")
        k = self.fa_quantizer.quant(k, qkv="k")
        v = self.fa_quantizer.quant(v, qkv="v")
        # --------------------fa3----------------------------- 

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                mode="torch",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                scale=self.scale
            )
        # attention computation end
        return attn       
    return new_single_forward