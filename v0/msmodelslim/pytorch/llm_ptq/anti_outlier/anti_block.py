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

import math
import copy
from typing import Optional, Tuple
import torch
from torch import nn
from easydict import EasyDict
from msmodelslim import logger as msmodelslim_logger

try:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import migration, migration_vit
except ImportError:
    migration, migration_vit = None, None
    msmodelslim_logger.warning(
        "The current CANN version does not support importing the migration and migration_vit packages."
    )


def check_migration_import(migration_import):
    if migration_import is None:
        return False
    return True


def get_config():
    a_qconfig = {
        'quantizer': 'FixedFakeQuantize',
        'bit': 8,
        'symmetric': False,
        'ch_axis': -1,

    }
    w_qconfig = {
        'quantizer': 'FixedFakeQuantize',
        'bit': 8,
        'symmetric': True,
        'ch_axis': 0,
    }
    return EasyDict(a_qconfig), EasyDict(w_qconfig)


def check_internvl2_8b_model(cfg):
    internvl2_8b_vision_layers = 24
    internvl2_8b_llm_layers = 32
    if cfg.vision_config.num_hidden_layers == internvl2_8b_vision_layers:
        if cfg.llm_config.num_hidden_layers == internvl2_8b_llm_layers:
            return True
    return False


class QuantV2QwenBlock(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.self_attn = org_layer.attn
        self.mlp = org_layer.mlp
        self.input_layernorm = org_layer.ln_1
        self.post_attention_layernorm = org_layer.ln_2
        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True

        self.layername = layername

    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if not check_migration_import(migration):
            raise ImportError("The current CANN version does not support migration algorithm.")

        hidden_states = args[0]
        rotary_pos_emb = kwargs.pop("rotary_pos_emb")
        registered_causal_mask = kwargs.pop("registered_causal_mask")
        layer_past = kwargs.pop("layer_past")
        attention_mask = kwargs.pop("attention_mask")
        head_mask = kwargs.pop("head_mask")
        use_cache = kwargs.pop("use_cache")
        output_attentions = kwargs.pop("output_attentions")

        org_hidden_states = copy.deepcopy(hidden_states)
        layernorm_output = self.input_layernorm(org_hidden_states)
 
        if self.cac_migrate_attn:
            msmodelslim_logger.info("current block is QuantV2QwenBlock , layername:`{}` ".format(self.layername))
            weight_all = self.self_attn.c_attn.weight
            bias_list = None
            if self.self_attn.c_attn.bias is not None:
                bias_list = torch.cat([self.self_attn.c_attn.bias])

            extra_dict = {
                'split_size': self.self_attn.split_size,
                'num_heads': self.self_attn.num_heads,
                'head_dim': self.self_attn.head_dim,
                'scale_attn_weights': self.self_attn.scale_attn_weights,
                'head_mask': head_mask,
                'observation_mask': None,
                'attention_mask': attention_mask,
            }
            # update scale
            a_qconfig, w_qconfig = get_config()
            best_scale = \
                migration(layernorm_output, weight_all, a_qconfig, w_qconfig, 'qkv', extra_dict, bias=bias_list)
            layernorm_output /= best_scale
            self.self_attn.c_attn.weight.data *= best_scale

            self.input_layernorm.weight.data = self.input_layernorm.weight.data.to('npu')
            self.input_layernorm.weight.data /= best_scale
            self.cac_migrate_attn = False
        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=layernorm_output,
            rotary_pos_emb=rotary_pos_emb,
            registered_causal_mask=registered_causal_mask,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs_tmp = attn_outputs[1:]
        residual = hidden_states

        layernorm_input = residual + attn_output
        
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        residual = layernorm_input

        if self.cac_migrate_mlp:            
            weight_list = torch.cat([self.mlp.w2.weight,  # gate_proj
                                     self.mlp.w1.weight]) # up_proj
            extra_dict = {
                'observation_mask': None, 
            }

            a_qconfig, w_qconfig = get_config()

            best_scale = \
                migration(layernorm_output, weight_list, a_qconfig, w_qconfig, 'up_and_gate', extra_dict)
            # update scale
            layernorm_output /= best_scale
            self.mlp.w1.weight.data *= best_scale
            self.mlp.w2.weight.data *= best_scale
            self.post_attention_layernorm.weight.data /= best_scale
            self.cac_migrate_mlp = False

        mlp_output = self.mlp(layernorm_output)
        
        hidden_states = residual + mlp_output
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += outputs_tmp
        else:
            outputs += outputs_tmp[1:]
        return outputs 


class QuantVisualAttentionBlock(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.ln_1 = org_layer.ln_1
        self.ln_2 = org_layer.ln_2
        self.attn = org_layer.attn
        self.mlp = org_layer.mlp
        self.ln_1_kv = org_layer.ln_1_kv if hasattr(org_layer, 'ln_1_kv') else None

        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True
        self.layername = layername

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if not check_migration_import(migration_vit):
            raise ImportError("The current CANN version does not support migration_vit algorithm.")

        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        post_ln_1 = self.ln_1(q_x)
         
        if self.cac_migrate_attn:
            msmodelslim_logger.info(
                f"current block is QuantVisualAttentionBlock , "
                f"layername:`{self.layername}` "
            )
            channel_max = post_ln_1.max(0)[0].max(0)[0]
            channel_min = post_ln_1.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            post_ln_1 -= shift
            if (self.attn.in_proj.bias is None):
                msmodelslim_logger.warning("attn.in_proj.bias is None")
            self.attn.in_proj.bias.data += shift @ self.attn.in_proj.weight.data.T
                
            # calculate scale
            weight_list = torch.cat([self.attn.in_proj.weight])
            extra_dict = {
                'hidden_size_per_partition': self.attn.hidden_size_per_partition,
                'norm_factor': self.attn.norm_factor,
                'hidden_size_per_attention_head': self.attn.hidden_size_per_attention_head,
                'num_attention_heads_per_partition': self.attn.num_attention_heads_per_partition,
                'attn_mask': attn_mask,
                'bias': torch.cat([self.attn.in_proj.bias]),
                'shift': shift,
            }

            a_qconfig, w_qconfig = get_config()

            # update scale
            best_scale = \
                migration_vit(post_ln_1, weight_list, a_qconfig, w_qconfig, 'vit_qkv_function', extra_dict)
            post_ln_1 /= best_scale
            ## linear and ln
            self.attn.in_proj.weight.data *= best_scale
            self.ln_1.bias.data -= shift
            self.ln_1.weight.data /= best_scale
            self.ln_1.bias.data /= best_scale
            self.cac_migrate_attn = False
            

        x = q_x + self.attention(q_x=post_ln_1, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        
        post_ln_2 = self.ln_2(x)
        
        if self.cac_migrate_mlp:
            channel_max = post_ln_2.max(0)[0].max(0)[0]
            channel_min = post_ln_2.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            post_ln_2 -= shift
            if (self.mlp.c_fc.bias is None):
                msmodelslim_logger.warning("mlp.c_fc.bias is None")
            self.mlp.c_fc.bias.data += shift @ self.mlp.c_fc.weight.data.T
            # calculate scale
            weight_list = torch.cat([self.mlp.c_fc.weight])
            extra_dict = {
                'bias': torch.cat([self.mlp.c_fc.bias]),
                'shift': shift,
                'observation_mask': None  # test
                }

            a_qconfig, w_qconfig = get_config()

            # update scale
            best_scale = \
                migration_vit(post_ln_2, weight_list, a_qconfig, w_qconfig, 'c_fc', extra_dict)
            post_ln_2 /= best_scale
            ## linear and ln
            self.mlp.c_fc.weight.data *= best_scale
            self.ln_2.bias.data -= shift
            self.ln_2.weight.data /= best_scale
            self.ln_2.bias.data /= best_scale
            self.cac_migrate_mlp = False

        x = x + self.mlp(post_ln_2)
        return x
       

class LlavaQuantDecoder(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.self_attn = org_layer.self_attn 
        self.mlp = org_layer.mlp
        self.input_layernorm = org_layer.input_layernorm 
        self.post_attention_layernorm = org_layer.post_attention_layernorm
        self.act_fn = org_layer.mlp.act_fn

        self.layername = layername

        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True

        self.cfg = cfg
        if hasattr(self.cfg, "llm_config"):
            self.cfg = cfg.llm_config

    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if not check_migration_import(migration):
            raise ImportError("The current CANN version does not support migration algorithm.")
        
        hidden_states = args[0]
        attention_mask = kwargs.pop("attention_mask")
        position_ids = kwargs.pop("position_ids")
        past_key_value = kwargs.pop("past_key_value")
        output_attentions = kwargs.pop("output_attentions")
        use_cache = kwargs.pop("use_cache")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states_device = hidden_states.device
        if self.cac_migrate_attn:
            msmodelslim_logger.info("current block is LlavaQuantDecoder , layername:`{}` ".format(self.layername))
            weight_list = torch.cat([self.self_attn.q_proj.weight,
                                     self.self_attn.k_proj.weight,
                                     self.self_attn.v_proj.weight])
            bias_list = None
            cos_cached = None
            sin_cached = None
            if hasattr(self.self_attn.rotary_emb, "cos_cached") and hasattr(self.self_attn.rotary_emb, "sin_cached"):
                cos_cached = self.self_attn.rotary_emb.cos_cached
                sin_cached = self.self_attn.rotary_emb.sin_cached
            else:
                try:
                    if self.cfg.num_attention_heads == 0:
                        raise ValueError("num_attention_heads can not be zero in model config.")
                    dim = self.cfg.hidden_size // self.cfg.num_attention_heads
                    max_position_embeddings = self.cfg.max_position_embeddings
                    base = self.cfg.rope_theta
                    device = hidden_states.device
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
                    t = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
                    freqs = torch.outer(t, inv_freq)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    cos_cached = emb.cos().to(torch.get_default_dtype())
                    sin_cached = emb.sin().to(torch.get_default_dtype())
                except Exception as e:
                    raise AttributeError("The model config has no attribute hidden_size or num_attention_heads" + 
                                        "or max_position_embeddings or rope_theta," + 
                                        "please check transformers version and model config.") from e
            if cos_cached is None or sin_cached is None:
                raise ValueError("cos_cached or sin_cached is None," + \
                                 "please check transformers version and model config.")
            extra_dict = {
                'num_heads': self.self_attn.num_heads,
                'num_key_value_heads': self.self_attn.num_key_value_heads,
                'num_key_value_groups': self.self_attn.num_key_value_groups,
                'cos_cached': cos_cached,
                'sin_cached': sin_cached,
                'head_dim': self.self_attn.head_dim,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'observation_mask': None

            }
            a_qconfig, w_qconfig = get_config()
            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'llama_qkv', extra_dict, bias=bias_list)
            hidden_states /= best_scale
            self.self_attn.q_proj.weight.data *= best_scale
            self.self_attn.k_proj.weight.data *= best_scale
            self.self_attn.v_proj.weight.data *= best_scale
            self.input_layernorm.weight.data = self.input_layernorm.weight.data.to(hidden_states_device)
            self.input_layernorm.weight.data /= best_scale
            self.cac_migrate_attn = False

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.cac_migrate_mlp:

            weight_list = torch.cat([self.mlp.gate_proj.weight,
                                     self.mlp.up_proj.weight])
            
            extra_dict = {
                'observation_mask': None, 
                'act_fn': self.act_fn
            }

            a_qconfig, w_qconfig = get_config()

            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'up_and_gate', extra_dict)
            # update scale
            hidden_states /= best_scale
            self.mlp.gate_proj.weight.data *= best_scale
            self.mlp.up_proj.weight.data *= best_scale
            self.post_attention_layernorm.weight.data /= best_scale
            self.cac_migrate_mlp = False

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
     
        return outputs


class LlavaClipVision(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.layer_norm1 = org_layer.layer_norm1
        self.layer_norm2 = org_layer.layer_norm2
        self.self_attn = org_layer.self_attn 
        self.mlp = org_layer.mlp
        self.act_fn = org_layer.mlp.activation_fn
        self.layername = layername

        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        if not check_migration_import(migration):
            raise ImportError("The current CANN version does not support migration algorithm.")

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.cac_migrate_attn:
            msmodelslim_logger.info("current block is LlavaClipVision , layername:`{}` ".format(self.layername))
            channel_max = hidden_states.max(0)[0].max(0)[0]
            channel_min = hidden_states.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            hidden_states -= shift

            if hasattr(self.self_attn.q_proj, 'bias') and self.self_attn.q_proj.bias is not None:
                self.self_attn.q_proj.bias.data += shift @ self.self_attn.q_proj.weight.data.T
            if hasattr(self.self_attn.k_proj, 'bias') and self.self_attn.k_proj.bias is not None:
                self.self_attn.k_proj.bias.data += shift @ self.self_attn.k_proj.weight.data.T
            if hasattr(self.self_attn.v_proj, 'bias') and self.self_attn.v_proj.bias is not None:
                self.self_attn.v_proj.bias.data += shift @ self.self_attn.v_proj.weight.data.T

            # calculate scale
            weight_list = torch.cat([
                self.self_attn.q_proj.weight, 
                self.self_attn.k_proj.weight, 
                self.self_attn.v_proj.weight]
            )
            bias_list = torch.cat([
                self.self_attn.q_proj.bias, 
                self.self_attn.k_proj.bias, 
                self.self_attn.v_proj.bias]
            )
            

            extra_dict = {
                'split_size': self.self_attn.embed_dim,
                'num_heads': self.self_attn.num_heads,
                'head_dim': self.self_attn.head_dim,
                'causal_attention_mask': None,
                'observation_mask': None,
                'attention_mask': attention_mask,
                'bias': bias_list,
            }

            a_qconfig, w_qconfig = get_config()
            

            # update scale
            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'llava_vit_qkv', extra_dict)
            hidden_states /= best_scale
            ## linear and ln
            self.self_attn.q_proj.weight.data *= best_scale
            self.self_attn.k_proj.weight.data *= best_scale
            self.self_attn.v_proj.weight.data *= best_scale

            self.layer_norm1.bias.data -= shift
            self.layer_norm1.weight.data /= best_scale
            self.layer_norm1.bias.data /= best_scale
            self.cac_migrate_attn = False

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        if self.cac_migrate_mlp:
            channel_max = hidden_states.max(0)[0].max(0)[0]
            channel_min = hidden_states.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            hidden_states -= shift
            if (self.mlp.fc1.bias is None):
                msmodelslim_logger.warning("mlp.fc1.bias is None")
            self.mlp.fc1.bias.data += shift @ self.mlp.fc1.weight.data.T

            weight_list = torch.cat([self.mlp.fc1.weight])


            extra_dict = {
                'bias': torch.cat([self.mlp.fc1.bias]),
                'act_fn': self.act_fn,
                'observation_mask': None,
            }

            a_qconfig, w_qconfig = get_config()


            # update scale
            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'c_fc', extra_dict)

            hidden_states /= best_scale
            self.mlp.fc1.weight.data *= best_scale

            self.layer_norm2.bias.data -= shift
            self.layer_norm2.weight.data /= best_scale
            self.layer_norm2.bias.data /= best_scale
            
            self.cac_migrate_mlp = False

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
    
        return outputs


class QuantQwen2VLDecoderLayer(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.self_attn = org_layer.self_attn
        self.mlp = org_layer.mlp
        self.input_layernorm = org_layer.input_layernorm
        self.post_attention_layernorm = org_layer.post_attention_layernorm
        self.act_fn = org_layer.mlp.act_fn
        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True
        self.cfg = cfg
        self.layername = layername

    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if not check_migration_import(migration):
            raise ImportError("The current CANN version does not support migration algorithm.")

        hidden_states = args[0]
        attention_mask = kwargs.pop("attention_mask")
        position_ids = kwargs.pop("position_ids")
        past_key_value = kwargs.pop("past_key_value")
        output_attentions = kwargs.pop("output_attentions")
        use_cache = kwargs.pop("use_cache")
        cache_position = kwargs.pop("cache_position")
        position_embeddings = kwargs.pop("position_embeddings")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.cfg.hidden_size == 0:
            raise ValueError("hidden_size can not be zero in model config.")
        if self.cac_migrate_attn:
            msmodelslim_logger.info(f'current block is QuantQwen2VLDecoderLayer, layername:{self.layername}')
            weight_list = torch.cat([self.self_attn.q_proj.weight,
                                     self.self_attn.k_proj.weight,
                                     self.self_attn.v_proj.weight])
            bias_list = None
            if self.cfg.num_attention_heads == 0:
                raise ValueError("num_attention_heads can not be zero in model config.")
            dim = self.cfg.hidden_size // self.cfg.num_attention_heads
            max_position_embeddings = self.cfg.max_position_embeddings
            base = self.cfg.rope_theta
            device = hidden_states.device
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
            t = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_cached = emb.cos().to(torch.get_default_dtype())
            sin_cached = emb.sin().to(torch.get_default_dtype())
            extra_dict = {
                'num_heads': self.self_attn.num_heads,
                'num_key_value_heads': self.self_attn.num_key_value_heads,
                'num_key_value_groups': self.self_attn.num_key_value_groups,
                'cos_cached': cos_cached,
                'sin_cached': sin_cached,
                'head_dim': self.self_attn.head_dim,
                'position_ids': position_ids[0],
                'attention_mask': attention_mask,
                'observation_mask': None
            }
            # update scale
            a_qconfig, w_qconfig = get_config()
            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'llama_qkv', extra_dict, bias=bias_list)
            hidden_states /= best_scale
            self.self_attn.q_proj.weight.data *= best_scale
            self.self_attn.k_proj.weight.data *= best_scale
            self.self_attn.v_proj.weight.data *= best_scale
            self.input_layernorm.weight.data = self.input_layernorm.weight.data.to(hidden_states.device)
            self.input_layernorm.weight.data /= best_scale
            self.cac_migrate_attn = False
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.cac_migrate_mlp:            
            weight_list = torch.cat([self.mlp.gate_proj.weight,
                                     self.mlp.up_proj.weight])
            extra_dict = {
                'observation_mask': None, 
                'act_fn': self.act_fn
            }
            a_qconfig, w_qconfig = get_config()
            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'up_and_gate', extra_dict)
            # update scale
            hidden_states /= best_scale
            self.mlp.gate_proj.weight.data *= best_scale
            self.mlp.up_proj.weight.data *= best_scale
            self.post_attention_layernorm.weight.data /= best_scale
            self.cac_migrate_mlp = False
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class QuantQwen2VLVisionBlock(nn.Module):
    def __init__(self, org_layer, config, layername) -> None:
        super().__init__()
        config = config.vision_config
        self.norm1 = org_layer.norm1
        self.norm2 = org_layer.norm2
        self.attn = org_layer.attn
        self.mlp = org_layer.mlp
        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True
        self.layername = layername
        self.cfg = config

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        if not check_migration_import(migration_vit):
            raise ImportError("The current CANN version does not support migration_vit algorithm.")
        post_ln_1 = self.norm1(hidden_states)
        if self.cac_migrate_attn:
            msmodelslim_logger.info(f'current block is QuantQwen2VLVisionBlock, layername:{self.layername}')
            if self.cfg.num_heads == 0:
                raise ValueError("num_heads can not be zero in model config.")

            channel_max = post_ln_1.max(0)[0]
            channel_min = post_ln_1.min(0)[0]
            shift = (channel_max + channel_min) / 2 
            post_ln_1 -= shift
            if self.attn.qkv.bias is not None:
                self.attn.qkv.bias.data += shift @ self.attn.qkv.weight.data.T
            # calculate scale
            weight_list = torch.cat([self.attn.qkv.weight])
            extra_dict = {
                'num_attention_heads_per_partition': self.attn.num_heads,
                'cu_seqlens': cu_seqlens,
                'rotary_pos_emb': rotary_pos_emb,
                'embed_dim': self.cfg.embed_dim,
                'head_dim': self.cfg.hidden_size // self.cfg.num_heads
            }
            
            a_qconfig, w_qconfig = get_config()
            post_ln_1 = post_ln_1.unsqueeze(1)
            # update scale
            best_scale = \
                migration_vit(post_ln_1, weight_list, a_qconfig, w_qconfig, 'qwen2vl_function', extra_dict)
            post_ln_1 /= best_scale
            ## linear and ln
            self.attn.qkv.weight.data *= best_scale
            self.norm1.bias.data -= shift
            self.norm1.weight.data /= best_scale
            self.norm1.bias.data /= best_scale
            self.cac_migrate_attn = False
        hidden_states = hidden_states + self.attn(
            post_ln_1, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        post_ln_2 = self.norm2(hidden_states)
        if self.cac_migrate_mlp:
            channel_max = post_ln_2.max(0)[0]
            channel_min = post_ln_2.min(0)[0]
            shift = (channel_max + channel_min) / 2
            post_ln_2 -= shift
            if self.mlp.fc1.bias is not None:
                self.mlp.fc1.bias.data += shift @ self.mlp.fc1.weight.data.T  
            # calculate scale
            weight_list = torch.cat([self.mlp.fc1.weight])
            extra_dict = {
                'bias': torch.cat([self.mlp.fc1.bias]),
                'shift': shift,
                'observation_mask': None
                }
            a_qconfig, w_qconfig = get_config()
            # update scale
            best_scale = \
                migration_vit(post_ln_2, weight_list, a_qconfig, w_qconfig, 'c_fc', extra_dict)
            post_ln_2 /= best_scale
            ## linear and ln
            self.mlp.fc1.weight.data *= best_scale
            self.norm2.bias.data -= shift
            self.norm2.weight.data /= best_scale
            self.norm2.bias.data /= best_scale
            self.cac_migrate_mlp = False
        hidden_states = hidden_states + self.mlp(post_ln_2)
        return hidden_states


class QuantQwen25VLVisionBlock(nn.Module):
    """
    适配OS+优化的migrator异常值抑制算法，在原始Qwen2.5-VL视觉部分Vision Block中调用异常值抑制算法接口处理，
    在前向执行过程中直接完成异常值抑制处理。
    """
    def __init__(self, org_layer: nn.Module, config, layername: str) -> None:
        super().__init__()
        config = config.vision_config
        self.norm1 = org_layer.norm1
        self.norm2 = org_layer.norm2
        self.attn = org_layer.attn
        self.mlp = org_layer.mlp
        self.act_fn = org_layer.mlp.act_fn
        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True
        self.layername = layername
        self.cfg = config

    def forward(self, hidden_states, cu_seqlens, **kwargs) -> torch.Tensor:
        if not check_migration_import(migration_vit):
            raise ImportError("The current CANN version does not support migration_vit algorithm.")
        
        rotary_pos_emb = kwargs.pop("rotary_pos_emb", None)
        position_embeddings = kwargs.pop("position_embeddings", None)

        post_ln_1 = self.norm1(hidden_states)
        if self.cac_migrate_attn:
            msmodelslim_logger.info(f'current block is QuantQwen25VLVisionBlock, layername:{self.layername}')
            if self.cfg.num_heads == 0:
                raise ValueError("num_heads can not be zero in model config.")

            # calculate scale
            weight_list = torch.cat([self.attn.qkv.weight])
            extra_dict = {
                'num_attention_heads_per_partition': self.attn.num_heads,
                'cu_seqlens': cu_seqlens,
                'rotary_pos_emb': rotary_pos_emb,
                'position_embeddings': position_embeddings,
                'embed_dim': self.cfg.hidden_size,
                'head_dim': self.cfg.out_hidden_size // self.cfg.num_heads
            }
            
            a_qconfig, w_qconfig = get_config()
            post_ln_1 = post_ln_1.unsqueeze(1)
            # update scale
            best_scale = migration_vit(post_ln_1, weight_list, a_qconfig, w_qconfig, 'qwen2vl_function', extra_dict)
            post_ln_1 /= best_scale
            ## linear and ln
            self.attn.qkv.weight.data *= best_scale
            self.norm1.weight.data /= best_scale
            self.cac_migrate_attn = False
        hidden_states = hidden_states + self.attn(
            post_ln_1, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings
        )
        post_ln_2 = self.norm2(hidden_states)
        if self.cac_migrate_mlp:
            # calculate scale
            weight_list = torch.cat([self.mlp.gate_proj.weight, self.mlp.up_proj.weight])
            bias_list = torch.cat([self.mlp.gate_proj.bias, self.mlp.up_proj.bias])
            extra_dict = {
                'bias': bias_list,
                'observation_mask': None,
                'act_fn': self.act_fn
                }
            a_qconfig, w_qconfig = get_config()
            post_ln_2 = post_ln_2.unsqueeze(0)
            # update scale
            best_scale = \
                migration_vit(post_ln_2, weight_list, a_qconfig, w_qconfig, 'up_and_gate', extra_dict)
            post_ln_2 /= best_scale
            ## linear and ln
            self.mlp.gate_proj.weight.data *= best_scale
            self.mlp.up_proj.weight.data *= best_scale
            self.norm2.weight.data /= best_scale
            self.cac_migrate_mlp = False
        hidden_states = hidden_states + self.mlp(post_ln_2)
        hidden_states = hidden_states.squeeze(0)
        return hidden_states


class QuantInternLM2DecoderLayer(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.cfg = cfg.llm_config
        self.attention = org_layer.attention
        self.act_fn = org_layer.feed_forward.act_fn
        self.feed_forward = org_layer.feed_forward
        self.attention_norm = org_layer.attention_norm
        self.ffn_norm = org_layer.ffn_norm
        self.layername = layername
        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True

    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if not check_migration_import(migration):
            raise ImportError("The current CANN version does not support migration algorithm.")
        hidden_states = args[0]
        attention_mask = kwargs.pop("attention_mask")
        position_ids = kwargs.pop("position_ids")
        past_key_value = kwargs.pop("past_key_value")
        output_attentions = kwargs.pop("output_attentions")
        use_cache = kwargs.pop("use_cache")

        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        if self.cfg.num_attention_heads == 0 or self.cfg.num_key_value_heads == 0:
            raise ValueError("num_attention_heads or num_key_value_heads can not be zero in model config.")
        if self.cac_migrate_attn:
            msmodelslim_logger.info(f'current block is InternLM2DecoderLayer, layername:{self.layername}')
            weight_list = torch.cat([self.attention.wqkv.weight])
            bias_list = None

            hidden_size = self.cfg.hidden_size
            num_heads = self.cfg.num_attention_heads
            head_dim = hidden_size // num_heads
            num_key_value_heads = self.cfg.num_key_value_heads
            num_key_value_groups = num_heads // num_key_value_heads
        

            extra_dict = {
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'past_key_value': past_key_value,
                'output_attentions': output_attentions,
                'use_cache': use_cache,
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'num_key_value_groups': num_key_value_groups,
                'wo': self.attention.wo,
                'wqkv': self.attention.wqkv,
                'rotary_emb': self.attention.rotary_emb
            }
            a_qconfig, w_qconfig = get_config()
            best_scale = migration(hidden_states, weight_list, a_qconfig, w_qconfig, \
                                    'internvl_llm_function', extra_dict, bias=bias_list)
            hidden_states /= best_scale
            self.attention.wqkv.weight.data *= best_scale

            self.attention_norm.weight.data /= best_scale

            self.cac_migrate_attn = False

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states


        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)


        if self.cac_migrate_mlp:
            weight_list = torch.cat([self.feed_forward.w1.weight,
                                     self.feed_forward.w3.weight])
            
            extra_dict = {
                'observation_mask': None, 
                'act_fn': self.act_fn
            }

            a_qconfig, w_qconfig = get_config()

            best_scale = \
                migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'up_and_gate', extra_dict)
            # update scale
            hidden_states /= best_scale
            self.feed_forward.w1.weight.data *= best_scale
            self.feed_forward.w3.weight.data *= best_scale
            self.ffn_norm.weight.data /= best_scale
            self.cac_migrate_mlp = False

        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QuantInternVisionEncoderLayer(nn.Module):
    def __init__(self, org_layer, cfg, layername):
        super().__init__()
        self.all_cfg = cfg
        self.cfg = cfg.vision_config

        self.embed_dim = self.cfg.hidden_size
        self.intermediate_size = self.cfg.intermediate_size
        self.norm_type = self.cfg.norm_type

        self.attn = org_layer.attn
        self.mlp = org_layer.mlp
        self.norm1 = org_layer.norm1
        self.norm2 = org_layer.norm2

        self.ls1 = org_layer.ls1
        self.ls2 = org_layer.ls2
        self.drop_path1 = org_layer.drop_path1
        self.drop_path2 = org_layer.drop_path2

        self.layername = layername
        
        self.cac_migrate_attn = True
        self.cac_migrate_mlp = True

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        if not check_migration_import(migration_vit):
            raise ImportError("The current CANN version does not support migration_vit algorithm.")
        qk_normalization = self.cfg.qk_normalization
        post_ln_1 = self.norm1(hidden_states).to(hidden_states.dtype)
        if self.cac_migrate_attn:
            msmodelslim_logger.info(f'current block is InternVisionEncoderLayer,layername:{self.layername}')
            channel_max = post_ln_1.max(0)[0].max(0)[0]
            channel_min = post_ln_1.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            post_ln_1 -= shift

            if self.attn.qkv.bias is not None:
                self.attn.qkv.bias.data += shift @ self.attn.qkv.weight.data.T
            if self.cfg.num_attention_heads == 0:
                raise ValueError("num_attention_heads can not be zero in model config.")
            head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads
            # calculate scale
            weight_list = torch.cat([self.attn.qkv.weight])
            extra_dict = {  
                'num_heads': self.cfg.num_attention_heads,
                'scale': head_dim ** -0.5,
                'attn_drop': self.attn.attn_drop,
                'proj': self.attn.proj,
                'proj_drop': self.attn.proj_drop,
                'qk_normalization': qk_normalization,
            }
            if qk_normalization:
                extra_dict['q_norm'] = self.attn.q_norm
                extra_dict['k_norm'] = self.attn.k_norm
            a_qconfig, w_qconfig = get_config()
            # update scale
            best_scale = \
                migration_vit(post_ln_1, weight_list, a_qconfig, w_qconfig, 'internvl_vit_function', extra_dict)
            post_ln_1 /= best_scale
            ## linear and ln
            self.attn.qkv.weight.data *= best_scale
            self.norm1.weight.data /= best_scale
            if not qk_normalization:
                self.norm1.bias.data -= shift
                self.norm1.bias.data /= best_scale
            self.cac_migrate_attn = False
            
        hidden_states = hidden_states + self.drop_path1(self.attn(post_ln_1) * self.ls1)
        post_ln_2 = self.norm2(hidden_states).to(hidden_states.dtype)
        if self.cac_migrate_mlp:
            channel_max = post_ln_2.max(0)[0].max(0)[0]
            channel_min = post_ln_2.min(0)[0].min(0)[0]
            shift = (channel_max + channel_min) / 2
            post_ln_2 -= shift
            if check_internvl2_8b_model(self.all_cfg):
                if self.mlp.fc1.bias is not None:
                    self.mlp.fc1.bias.data += shift @ self.mlp.fc1.weight.data.T

            # calculate scale
            weight_list = torch.cat([self.mlp.fc1.weight])
            extra_dict = {
                'bias': torch.cat([self.mlp.fc1.bias]),
                'act_fn': self.mlp.act,
                'observation_mask': None
                }

            a_qconfig, w_qconfig = get_config()

            # update scale
            best_scale = \
                migration_vit(post_ln_2, weight_list, a_qconfig, w_qconfig, 'c_fc', extra_dict)
            post_ln_2 /= best_scale
            ## linear and ln
            self.mlp.fc1.weight.data *= best_scale
            self.norm2.weight.data /= best_scale
            if not qk_normalization:
                self.norm2.bias.data -= shift
                self.norm2.bias.data /= best_scale
            self.cac_migrate_mlp = False

        hidden_states = hidden_states + self.drop_path2(self.mlp(post_ln_2) * self.ls2)

        return hidden_states
