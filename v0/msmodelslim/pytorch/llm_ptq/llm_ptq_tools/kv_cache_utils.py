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


import torch
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    fake_quantize, linear_quantization_params
)


def set_kvcache_vari_func(mod, cache_sub_dict, cfg, num_layers=None):
    """
    Sets various attributes related to key-value cache calibration to a module.
    """
    kv_sym = True if not hasattr(cfg, 'kv_sym') else cfg.kv_sym
    attributes = {
        'is_calib': False,
        'k_min': None, 'k_max': None,
        'v_min': None, 'v_max': None,
        'k_scale': None, 'k_offset': None,
        'v_scale': None, 'v_offset': None,
        'cache_seq_index': None, 'cache_bz_index': None,
        'layer_idx': 0, 'kv_sym': kv_sym,
    }
    for attr, value in attributes.items():
        if not hasattr(mod, attr):
            setattr(mod, attr, value)

    attr_map = {
            'k_scale': ['k_proj', 'scale'], 
            'k_offset': ['k_proj', 'offset'],  
            'v_scale': ['v_proj', 'scale'],
            'v_offset': ['v_proj', 'offset']  
    }
    for new_att, att_keys in attr_map.items():
        position_str, att_str = att_keys
        for key in cache_sub_dict.keys():
            if position_str in key and att_str in key:
                value_alt = cache_sub_dict[key] 
                setattr(mod, new_att, value_alt)
    # set layer number
    num_hidden_layers = 'num_hidden_layers'
    layer_names = ['num_layers', 'n_layer', num_hidden_layers]
    for name in layer_names:
        if hasattr(mod, 'config') and hasattr(mod.config, name) and not hasattr(mod, num_hidden_layers):
            setattr(mod, num_hidden_layers, getattr(mod.config, name))
    if num_layers is not None and not hasattr(mod, num_hidden_layers):
        setattr(mod, num_hidden_layers, num_layers)
    # set kvcache functions
    functions = {
                "disable_calib": disable_calib,
                "enable_calib": enable_calib,
                "any_none": any_none,
                "get_kvcache_scale_offset": get_kvcache_scale_offset,
    }
    for attr, value in functions.items():
        if not hasattr(mod, attr):
            setattr(mod, attr, value.__get__(mod, mod.__class__))


def update_extremum(extremum, torch_function, value):
    """
    Update the extremum value with a new value using the specified torch function.
    If current_extremum is None, initialize it with new_value.
    """
    if extremum is not None:
        extremum = torch_function(extremum, value)
    else:
        extremum = value
    return extremum


def process_tensor(batch_size, seq_len, channel_max, channel_min, tensor):
    """
    Process a tensor to update channel-wise max and min values.
    """
    tensor = tensor.reshape(batch_size, seq_len, -1)
    tensor = tensor.view(tensor.shape[0] * tensor.shape[1], -1)
    # calculate current min_max
    coming_max = torch.max(tensor, dim=0)[0]
    coming_min = torch.min(tensor, dim=0)[0]
    # update min_max
    channel_max = update_extremum(channel_max, torch.max, coming_max)
    channel_min = update_extremum(channel_min, torch.min, coming_min)

    return channel_max, channel_min


def split_kvcache(kvcache):
    '''
    Splits the kv-cache into separate key and value caches.
    '''
    if isinstance(kvcache, tuple):
        if len(kvcache) == 1:
            kvcache = torch.chunk(kvcache[0], 2, dim=2).detach()
        else:
            cache_k = [kvcache[0]]
            cache_v = [kvcache[1]]
    # kv cache适配glm模型，解析tensor类型的kvcache
    elif isinstance(kvcache, torch.Tensor):
        cache_k = [kvcache[0][0]]
        cache_v = [kvcache[0][1]]
    else:
        cache_k = kvcache.key_cache
        cache_v = kvcache.value_cache
    return cache_k, cache_v


def get_var_index(x, cache_k):
    batch_size = x[0].shape[0]
    seq_len = x[0].shape[1]
    cache_seq_index = torch.where(torch.tensor(cache_k[0].shape) == seq_len)[0]
    cache_bz_index = torch.where(torch.tensor(cache_k[0].shape) == batch_size)[0]
    return cache_seq_index, cache_bz_index


def update_bz_seq(tensor, bz_index, seq_index):
    batch_size = tensor.shape[bz_index]
    seq_len = tensor.shape[seq_index]
    return batch_size, seq_len


def fake_quantize_process(batch_size, seq_len,
                          k_scale, k_offset, tensor):
    """
    Fake quantizes a tensor with the given scale and offset.
    """
    ori_shape = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len, -1)
    _, tensor_dequant = fake_quantize(
        tensor, k_scale, k_offset, 8, is_signed=True, dequant=True
    )
    tensor_dequant = tensor_dequant.reshape(ori_shape)
    return tensor_dequant


def fake_quantize_cache(batch_size, seq_len, kv_scale_offset,
                        cache_k, cache_v):
    k_scale, k_offset = kv_scale_offset[0][0], kv_scale_offset[0][1]
    v_scale, v_offset = kv_scale_offset[1][0], kv_scale_offset[1][1]
    fake_cache_k = fake_quantize_process(batch_size, seq_len, k_scale, k_offset, cache_k)
    fake_cache_v = fake_quantize_process(batch_size, seq_len, v_scale, v_offset, cache_v)
    return fake_cache_k, fake_cache_v


def disable_calib(self,):
    self.is_calib = False


def enable_calib(self,):
    self.is_calib = True


def get_kvcache_scale_offset(self):
    # update the scale and offset of kvcache 
    self.k_scale, self.k_offset = linear_quantization_params(8, self.k_min, self.k_max, 
                                    integral_zero_point=True, q_signed=True, sym=self.kv_sym)
    self.v_scale, self.v_offset = linear_quantization_params(8, self.v_min, self.v_max, 
                                    integral_zero_point=True, q_signed=True, sym=self.kv_sym)


def any_none(self):
    return any(val is None for val in [self.k_scale, self.k_offset, self.v_scale, self.v_offset])


# 定义新的forward函数
def new_forward(
    self,
    *args,
    **kwargs,
):
    """
    A modified forward function that wraps the original forward function of a module.
    It processes the kv-cache for self-attention mechanisms and applies fake quantization when the model is inferencing
    """
    use_cache_str = 'use_cache'
    try:
        use_cache = kwargs['use_cache']
    except KeyError as e:
        raise Exception("The input variable of attention don't have use_cache {e}") from e
    if use_cache_str in kwargs.keys() and not use_cache:
        kwargs[use_cache_str] = True
    x = self.original_forward(
        *args,
        **kwargs,
    )
    kvcache = x[-1]
    if kvcache is not None:
        cache_k, cache_v = split_kvcache(kvcache)
        # get the index of sequence length and batch size
        if self.cache_seq_index is None or self.cache_bz_index is None:
            self.cache_seq_index, self.cache_bz_index = get_var_index(x, cache_k)

        # update the min max of kvcache and fake quantize 
        if self.is_calib:
            for tensor_k in cache_k:
                batch_size, seq_len = update_bz_seq(tensor_k, self.cache_bz_index, self.cache_seq_index)
                self.k_max, self.k_min = process_tensor(batch_size, seq_len,
                                                        self.k_max, self.k_min, tensor_k)
            for tensor_v in cache_v:
                batch_size, seq_len = update_bz_seq(tensor_v, self.cache_bz_index, self.cache_seq_index)
                self.v_max, self.v_min = process_tensor(batch_size, seq_len,
                                                        self.v_max, self.v_min, tensor_v)
        else:
            if self.any_none():
                self.get_kvcache_scale_offset()
            kv_scale_offset = [
                [self.k_scale, self.k_offset],
                [self.v_scale, self.v_offset]
            ]
            if len(cache_k) != self.num_hidden_layers:
                batch_size, seq_len = update_bz_seq(cache_k[-1], self.cache_bz_index, self.cache_seq_index)
                fake_cache_k, fake_cache_v = fake_quantize_cache(batch_size, seq_len, kv_scale_offset,
                                                                cache_k[-1], cache_v[-1])
                cache_k[-1], cache_v[-1] = fake_cache_k, fake_cache_v
                self.layer_idx = len(cache_k) - 1
            else:
                batch_size, seq_len = update_bz_seq(cache_k[self.layer_idx], self.cache_bz_index, self.cache_seq_index)
                pre_k = cache_k[self.layer_idx].narrow(self.cache_seq_index.item(), 0, seq_len-1)
                incre_k = cache_k[self.layer_idx].narrow(self.cache_seq_index.item(), -1, 1)
                pre_v = cache_v[self.layer_idx].narrow(self.cache_seq_index.item(), 0, seq_len-1)
                incre_v = cache_v[self.layer_idx].narrow(self.cache_seq_index.item(), -1, 1)
                
                incre_k, incre_v = fake_quantize_cache(batch_size, 1, kv_scale_offset,
                                                       incre_k, incre_v)
                
                cache_k[self.layer_idx] = torch.cat((pre_k, incre_k), dim=self.cache_seq_index.item())
                cache_v[self.layer_idx] = torch.cat((pre_v, incre_v), dim=self.cache_seq_index.item())                  
    return x
