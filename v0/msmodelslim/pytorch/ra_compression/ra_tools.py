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

import torch
import torch.nn as nn

from ascend_utils.common.security import check_type, check_int, SafeWriteUmask, get_valid_write_path
from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook
from msmodelslim.pytorch.ra_compression.ra_config import RACompressConfig
from msmodelslim.pytorch.ra_compression.ra_compression_kia import get_wins
from msmodelslim import logger


class RACompressor(object):

    def __init__(self, model, cfg: RACompressConfig):
        check_type(model, nn.Module)
        check_type(cfg, RACompressConfig, param_name="cfg")
        if not hasattr(model, "config"):
            raise ValueError("Model does not have attribute `config`. \
                              Model must be a huggingface model.")
        if not hasattr(model.config, 'hidden_size'):
            raise ValueError("Model must have a `config` attribute with a `hidden_size` property. \
                              Model must be a huggingface model.")
        if not hasattr(model.config, 'num_attention_heads') or \
            model.config.num_attention_heads == 0:
            raise ValueError("Model must have a `config` attribute with a `num_attention_heads` property. \
                              Model must be a huggingface model.")
        self.model = model
        self.cfg = cfg
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.hidden_size // self.model.config.num_attention_heads
        check_int(self.head_dim, min_value=1, max_value=1000, param_name="head_dim")
    
    def get_alibi_windows(self, save_path):
        check_type(save_path, str, param_name="save_path")
        wins = []
        qk_list = self._get_qk_weight_and_reshape_by_num_heads()
        slopes = torch.Tensor(self._get_interleave(self.model.config.num_attention_heads))
        wins = get_wins(qk_list, self.cfg.theta, self.cfg.alpha, slopes)
        wins = torch.tensor(wins)
        with SafeWriteUmask(umask=0o377):
            output_model_path = get_valid_write_path(save_path, extensions=".pt")
            torch.save(wins, output_model_path)
        logger.info("windows is stored in %r ", output_model_path)
    
    def _get_attention_mlp_blocks(self, model):
        dfs = [model]
        while len(dfs) > 0:
            cur = dfs.pop(0)
            if isinstance(cur, (torch.nn.ModuleList, torch.nn.Sequential)):
                return cur
            dfs.extend(cur.modules())
        return None
    
    def _get_qkv_name(self, attention_mlp_block, hidden_size):
        dag = DagTorchHook(attention_mlp_block, torch.ones([1, 1, hidden_size]).float())

        norm_node_met, attn_linears = 0, []
        for node in dag.dag_node_list:
            if "norm" in node.op_type.lower():
                norm_node_met += 1
                if norm_node_met > 1:
                    break
            if node.op_type == "Linear":
                attn_linears.append(node.name)
        return attn_linears if len(attn_linears) == 0 else attn_linears[:-1]  # Exclude output linear
    
    def _split_qkv_weight_to_query_key(self, weight_list, num_attention_heads):
        if len(weight_list) == 1:  # qkv
            qkv_weight = weight_list[0]
            # 添加形状验证：确保第一维可被3整除
            if qkv_weight.shape[0] % 3 != 0:
                raise ValueError(
                    f"QKV fused weight first dimension must be divisible by 3, "
                    f"but got shape {qkv_weight.shape}. "
                    f"Please check model configuration or weight structure."
                )
            qkv_weight = qkv_weight.reshape(3, qkv_weight.shape[0] // 3, num_attention_heads, -1)
            return (qkv_weight[0], qkv_weight[1])
        elif len(weight_list) == 2:  # q, kv
            q_weight, kv_weight = weight_list
             # 添加形状验证：确保KV权重第一维可被2整除
            if kv_weight.shape[0] % 2 != 0:
                raise ValueError(
                    f"KV fused weight first dimension must be divisible by 2, "
                    f"but got shape {kv_weight.shape}. "
                    f"Please check model configuration or weight structure."
                )
            kv_weight = kv_weight.reshape(2, kv_weight.shape[0] // 2, num_attention_heads, -1)
            return (q_weight.reshape(q_weight.shape[0], num_attention_heads, -1), kv_weight[0])
        else:  # q, k, v
            q_weight = weight_list[0].reshape(weight_list[0].shape[0], num_attention_heads, -1)
            k_weight = weight_list[1].reshape(weight_list[1].shape[0], num_attention_heads, -1)
            return (q_weight, k_weight)

    def _get_qk_weight_and_reshape_by_num_heads(self):
        attention_mlp_blocks = self._get_attention_mlp_blocks(self.model)
        if not attention_mlp_blocks:
            raise ValueError("Found no ModuleList in model")
        qkv_names = self._get_qkv_name(attention_mlp_blocks[0], self.hidden_size)
        if not qkv_names:
            raise ValueError("Found no Linear node in model attention block")
        if len(qkv_names) > 3:
            raise ValueError(f"Found {len(qkv_names)} Linears node in model attention block, should be <= 3")

        qk_list, cur_list = [], []
        qkv_names_suffix = [ii + ".weight" for ii in qkv_names]
        for name, weight in attention_mlp_blocks.state_dict().items():
            if not any([name.endswith(ii) for ii in qkv_names_suffix]):
                continue
            cur_list.append(weight)
            if len(cur_list) == len(qkv_names_suffix):
                qk_list.append(self._split_qkv_weight_to_query_key(cur_list, self.model.config.num_attention_heads))
                cur_list = []
        return qk_list

    def _get_interleave(self, n):
        """
        用于计算alibi编码的斜率slopes
        """
        def _get_interleave_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * start ** i for i in range(n)]

        if math.log2(n).is_integer():
            # 如果n是2的幂
            return _get_interleave_power_of_2(n)
        else:
            """
            找到小于或等于n的最大的2的幂
            生成这个2的幂对应的数列
            调用get_interleave来生成2*closest_power_of_2的数列
            从两倍数列中每隔一个元素取一个，直到取得n - closest_power_of_2个元素为止
            最后将这两个数列合并
            """
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            alibi_slopes = _get_interleave_power_of_2(closest_power_of_2) + \
                self._get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            return alibi_slopes
  
