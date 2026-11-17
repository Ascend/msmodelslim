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

from abc import abstractmethod

import torch
from torch import nn

from msmodelslim.utils.exception import SpecError, ToDoError
from msmodelslim.utils.logging import get_logger


class VirtualVModuleBase(nn.Module):
    @abstractmethod
    def update_weights(self):
        raise ToDoError("update_weights is not implemented.", action="Please implement this method in the subclass.")

    def forward(self, x):
        raise SpecError(
            "VirtualVModuleBase is not supported for forward.", action="Please do Not use this module for forward."
        )


class VirtualVModuleFromQKVFused(VirtualVModuleBase):
    """Virtual V module for handling QKV fusion, supports MHA, MQA and GQA"""

    def __init__(self, qkv_module: nn.Linear, num_attention_heads: int, num_key_value_heads: int):
        super().__init__()
        self.qkv_module = qkv_module
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # Determine attention mechanism type
        self.attention_type = self._determine_attention_type()

        # Extract V part weights and bias
        self._extract_v_weights()

    def update_weights(self):
        """Update V weights back to the original QKV module"""
        qkv_weight = self.qkv_module.weight
        qkv_bias = getattr(self.qkv_module, 'bias', None)

        # Calculate head dimension
        head_dim = qkv_weight.shape[1] // self.num_attention_heads

        # Get V part indices based on attention type
        v_start, v_end = self._get_v_indices(head_dim)

        # Update V part weights
        with torch.no_grad():
            qkv_weight[v_start:v_end] = self.weight.data

            # Handle bias update: if bias is produced after smoothing, update to qkv_module
            if hasattr(self, 'bias') and self.bias is not None:
                if qkv_bias is not None:
                    # Original qkv_module has bias, directly update V part
                    qkv_bias[v_start:v_end] = self.bias.data
                else:
                    # Original qkv_module has no bias, but bias is produced after smoothing
                    # Create zero bias, then update V part
                    device = qkv_weight.device
                    dtype = qkv_weight.dtype
                    new_bias = torch.zeros(qkv_weight.shape[0], device=device, dtype=dtype)
                    new_bias[v_start:v_end] = self.bias.data
                    self.qkv_module.bias = nn.Parameter(new_bias)

    def _determine_attention_type(self) -> str:
        """Determine attention mechanism type"""
        if self.num_key_value_heads == 1:
            return "MQA"  # Multi-Query Attention
        elif self.num_key_value_heads == self.num_attention_heads:
            return "MHA"  # Multi-Head Attention
        elif (
            self.num_key_value_heads < self.num_attention_heads
            and self.num_attention_heads % self.num_key_value_heads == 0
        ):
            return "GQA"  # Grouped-Query Attention
        else:
            get_logger().warning("Invalid attention type, please check.")
            return "UNKNOWN"

    def _get_v_indices(self, head_dim: int) -> tuple:
        """Get V part index range based on attention type"""
        if self.attention_type == "MHA":
            # MHA: QKV order is [Q, K, V], each has num_attention_heads heads
            q_size = self.num_attention_heads * head_dim
            k_size = self.num_attention_heads * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + self.num_attention_heads * head_dim

        elif self.attention_type == "MQA":
            # MQA: QKV order is [Q, K, V], Q has num_attention_heads heads, K/V only has 1 head
            q_size = self.num_attention_heads * head_dim
            k_size = 1 * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + 1 * head_dim

        elif self.attention_type == "GQA":
            # GQA: QKV order is [Q, K, V], Q has num_attention_heads heads, K/V has num_key_value_heads heads
            q_size = self.num_attention_heads * head_dim
            k_size = self.num_key_value_heads * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + self.num_key_value_heads * head_dim
        else:
            raise ValueError(f"Invalid attention type: {self.attention_type}")
        return v_start, v_end

    def _extract_v_weights(self):
        """Extract V part weights and bias from QKV module"""
        qkv_weight = self.qkv_module.weight
        qkv_bias = getattr(self.qkv_module, 'bias', None)

        # Calculate head dimension
        head_dim = qkv_weight.shape[1] // self.num_attention_heads

        # Get V part index range based on attention type
        v_start, v_end = self._get_v_indices(head_dim)

        # Extract V part weights
        self.weight = nn.Parameter(qkv_weight[v_start:v_end].clone())

        # Extract V part bias
        if qkv_bias is not None:
            self.bias = nn.Parameter(qkv_bias[v_start:v_end].clone())
        else:
            self.bias = None


class VirtualVModuleFromKVFused(VirtualVModuleBase):
    """
    Virtual V module for handling DeepSeek V3 KV fusion (expanded by head, row dimension concatenated as [K_nope, V])

    Assumes qk_nope_head_dim equals v_head_dim, and automatically derives each head size from weight out_features.
    """

    def __init__(self, kv_module: nn.Linear, num_attention_heads: int, qk_nope_head_dim: int, v_head_dim: int):
        super().__init__()
        self.kv_module = kv_module
        self.num_attention_heads = num_attention_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        # Derive number of rows per head from weight dimension, enforce K_nope and V equal length
        out_features = self.kv_module.weight.data.size(0)
        per_head_out = out_features // self.num_attention_heads
        if per_head_out != self.qk_nope_head_dim + self.v_head_dim:
            raise SpecError(
                'KV-fused module weight dimension mismatch, please check.',
                action='Please ensure the weight out_features dimension '
                'is (qk_nope_head_dim + v_head_dim) * num_attention_heads.',
            )

        # Calculate all V row indices: for each head, select the latter v_head_dim rows
        kv_w = self.kv_module.weight.data.view(self.num_attention_heads, per_head_out, -1)
        v_w = kv_w[:, self.qk_nope_head_dim :, :].contiguous().view(-1, kv_w.size(-1))
        self.weight = nn.Parameter(v_w.clone())
        if self.kv_module.bias is not None:
            kv_b = self.kv_module.bias.data.view(self.num_attention_heads, per_head_out)
            v_b = kv_b[:, self.qk_nope_head_dim :].contiguous().view(-1)
            self.bias = nn.Parameter(v_b.clone())

    def update_weights(self):
        """Update V weights back to the original KV module"""
        with torch.no_grad():
            out_features = self.kv_module.weight.data.size(0)
            per_head_out = out_features // self.num_attention_heads
            kv_w = self.kv_module.weight.data.view(self.num_attention_heads, per_head_out, -1)
            kv_w[:, self.qk_nope_head_dim :, :] = self.weight.data.view(self.num_attention_heads, self.v_head_dim, -1)

            # Handle bias update: if bias is produced after smoothing, update to kv_module
            if hasattr(self, 'bias') and self.bias is not None:
                if self.kv_module.bias is not None:
                    # Original kv_module has bias, directly update V part
                    kv_b = self.kv_module.bias.data.view(self.num_attention_heads, per_head_out)
                    # self.bias is 1D, needs reshape to [num_attention_heads, v_head_dim]
                    v_bias_reshaped = self.bias.data.view(self.num_attention_heads, self.v_head_dim)
                    kv_b[:, self.qk_nope_head_dim :] = v_bias_reshaped
                else:
                    # Original kv_module has no bias, but bias is produced after smoothing
                    # Create zero bias, then update V part
                    new_bias = torch.zeros(
                        out_features, device=self.kv_module.weight.device, dtype=self.kv_module.weight.dtype
                    )
                    kv_b = new_bias.view(self.num_attention_heads, per_head_out)
                    # self.bias is 1D, needs reshape to [num_attention_heads, v_head_dim]
                    v_bias_reshaped = self.bias.data.view(self.num_attention_heads, self.v_head_dim)
                    kv_b[:, self.qk_nope_head_dim :] = v_bias_reshaped
                    self.kv_module.bias = nn.Parameter(new_bias)
