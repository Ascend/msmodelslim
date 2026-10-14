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
# pylint: disable=consider-using-from-import

import pytest
import torch
import torch.nn as nn

from msmodelslim.processor.anti_outlier.common import VirtualVModuleFromQKVFused


class TestVirtualVModule:
    """测试VirtualVModuleFromQKVFused的各种功能"""

    @pytest.fixture
    def qkv_module_with_bias(self):
        """创建带有偏置的QKV模块"""
        # MHA: 8个注意力头，每个头64维，总共512维
        # QKV融合: [Q(8 * 64), K(8 * 64), V(8 * 64)] = [512, 512, 512] -> 输出1536维
        qkv_module = nn.Linear(512, 1536, bias=True)
        return qkv_module

    @pytest.fixture
    def qkv_module_without_bias(self):
        """创建不带偏置的QKV模块"""
        qkv_module = nn.Linear(512, 1536, bias=False)
        return qkv_module

    def test_virtual_v_module_init_normal(self, qkv_module_with_bias):
        """测试初始化是否正确"""
        # 测试MHA情况
        virtual_v = VirtualVModuleFromQKVFused(
            qkv_module=qkv_module_with_bias, num_attention_heads=8, num_key_value_heads=8
        )

        # 验证初始化成功
        assert virtual_v.attention_type == "MHA"
        assert virtual_v.num_attention_heads == 8
        assert virtual_v.num_key_value_heads == 8
        assert virtual_v.qkv_module is qkv_module_with_bias

        # 验证V部分权重和偏置被正确提取
        assert virtual_v.weight is not None
        assert virtual_v.bias is not None
        assert virtual_v.weight.shape == (512, 512)  # V部分: 8 * 64 = 512维
        assert virtual_v.bias.shape == (512,)

    def test_virtual_v_module_update_weights(self, qkv_module_with_bias):
        """测试权重更新是否正确"""
        virtual_v = VirtualVModuleFromQKVFused(
            qkv_module=qkv_module_with_bias, num_attention_heads=8, num_key_value_heads=8
        )

        # 保存原始权重
        original_qkv_weight = virtual_v.qkv_module.weight.clone()
        original_qkv_bias = virtual_v.qkv_module.bias.clone()

        # 修改V部分的权重和偏置
        new_v_weight = torch.randn_like(virtual_v.weight)
        new_v_bias = torch.randn_like(virtual_v.bias)
        virtual_v.weight.data = new_v_weight
        virtual_v.bias.data = new_v_bias

        # 更新权重
        virtual_v.update_weights()

        # 验证QKV模块的V部分权重和偏置被正确更新
        head_dim = 64
        v_start = 8 * head_dim + 8 * head_dim  # Q + K
        v_end = v_start + 8 * head_dim  # + V

        assert torch.allclose(virtual_v.qkv_module.weight[v_start:v_end], new_v_weight)
        assert torch.allclose(virtual_v.qkv_module.bias[v_start:v_end], new_v_bias)

        # 验证Q和K部分没有被修改
        assert torch.allclose(virtual_v.qkv_module.weight[:v_start], original_qkv_weight[:v_start])
        assert torch.allclose(virtual_v.qkv_module.bias[:v_start], original_qkv_bias[:v_start])

    def test_virtual_v_module_determine_attention_type_mha(self):
        """测试注意力类型识别为MHA"""
        qkv_module = nn.Linear(512, 1536)
        virtual_v = VirtualVModuleFromQKVFused(qkv_module=qkv_module, num_attention_heads=8, num_key_value_heads=8)

        assert virtual_v.attention_type == "MHA"

    def test_virtual_v_module_determine_attention_type_mqa(self):
        """测试注意力类型识别为MQA"""
        qkv_module = nn.Linear(512, 1024)  # MQA: 8 * 64 + 1 * 64 + 1 * 64 = 640
        virtual_v = VirtualVModuleFromQKVFused(qkv_module=qkv_module, num_attention_heads=8, num_key_value_heads=1)

        assert virtual_v.attention_type == "MQA"

    def test_virtual_v_module_determine_attention_type_gqa(self):
        """测试注意力类型识别为GQA"""
        qkv_module = nn.Linear(512, 1280)  # GQA: 8 * 64 + 4 * 64 + 4 * 64 = 1024
        virtual_v = VirtualVModuleFromQKVFused(qkv_module=qkv_module, num_attention_heads=8, num_key_value_heads=4)

        assert virtual_v.attention_type == "GQA"

    def test_virtual_v_module_get_v_indices_mha(self):
        """测试V部分索引计算（MHA）"""
        qkv_module = nn.Linear(512, 1536)
        virtual_v = VirtualVModuleFromQKVFused(qkv_module=qkv_module, num_attention_heads=8, num_key_value_heads=8)

        head_dim = 64
        v_start, v_end = virtual_v._get_v_indices(head_dim)

        expected_v_start = 8 * head_dim + 8 * head_dim  # 1024
        expected_v_end = expected_v_start + 8 * head_dim  # 1536

        assert v_start == expected_v_start
        assert v_end == expected_v_end

    def test_virtual_v_module_get_v_indices_mqa(self):
        """测试V部分索引计算（MQA）"""
        qkv_module = nn.Linear(512, 1024)
        virtual_v = VirtualVModuleFromQKVFused(qkv_module=qkv_module, num_attention_heads=8, num_key_value_heads=1)

        head_dim = 64
        v_start, v_end = virtual_v._get_v_indices(head_dim)

        expected_v_start = 8 * head_dim + 1 * head_dim  # 576
        expected_v_end = expected_v_start + 1 * head_dim  # 640

        assert v_start == expected_v_start
        assert v_end == expected_v_end

    def test_virtual_v_module_get_v_indices_gqa(self):
        """测试V部分索引计算（GQA）"""
        qkv_module = nn.Linear(512, 1280)
        virtual_v = VirtualVModuleFromQKVFused(qkv_module=qkv_module, num_attention_heads=8, num_key_value_heads=4)

        head_dim = 64
        v_start, v_end = virtual_v._get_v_indices(head_dim)
        expected_v_start = 8 * head_dim + 4 * head_dim  # 768
        expected_v_end = expected_v_start + 4 * head_dim  # 1024

        assert v_start == expected_v_start
        assert v_end == expected_v_end

    def test_virtual_v_module_extract_v_weights_with_bias(self, qkv_module_with_bias):
        """测试提取V部分权重和偏置（有偏置）"""
        virtual_v = VirtualVModuleFromQKVFused(
            qkv_module=qkv_module_with_bias, num_attention_heads=8, num_key_value_heads=8
        )

        # 验证V部分权重和偏置被正确提取
        assert virtual_v.weight is not None
        assert virtual_v.bias is not None

        # 验证权重形状
        assert virtual_v.weight.shape == (512, 512)  # V部分: 8 * 64 = 512维
        assert virtual_v.bias.shape == (512,)

        # 验证提取的权重和偏置与原始QKV模块的V部分一致
        head_dim = 64
        v_start = 8 * head_dim + 8 * head_dim  # Q + K
        v_end = v_start + 8 * head_dim  # + V

        assert torch.allclose(virtual_v.weight, qkv_module_with_bias.weight[v_start:v_end])
        assert torch.allclose(virtual_v.bias, qkv_module_with_bias.bias[v_start:v_end])

    def test_virtual_v_module_extract_v_weights_without_bias(self, qkv_module_without_bias):
        """测试提取V部分权重和偏置（无偏置）"""
        virtual_v = VirtualVModuleFromQKVFused(
            qkv_module=qkv_module_without_bias, num_attention_heads=8, num_key_value_heads=8
        )

        # 验证V部分权重被正确提取，偏置为None
        assert virtual_v.weight is not None
        assert virtual_v.bias is None

        # 验证权重形状
        assert virtual_v.weight.shape == (512, 512)  # V部分: 8 * 64 = 512维

        # 验证提取的权重与原始QKV模块的V部分一致
        head_dim = 64
        v_start = 8 * head_dim + 8 * head_dim  # Q + K
        v_end = v_start + 8 * head_dim  # + V

        assert torch.allclose(virtual_v.weight, qkv_module_without_bias.weight[v_start:v_end])

    def test_virtual_v_module_update_weights_without_bias(self, qkv_module_without_bias):
        """测试更新权重（无原始偏置）"""
        virtual_v = VirtualVModuleFromQKVFused(
            qkv_module=qkv_module_without_bias, num_attention_heads=8, num_key_value_heads=8
        )

        # 创建新的偏置
        new_v_bias = torch.randn(512)
        virtual_v.bias = nn.Parameter(new_v_bias)

        # 修改V部分的权重
        new_v_weight = torch.randn_like(virtual_v.weight)
        virtual_v.weight.data = new_v_weight

        # 更新权重
        virtual_v.update_weights()

        # 验证QKV模块现在有了偏置
        assert virtual_v.qkv_module.bias is not None

        # 验证V部分的权重和偏置被正确更新
        head_dim = 64
        v_start = 8 * head_dim + 8 * head_dim  # Q + K
        v_end = v_start + 8 * head_dim  # + V

        assert torch.allclose(virtual_v.qkv_module.weight[v_start:v_end], new_v_weight)
        assert torch.allclose(virtual_v.qkv_module.bias[v_start:v_end], new_v_bias)

        # 验证Q和K部分的偏置为0
        assert torch.allclose(
            virtual_v.qkv_module.bias[:v_start], torch.zeros_like(virtual_v.qkv_module.bias[:v_start])
        )

    def test_virtual_v_module_update_weights_with_bias(self, qkv_module_with_bias):
        """测试更新权重（有原始偏置）"""
        virtual_v = VirtualVModuleFromQKVFused(
            qkv_module=qkv_module_with_bias, num_attention_heads=8, num_key_value_heads=8
        )

        # 保存原始偏置
        original_bias = virtual_v.qkv_module.bias.clone()

        # 修改V部分的权重和偏置
        new_v_weight = torch.randn_like(virtual_v.weight)
        new_v_bias = torch.randn_like(virtual_v.bias)
        virtual_v.weight.data = new_v_weight
        virtual_v.bias.data = new_v_bias

        # 更新权重
        virtual_v.update_weights()

        # 验证V部分的权重和偏置被正确更新
        head_dim = 64
        v_start = 8 * head_dim + 8 * head_dim  # Q + K
        v_end = v_start + 8 * head_dim  # + V

        assert torch.allclose(virtual_v.qkv_module.weight[v_start:v_end], new_v_weight)
        assert torch.allclose(virtual_v.qkv_module.bias[v_start:v_end], new_v_bias)

        # 验证Q和K部分的偏置没有被修改
        assert torch.allclose(virtual_v.qkv_module.bias[:v_start], original_bias[:v_start])


if __name__ == "__main__":
    pytest.main([__file__])
