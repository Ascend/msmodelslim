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
# pylint: disable=attribute-defined-outside-init,consider-using-from-import

from unittest.mock import patch

from unittest.mock import Mock  # noqa: F401
import pytest
import torch
import torch.nn as nn

from msmodelslim.processor.anti_outlier.common import VirtualVModuleFromKVFused
from msmodelslim.utils.exception import SpecError


class TestVirtualVModuleFromKVFused:
    """VirtualVModuleFromKVFused 类的单元测试"""

    @staticmethod
    def test_init_error_recovery():
        """测试初始化错误恢复"""
        # 测试各种无效参数
        invalid_configs = [  # noqa: F841
            {'num_attention_heads': 0},
            {'num_attention_heads': -1},
            {'qk_nope_head_dim': 0},
            {'qk_nope_head_dim': -1},
            {'v_head_dim': 0},
            {'v_head_dim': -1},
        ]

    def setup_method(self):
        """测试前置设置"""
        # 测试参数设置
        self.num_attention_heads = 4
        self.qk_nope_head_dim = 16
        self.v_head_dim = 16
        self.input_features = 64

        # 创建一个模拟的 KV 融合模块
        self.kv_module = nn.Linear(self.input_features, 32 * self.num_attention_heads)  # 32 = 16 + 16

        # 设置权重和偏置
        with torch.no_grad():
            self.kv_module.weight.data.normal_(0, 0.02)
            self.kv_module.bias.data.normal_(0, 0.01)

    def test_init_with_valid_parameters(self):
        """测试有效的参数初始化"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 验证基本信息
        assert virtual_v.kv_module is self.kv_module
        assert virtual_v.num_attention_heads == self.num_attention_heads
        assert virtual_v.qk_nope_head_dim == self.qk_nope_head_dim
        assert virtual_v.v_head_dim == self.v_head_dim

        # 验证权重是否正确提取
        expected_weight_shape = (self.v_head_dim * self.num_attention_heads, self.input_features)
        assert virtual_v.weight.shape == expected_weight_shape

        # 验证偏置是否正确提取
        expected_bias_shape = (self.v_head_dim * self.num_attention_heads,)
        assert virtual_v.bias.shape == expected_bias_shape

    def test_init_with_dimension_mismatch(self):
        """测试维度不匹配的错误情况"""
        # 创建一个维度不匹配的 KV 模块
        invalid_kv_module = nn.Linear(self.input_features, 30 * self.num_attention_heads)  # 30 != 32

        with pytest.raises(SpecError) as exc_info:
            VirtualVModuleFromKVFused(
                kv_module=invalid_kv_module,
                num_attention_heads=self.num_attention_heads,
                qk_nope_head_dim=self.qk_nope_head_dim,
                v_head_dim=self.v_head_dim,
            )

        assert "KV-fused module weight dimension mismatch" in str(exc_info.value)

    def test_init_with_no_bias(self):
        """测试没有偏置的 KV 模块"""
        kv_module_no_bias = nn.Linear(self.input_features, 32 * self.num_attention_heads, bias=False)

        virtual_v = VirtualVModuleFromKVFused(
            kv_module=kv_module_no_bias,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 验证没有提取偏置
        assert not hasattr(virtual_v, 'bias')

    def test_weight_extraction_logic(self):
        """测试权重提取逻辑"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 验证提取的权重是 KV 权重的 V 部分
        kv_w = self.kv_module.weight.data.view(self.num_attention_heads, 32, -1)
        expected_v_weight = kv_w[:, self.qk_nope_head_dim :, :].contiguous().view(-1, self.input_features)

        torch.testing.assert_close(virtual_v.weight.data, expected_v_weight)

    def test_bias_extraction_logic(self):
        """测试偏置提取逻辑"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 验证提取的偏置是 KV 偏置的 V 部分
        kv_b = self.kv_module.bias.data.view(self.num_attention_heads, 32)
        expected_v_bias = kv_b[:, self.qk_nope_head_dim :].contiguous().view(-1)

        torch.testing.assert_close(virtual_v.bias.data, expected_v_bias)

    @patch('torch.no_grad')
    def test_update_weights_with_existing_bias(self, mock_no_grad):
        """测试更新权重（原始模块有偏置）"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 修改虚拟模块的权重和偏置
        original_weight = virtual_v.weight.data.clone()  # noqa: F841
        original_bias = virtual_v.bias.data.clone()  # noqa: F841

        virtual_v.weight.data.normal_(0, 0.1)
        virtual_v.bias.data.normal_(0, 0.05)

        new_weight = virtual_v.weight.data.clone()
        new_bias = virtual_v.bias.data.clone()

        # 更新权重
        virtual_v.update_weights()

        # 验证 KV 模块的 V 部分权重被更新
        kv_w_updated = self.kv_module.weight.data.view(self.num_attention_heads, 32, -1)
        updated_v_weight = kv_w_updated[:, self.qk_nope_head_dim :, :]
        expected_v_weight = new_weight.view(self.num_attention_heads, self.v_head_dim, -1)

        torch.testing.assert_close(updated_v_weight, expected_v_weight)

        # 验证 KV 模块的 V 部分偏置被更新
        kv_b_updated = self.kv_module.bias.data.view(self.num_attention_heads, 32)
        updated_v_bias = kv_b_updated[:, self.qk_nope_head_dim :]
        expected_v_bias = new_bias.view(self.num_attention_heads, self.v_head_dim)

        torch.testing.assert_close(updated_v_bias, expected_v_bias)

    def test_update_weights_without_bias(self):
        """测试更新权重（原始模块无偏置，虚拟模块有新偏置）"""
        kv_module_no_bias = nn.Linear(self.input_features, 32 * self.num_attention_heads, bias=False)

        virtual_v = VirtualVModuleFromKVFused(
            kv_module=kv_module_no_bias,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 由于原始模块没有偏置，虚拟模块也不应该有偏置
        assert not hasattr(virtual_v, 'bias')

        # 修改虚拟模块的权重
        virtual_v.weight.data.normal_(0, 0.1)
        new_weight = virtual_v.weight.data.clone()

        # 更新权重
        virtual_v.update_weights()

        # 验证 KV 模块的 V 部分权重被更新
        kv_w_updated = kv_module_no_bias.weight.data.view(self.num_attention_heads, 32, -1)
        updated_v_weight = kv_w_updated[:, self.qk_nope_head_dim :, :]
        expected_v_weight = new_weight.view(self.num_attention_heads, self.v_head_dim, -1)

        torch.testing.assert_close(updated_v_weight, expected_v_weight)

    def modify_virtual_module_with_synthetic_bias(self):
        """辅助方法：修改虚拟模块并添加合成偏置"""
        kv_module_no_bias = nn.Linear(self.input_features, 32 * self.num_attention_heads, bias=False)

        virtual_v = VirtualVModuleFromKVFused(
            kv_module=kv_module_no_bias,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 手动添加偏置到虚拟模块（模拟平滑后产生偏置的情况）
        synthetic_bias = torch.randn(virtual_v.weight.shape[0])
        virtual_v.bias = nn.Parameter(synthetic_bias)

        return virtual_v, kv_module_no_bias

    def test_update_weights_device_and_dtype_preservation(self):
        """测试更新权重时设备和数据类型保持不变"""
        # 测试不同设备和数据类型
        for device in [torch.device('cpu')]:
            for dtype in [torch.float32, torch.float16]:
                kv_module = nn.Linear(self.input_features, 32 * self.num_attention_heads, bias=True, dtype=dtype).to(
                    device
                )

                virtual_v = VirtualVModuleFromKVFused(
                    kv_module=kv_module,
                    num_attention_heads=self.num_attention_heads,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    v_head_dim=self.v_head_dim,
                )

                # 修改虚拟模块的权重和偏置
                virtual_v.weight.data.normal_(0, 0.1)
                virtual_v.bias.data.normal_(0, 0.05)

                # 更新权重
                virtual_v.update_weights()

                # 验证设备和数据类型保持不变
                assert kv_module.weight.device == device
                assert kv_module.weight.dtype == dtype
                assert kv_module.bias.device == device
                assert kv_module.bias.dtype == dtype

    def test_update_weights_edge_case_zero_bias_vector(self):
        """测试零偏置向量的更新"""
        virtual_v, kv_module_no_bias = self.modify_virtual_module_with_synthetic_bias()

        # 设置零偏置
        virtual_v.bias.data.zero_()

        # 更新权重
        virtual_v.update_weights()

        # 验证 KV 模块的 V 部分偏好为零
        kv_b_updated = kv_module_no_bias.bias.data.view(self.num_attention_heads, 32)
        updated_v_bias = kv_b_updated[:, self.qk_nope_head_dim :]

        assert torch.allclose(updated_v_bias, torch.zeros_like(updated_v_bias))

    @pytest.mark.parametrize("num_heads,qk_dim,v_dim", [(1, 8, 8), (2, 12, 8), (8, 16, 16), (16, 32, 64), (32, 8, 8)])
    def test_parametric_dimensions(self, num_heads, qk_dim, v_dim):
        """参数化测试不同维度和头数组合"""
        kv_module = nn.Linear(128, (qk_dim + v_dim) * num_heads)

        virtual_v = VirtualVModuleFromKVFused(
            kv_module=kv_module, num_attention_heads=num_heads, qk_nope_head_dim=qk_dim, v_head_dim=v_dim
        )

        # 验证权重形状
        assert virtual_v.weight.shape == (v_dim * num_heads, 128)

        # 验证偏置形状
        assert virtual_v.bias.shape == (v_dim * num_heads,)

    def test_training_mode_behavior(self):
        """测试训练模式下的行为"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 测试训练模式
        virtual_v.train()
        assert virtual_v.training

        # 测试评估模式
        virtual_v.eval()
        assert not virtual_v.training

    def test_clone_and_detach_behavior(self):
        """测试克隆和分离行为"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 克隆权重
        weight_clone = virtual_v.weight.clone()

        # 分离权重
        weight_detached = virtual_v.weight.detach()

        # 验证克隆和分离的数据内容相同但对象不同
        torch.testing.assert_close(weight_clone, virtual_v.weight.data)
        torch.testing.assert_close(weight_detached, virtual_v.weight.data)

        assert weight_clone is not virtual_v.weight
        assert weight_detached is not virtual_v.weight

    def test_cpu_cuda_device_transfer(self):
        """测试 CPU-CUDA 设备转换（如果 CUDA 可用）"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # 创建 CUDA 设备上的 KV 模块
        kv_module_cuda = self.kv_module.cuda()

        virtual_v = VirtualVModuleFromKVFused(
            kv_module=kv_module_cuda,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 验证权重在 CUDA 设备上
        assert virtual_v.weight.device.type == 'cuda'

        # 转换回 CPU
        virtual_v_cpu = virtual_v.cpu()

        # 验证转换后的权重在 CPU 上
        assert virtual_v_cpu.weight.device.type == 'cpu'

    def test_print_and_str_methods(self):
        """测试打印和字符串方法"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 测试 __repr__ 和 __str__
        repr_str = repr(virtual_v)
        str_str = str(virtual_v)

        assert "VirtualVModuleFromKVFused" in repr_str
        assert "VirtualVModuleFromKVFused" in str_str

    def test_extra_repr(self):
        """测试 extra_repr 方法"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 测试 extra_repr
        extra_info = virtual_v.extra_repr()

        # 验证包含关键信息（假设基类或模块包含了维度和设备信息）
        assert isinstance(extra_info, str)

    def test_state_serialization(self):
        """测试状态序列化"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 获取状态字典
        state_dict = virtual_v.state_dict()

        # 验证状态字典包含权重和偏置
        assert 'weight' in state_dict
        assert 'bias' in state_dict

        # 验证状态字典的形状
        assert state_dict['weight'].shape == virtual_v.weight.shape
        assert state_dict['bias'].shape == virtual_v.bias.shape

    def test_buffer_handling(self):
        """测试缓冲区处理"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 注册一个缓冲区
        buffer_tensor = torch.randn(10, 10)
        virtual_v.register_buffer('test_buffer', buffer_tensor)

        # 验证缓冲区已注册
        assert 'test_buffer' in dict(virtual_v.named_buffers())

        # 验证缓冲区可访问
        retrieved_buffer = virtual_v.test_buffer
        torch.testing.assert_close(retrieved_buffer, buffer_tensor)

    def test_named_parameters_collection(self):
        """测试命名参数收集"""
        virtual_v = VirtualVModuleFromKVFused(
            kv_module=self.kv_module,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 获取命名参数
        named_params = dict(virtual_v.named_parameters())

        # 验证包含权重和偏置
        assert 'weight' in named_params
        assert 'bias' in named_params

        # 验证参数对象正确
        assert named_params['weight'] is virtual_v.weight
        assert named_params['bias'] is virtual_v.bias
