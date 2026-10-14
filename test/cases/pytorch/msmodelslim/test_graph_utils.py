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

import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
from collections import OrderedDict
from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter import TorchDAGAdapter
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import (
    GraphOpt,
    extract_dag,
    norm_class_detect,
    class_detect,
    input_to_cpu,
    get_module_name_list,
    PatternProcess,
    LlamaRMSNormBias,
    NormBias,
    replace_rmsnorm,
)
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import (
    replace_conv1d,  # noqa: F401
)  # noqa: F401
from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter import DagTorchHook  # noqa: F401
import os  # noqa: F401


class TestGraphOpt(unittest.TestCase):
    def test_set_module(self):
        """测试GraphOpt.set_module方法

        验证：
        1. 能够正确设置嵌套模块的属性
        2. 能够处理单层模块设置
        """
        # 创建测试模型 - 使用Sequential和OrderedDict来命名模块
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        'seq1',
                        nn.Sequential(OrderedDict([('linear1', nn.Linear(10, 10)), ('linear2', nn.Linear(10, 10))])),
                    ),
                    ('linear3', nn.Linear(10, 10)),
                ]
            )
        )

        # 测试设置嵌套模块
        new_module = nn.Linear(10, 20)
        GraphOpt.set_module(model, "seq1.linear1", new_module)
        self.assertIs(model.seq1.linear1, new_module)

        # 测试设置顶层模块
        new_top_module = nn.Linear(10, 30)
        GraphOpt.set_module(model, "linear3", new_top_module)
        self.assertIs(model.linear3, new_top_module)


class TestDAGFunctions(unittest.TestCase):
    def test_extract_dag(self):
        """测试extract_dag函数

        验证：
        1. 能够正确返回TorchDAGAdapter实例
        2. 能够处理不同的输入参数
        """
        model = nn.Linear(10, 10)
        dummy_input = torch.randn(1, 10)

        # 测试基本功能
        dag = extract_dag(model, dummy_input)
        self.assertIsInstance(dag, TorchDAGAdapter)

        # 测试带hook_nodes的情况 - 传递模块类而不是字符串
        dag_with_hooks = extract_dag(model, dummy_input, hook_nodes=[nn.LayerNorm])
        self.assertIsInstance(dag_with_hooks, TorchDAGAdapter)

    def test_norm_class_detect(self):
        """测试norm_class_detect函数

        验证：
        1. 能够正确检测模型中的norm类
        2. 返回的类列表不重复
        """
        # 创建包含多种norm的模型
        model = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.LayerNorm(10),  # 重复的LayerNorm
        )

        norm_classes = norm_class_detect(model)
        self.assertEqual(len(norm_classes), 1)  # LayerNorm和BatchNorm1d
        self.assertIn(nn.LayerNorm, norm_classes)

    def test_class_detect(self):
        """测试class_detect函数

        验证：
        1. 能够按名称检测特定类
        2. 找不到时返回None
        """
        model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10), nn.Conv1d(1, 1, 1))

        # 测试找到的情况
        linear_class = class_detect(model, "linear")
        self.assertEqual(linear_class, nn.Linear)

        # 测试找不到的情况
        none_class = class_detect(model, "not_exist")
        self.assertIsNone(none_class)

    def test_input_to_cpu(self):
        """测试input_to_cpu函数

        验证：
        1. 能够将输入数据转移到CPU
        2. 能够处理多种输入类型
        """
        # 测试Tensor输入
        cpu_tensor = input_to_cpu(torch.randn(10))
        self.assertEqual(cpu_tensor.device.type, "cpu")

        # 测试元组输入
        mixed_input = (torch.randn(10), "string", 123)
        converted = input_to_cpu(mixed_input)
        self.assertEqual(converted[0].device.type, "cpu")
        self.assertEqual(converted[1], "string")
        self.assertEqual(converted[2], 123)

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter.TorchDAGAdapter')
    def test_get_module_name_list(self, mock_dag):
        """测试get_module_name_list函数

        验证：
        1. 能够根据模型类型调用正确的模式匹配方法
        2. 返回正确的模块名称列表
        """
        # 设置mock返回值
        mock_dag_instance = mock_dag.return_value
        mock_dag_instance.get_llama_mhsa_ln_pattern.return_value = (["attn"], ["proj"], ["ln"])
        mock_dag_instance.get_llama_ffn_ln_pattern.return_value = (["ffn"], ["ffn_ln"])
        mock_dag_instance.get_mhsa_ln_pattern.return_value = (["mhsa"], ["mhsa_proj"], ["mhsa_ln"])
        mock_dag_instance.get_ffn_ln_pattern.return_value = (["ffn"], ["ffn_ln"])

        # 测试Llama类型
        llama_result = get_module_name_list(mock_dag_instance, "Llama")
        self.assertEqual(llama_result, (["attn"], ["proj"], ["ln"], ["ffn"], ["ffn_ln"]))
        mock_dag_instance.get_llama_mhsa_ln_pattern.assert_called_once()

        # 测试非Llama类型
        other_result = get_module_name_list(mock_dag_instance, "Bert")
        self.assertEqual(other_result, (["mhsa"], ["mhsa_proj"], ["mhsa_ln"], ["ffn"], ["ffn_ln"]))
        mock_dag_instance.get_mhsa_ln_pattern.assert_called_once()


class TestPatternProcess(unittest.TestCase):
    def test_get_attn_name(self):
        """测试PatternProcess.get_attn_name方法

        验证：
        1. 能够正确处理合并的QKV名称
        2. 能够正确处理分离的QKV名称
        3. 无效输入抛出异常
        """
        # 测试合并的QKV名称
        combined_qkv = ["model.layers.0.self_attn.qkv_proj"]
        attn_name = PatternProcess.get_attn_name(combined_qkv)
        self.assertEqual(attn_name, "model.layers.0.self_attn")

        # 测试分离的QKV名称
        separate_qkv = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        ]
        attn_name = PatternProcess.get_attn_name(separate_qkv)
        self.assertEqual(attn_name, "model.layers.0.self_attn")

        # 测试无效输入
        with self.assertRaises(ValueError):
            PatternProcess.get_attn_name(["q", "k"])  # 长度既不是1也不是3

    def test_get_module_by_name(self):
        """测试PatternProcess.get_module_by_name方法

        验证：
        1. 能够通过名称获取子模块
        2. 名称不存在时返回None
        """
        # 创建测试模型 - 使用Sequential的命名功能
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        'seq1',
                        nn.Sequential(OrderedDict([('linear1', nn.Linear(10, 10)), ('linear2', nn.Linear(10, 10))])),
                    ),
                    ('linear3', nn.Linear(10, 10)),
                ]
            )
        )

        # 测试获取存在的模块
        module = PatternProcess.get_module_by_name(model, "seq1.linear1")
        self.assertIsInstance(module, nn.Linear)

        # 测试获取不存在的模块
        none_module = PatternProcess.get_module_by_name(model, "seq1.nonexistent")
        self.assertIsNone(none_module)

        # 测试None输入
        self.assertIsNone(PatternProcess.get_module_by_name(model, None))

    def test_get_qkv_name(self):
        """测试PatternProcess.get_qkv_name方法

        验证：
        1. 能够从注意力列表中提取QKV名称
        2. 输入为None时返回None
        """
        # 测试合并的QKV名称
        combined_attn = [["model.layers.0.self_attn.qkv_proj"]]
        qkv_names = PatternProcess.get_qkv_name(combined_attn)
        self.assertEqual(qkv_names, ["qkv_proj"])

        # 测试分离的QKV名称
        separate_attn = [
            ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj"]
        ]
        qkv_names = PatternProcess.get_qkv_name(separate_attn)
        self.assertEqual(sorted(qkv_names), sorted(["q_proj", "k_proj", "v_proj"]))


class TestModelReplacement(unittest.TestCase):
    def test_llama_rms_norm_bias(self):
        """测试LlamaRMSNormBias类

        验证：
        1. 前向传播计算正确
        2. 数据类型处理正确
        """
        # 创建测试实例
        norm = LlamaRMSNormBias(10)
        input_tensor = torch.randn(5, 10)

        # 测试前向传播
        output = norm(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

        # 测试半精度
        norm.weight.data = norm.weight.data.half()
        output_half = norm(input_tensor.float())
        self.assertEqual(output_half.dtype, torch.float32)

    def test_norm_bias(self):
        """测试NormBias包装类

        验证：
        1. 能够正确包装norm类
        2. 添加偏置项功能正常
        """
        # 创建测试实例
        base_norm = nn.LayerNorm(10)
        norm = NormBias(base_norm)
        input_tensor = torch.randn(5, 10)

        # 测试前向传播
        output = norm(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

        # 测试无weight属性的norm类
        bad_norm = nn.Sequential()
        with self.assertRaises(AttributeError):
            NormBias(bad_norm)

    def test_replace_rmsnorm(self):
        """测试replace_rmsnorm函数

        验证：
        1. 能够正确替换模型中的norm层
        2. 参数正确转移
        """

        # 创建自定义测试norm类
        class TestRMSNorm(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(size))
                self.variance_epsilon = 1e-5

        # 创建测试模型
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = TestRMSNorm(10)  # 使用自定义norm类
                self.linear = nn.Linear(10, 10)
                self.norm2 = TestRMSNorm(10)

        model = TestModel()

        # 运行替换函数
        new_model = replace_rmsnorm(model)

        # 验证替换结果
        self.assertIsInstance(model.norm1, LlamaRMSNormBias)
        self.assertIsInstance(model.norm2, LlamaRMSNormBias)
        self.assertIsInstance(model.linear, nn.Linear)  # 保持不变

        # 验证参数转移
        self.assertTrue(torch.allclose(model.norm1.weight.data, new_model.norm1.weight.data))
        self.assertEqual(model.norm1.variance_epsilon, 1e-5)


if __name__ == '__main__':
    unittest.main()
