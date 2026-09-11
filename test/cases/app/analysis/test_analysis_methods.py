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

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.utils.exception import UnsupportedError


class TestAnalysisMethods(unittest.TestCase):
    """测试分析方法"""

    def test_analysis_target_matcher_get_linear_conv_layers(self):
        """测试 Std 分析方法 get_target_layers 返回 Linear/Conv2d 层名"""
        from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod

        # 使用真实子模块，以便 _matches(module) 能匹配 Linear/Conv2d
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(2, 2)
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.other1 = nn.Identity()

        model = TinyModel()
        result = StdAnalysisMethod().get_target_layers(model)

        self.assertEqual(sorted(result), sorted(['linear1', 'conv1']))

    def test_analysis_target_matcher_filter_layers_by_patterns(self):
        """测试AnalysisTargetMatcher的filter_layers_by_patterns方法"""
        from msmodelslim.processor.analysis.methods_base import AnalysisTargetMatcher

        layer_names = ['layer1.linear', 'layer2.conv', 'layer3.other']

        # 测试通配符模式
        result = AnalysisTargetMatcher.filter_layers_by_patterns(layer_names, ['*'])
        self.assertEqual(result, layer_names)

        # 测试具体模式
        result = AnalysisTargetMatcher.filter_layers_by_patterns(layer_names, ['layer1.*'])
        self.assertEqual(result, ['layer1.linear'])

        # 测试空模式
        result = AnalysisTargetMatcher.filter_layers_by_patterns(layer_names, [])
        self.assertEqual(result, layer_names)

    def test_quantile_analysis_method(self):
        """测试QuantileAnalysisMethod"""
        from msmodelslim.processor.analysis.unary_operator.metrics.quantile import QuantileAnalysisMethod

        method = QuantileAnalysisMethod(sample_step=10)

        # 测试name属性
        self.assertEqual(method.name, 'quantile')

        # 测试compute_score方法
        layer_data = {'tensor': [torch.tensor([[1.0, 2.0, 3.0, 4.0]])], 'device': torch.device('cpu')}

        method.compute_score(layer_data)

        # 测试get_hook方法
        hook = method.get_hook()
        self.assertTrue(callable(hook))

    def test_quantile_analysis_method_hook_basic_functionality(self):
        """测试QuantileAnalysisMethod.get_hook的基本功能"""
        from msmodelslim.processor.analysis.unary_operator.metrics.quantile import QuantileAnalysisMethod

        method = QuantileAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        layer_name = 'test_layer'
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        output_tensor = None
        mock_module = MagicMock()

        # 调用hook函数
        hook(mock_module, input_tensor, output_tensor, layer_name, stats_dict)

        # 验证数据存储
        self.assertIn(layer_name, stats_dict)
        layer_data = stats_dict.get(layer_name, {})
        self.assertIn('tensor', layer_data)
        self.assertIn('device', layer_data)
        self.assertEqual(len(layer_data.get('tensor', [])), 1)
        self.assertEqual(layer_data.get('device'), input_tensor.device)

    def test_quantile_analysis_method_hook_tuple_input(self):
        """测试QuantileAnalysisMethod.get_hook处理tuple输入"""
        from msmodelslim.processor.analysis.unary_operator.metrics.quantile import QuantileAnalysisMethod

        method = QuantileAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据，输入是tuple
        stats_dict = {}
        layer_name = 'test_layer'
        input_tensor = (torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), torch.tensor([[6.0, 7.0]]))
        output_tensor = None
        mock_module = MagicMock()

        # 调用hook函数
        hook(mock_module, input_tensor, output_tensor, layer_name, stats_dict)

        # 验证数据存储，取tuple的第一个元素
        self.assertIn(layer_name, stats_dict)
        layer_data = stats_dict.get(layer_name, {})
        self.assertIn('tensor', layer_data)
        self.assertEqual(len(layer_data.get('tensor', [])), 1)

    def test_quantile_analysis_method_hook_data_accumulation(self):
        """测试QuantileAnalysisMethod.get_hook数据累积行为"""
        from msmodelslim.processor.analysis.unary_operator.metrics.quantile import QuantileAnalysisMethod

        method = QuantileAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        layer_name = 'test_layer'
        mock_module = MagicMock()
        output_tensor = None

        # 第一次调用
        input_tensor1 = torch.tensor([[1.0, 2.0, 3.0]])
        hook(mock_module, input_tensor1, output_tensor, layer_name, stats_dict)

        # 验证第一次调用结果
        layer_data = stats_dict.get(layer_name, {})
        self.assertEqual(len(layer_data.get('tensor', [])), 1)

        # 第二次调用
        input_tensor2 = torch.tensor([[4.0, 5.0, 6.0]])
        hook(mock_module, input_tensor2, output_tensor, layer_name, stats_dict)

        # 验证累积结果
        layer_data = stats_dict.get(layer_name, {})
        self.assertEqual(len(layer_data.get('tensor', [])), 2)

        # 验证设备信息保持一致
        self.assertEqual(layer_data.get('device'), input_tensor1.device)

    def test_quantile_analysis_method_hook_multiple_layers(self):
        """测试QuantileAnalysisMethod.get_hook多层处理"""
        from msmodelslim.processor.analysis.unary_operator.metrics.quantile import QuantileAnalysisMethod

        method = QuantileAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        mock_module = MagicMock()
        output_tensor = None

        # 处理第一个层
        layer_name1 = 'layer1'
        input_tensor1 = torch.tensor([[1.0, 2.0, 3.0]])
        hook(mock_module, input_tensor1, output_tensor, layer_name1, stats_dict)

        # 处理第二个层
        layer_name2 = 'layer2'
        input_tensor2 = torch.tensor([[4.0, 5.0, 6.0, 7.0]])
        hook(mock_module, input_tensor2, output_tensor, layer_name2, stats_dict)

        # 验证两个层的数据都正确存储
        self.assertIn(layer_name1, stats_dict)
        self.assertIn(layer_name2, stats_dict)

        layer1_data = stats_dict.get(layer_name1, {})
        layer2_data = stats_dict.get(layer_name2, {})
        self.assertEqual(len(layer1_data.get('tensor', [])), 1)
        self.assertEqual(len(layer2_data.get('tensor', [])), 1)

        # 验证设备信息
        self.assertEqual(layer1_data.get('device'), input_tensor1.device)
        self.assertEqual(layer2_data.get('device'), input_tensor2.device)

    def test_std_analysis_method(self):
        """测试StdAnalysisMethod"""
        from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod

        method = StdAnalysisMethod()

        # 测试name属性
        self.assertEqual(method.name, 'std')

        # 测试compute_score方法
        layer_data = {'t_max': torch.tensor(5.0), 't_min': torch.tensor(1.0), 'std': torch.tensor(2.0)}

        score = method.compute_score(layer_data)
        self.assertIsInstance(score, float)

        # DP merge of packed stats must match single-card aggregation semantics
        packed_a = method.pack_stats_for_distributed_merge(
            {'t_max': torch.tensor(5.0), 't_min': torch.tensor(-1.0), 'std': torch.tensor(2.0)}
        )
        packed_b = method.pack_stats_for_distributed_merge(
            {'t_max': torch.tensor(3.0), 't_min': torch.tensor(-4.0), 'std': torch.tensor(2.5)}
        )
        merged = method.merge_distributed_stats([packed_a, packed_b])
        self.assertEqual(merged, {'t_max': 5.0, 't_min': -4.0, 'std': 2.5})
        self.assertAlmostEqual(method.compute_score(merged), 5.0 / 2.5)

        # 测试get_hook方法
        hook = method.get_hook()
        self.assertTrue(callable(hook))

    def test_std_analysis_method_hook_basic_functionality(self):
        """测试StdAnalysisMethod.get_hook的基本功能"""
        from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod

        method = StdAnalysisMethod()
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        layer_name = 'test_layer'
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        output_tensor = None
        mock_module = MagicMock()

        # 调用hook函数
        hook(mock_module, input_tensor, output_tensor, layer_name, stats_dict)

        # 验证数据存储
        self.assertIn(layer_name, stats_dict)
        layer_data = stats_dict.get(layer_name, {})
        self.assertIn('shift', layer_data)
        self.assertIn('t_max', layer_data)
        self.assertIn('t_min', layer_data)
        self.assertIn('std', layer_data)

        # 验证数据类型和形状
        self.assertEqual(layer_data.get('shift', torch.tensor([])).shape, torch.Size([5]))  # hidden_dim = 5
        self.assertIsInstance(layer_data.get('t_max'), torch.Tensor)
        self.assertIsInstance(layer_data.get('t_min'), torch.Tensor)
        self.assertIsInstance(layer_data.get('std'), torch.Tensor)

    def test_std_analysis_method_hook_tuple_input(self):
        """测试StdAnalysisMethod.get_hook处理tuple输入"""
        from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod

        method = StdAnalysisMethod()
        hook = method.get_hook()

        # 创建模拟数据，输入是tuple
        stats_dict = {}
        layer_name = 'test_layer'
        input_tensor = (torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), torch.tensor([[6.0, 7.0]]))
        output_tensor = None
        mock_module = MagicMock()

        # 调用hook函数
        hook(mock_module, input_tensor, output_tensor, layer_name, stats_dict)

        # 验证数据存储，取tuple的第一个元素
        self.assertIn(layer_name, stats_dict)
        layer_data = stats_dict.get(layer_name, {})
        self.assertIn('shift', layer_data)
        self.assertEqual(
            layer_data.get('shift', torch.tensor([])).shape, torch.Size([5])
        )  # 基于第一个tensor的hidden_dim

    def test_std_analysis_method_hook_data_accumulation(self):
        """测试StdAnalysisMethod.get_hook数据累积行为"""
        from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod

        method = StdAnalysisMethod()
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        layer_name = 'test_layer'
        mock_module = MagicMock()
        output_tensor = None

        # 第一次调用
        input_tensor1 = torch.tensor([[1.0, 2.0, 3.0]])
        hook(mock_module, input_tensor1, output_tensor, layer_name, stats_dict)

        # 记录第一次调用的结果
        layer_data = stats_dict.get(layer_name, {})
        first_max = layer_data.get('t_max', torch.tensor(0.0)).clone()
        first_min = layer_data.get('t_min', torch.tensor(0.0)).clone()

        # 第二次调用
        input_tensor2 = torch.tensor([[4.0, 5.0, 6.0]])
        hook(mock_module, input_tensor2, output_tensor, layer_name, stats_dict)

        # 验证数据累积：max应该变大，min应该变小或保持，std应该被更新
        layer_data = stats_dict.get(layer_name, {})
        second_max = layer_data.get('t_max', torch.tensor(0.0))
        second_min = layer_data.get('t_min', torch.tensor(0.0))

        # 由于第二次输入包含更大的值，t_max应该更新
        self.assertTrue(second_max >= first_max)
        # t_min应该更新
        self.assertTrue(second_min <= first_min)

    def test_std_analysis_method_hook_multiple_layers(self):
        """测试StdAnalysisMethod.get_hook多层处理"""
        from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod

        method = StdAnalysisMethod()
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        mock_module = MagicMock()
        output_tensor = None

        # 处理第一个层
        layer_name1 = 'layer1'
        input_tensor1 = torch.tensor([[1.0, 2.0, 3.0]])
        hook(mock_module, input_tensor1, output_tensor, layer_name1, stats_dict)

        # 处理第二个层
        layer_name2 = 'layer2'
        input_tensor2 = torch.tensor([[4.0, 5.0, 6.0, 7.0]])
        hook(mock_module, input_tensor2, output_tensor, layer_name2, stats_dict)

        # 验证两个层的数据都正确存储
        self.assertIn(layer_name1, stats_dict)
        self.assertIn(layer_name2, stats_dict)

        layer1_data = stats_dict.get(layer_name1, {})
        layer2_data = stats_dict.get(layer_name2, {})

        # 验证每个层都有完整的统计数据
        for _, layer_data in [(layer_name1, layer1_data), (layer_name2, layer2_data)]:
            self.assertIn('shift', layer_data)
            self.assertIn('t_max', layer_data)
            self.assertIn('t_min', layer_data)
            self.assertIn('std', layer_data)

    def test_std_analysis_method_hook_shift_calculation(self):
        """测试StdAnalysisMethod.get_hook的shift计算"""
        from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod

        method = StdAnalysisMethod()
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        layer_name = 'test_layer'
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # shape: [1, 5]
        output_tensor = None
        mock_module = MagicMock()

        # 调用hook函数
        hook(mock_module, input_tensor, output_tensor, layer_name, stats_dict)

        # 验证shift计算：shift = (tensor_max + tensor_min) / 2
        tensor_max = torch.max(input_tensor.reshape(-1, 5), dim=0)[0]  # [5.0]
        tensor_min = torch.min(input_tensor.reshape(-1, 5), dim=0)[0]  # [1.0]
        expected_shift = (tensor_max + tensor_min) / 2  # [3.0]

        layer_data = stats_dict.get(layer_name, {})
        self.assertTrue(torch.allclose(layer_data.get('shift', torch.tensor(0.0)), expected_shift))

    def test_kurtosis_analysis_method(self):
        """测试KurtosisAnalysisMethod"""
        from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import KurtosisAnalysisMethod

        method = KurtosisAnalysisMethod(sample_step=10)

        # 测试name属性
        self.assertEqual(method.name, 'kurtosis')

        # 测试compute_score方法
        layer_data = {'tensor': [torch.tensor([[1.0, 2.0, 3.0, 4.0]])], 'device': torch.device('cpu')}

        method.compute_score(layer_data)
        # 测试get_hook方法
        hook = method.get_hook()
        self.assertTrue(callable(hook))

    def test_kurtosis_analysis_method_hook_basic_functionality(self):
        """测试KurtosisAnalysisMethod.get_hook的基本功能"""
        from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import KurtosisAnalysisMethod

        method = KurtosisAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        layer_name = 'test_layer'
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        output_tensor = None
        mock_module = MagicMock()

        # 调用hook函数
        hook(mock_module, input_tensor, output_tensor, layer_name, stats_dict)

        # 验证数据存储
        self.assertIn(layer_name, stats_dict)
        layer_data = stats_dict.get(layer_name, {})
        self.assertIn('tensor', layer_data)
        self.assertIn('device', layer_data)
        self.assertEqual(len(layer_data.get('tensor', [])), 1)
        self.assertEqual(layer_data.get('device'), input_tensor.device)

    def test_kurtosis_analysis_method_hook_tuple_input(self):
        """测试KurtosisAnalysisMethod.get_hook处理tuple输入"""
        from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import KurtosisAnalysisMethod

        method = KurtosisAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据，输入是tuple
        stats_dict = {}
        layer_name = 'test_layer'
        input_tensor = (torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), torch.tensor([[6.0, 7.0]]))
        output_tensor = None
        mock_module = MagicMock()

        # 调用hook函数
        hook(mock_module, input_tensor, output_tensor, layer_name, stats_dict)

        # 验证数据存储，取tuple的第一个元素
        self.assertIn(layer_name, stats_dict)
        layer_data = stats_dict.get(layer_name, {})
        self.assertIn('tensor', layer_data)
        self.assertEqual(len(layer_data.get('tensor', [])), 1)

    def test_kurtosis_analysis_method_hook_data_accumulation(self):
        """测试KurtosisAnalysisMethod.get_hook数据累积行为"""
        from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import KurtosisAnalysisMethod

        method = KurtosisAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        layer_name = 'test_layer'
        mock_module = MagicMock()
        output_tensor = None

        # 第一次调用
        input_tensor1 = torch.tensor([[1.0, 2.0, 3.0]])
        hook(mock_module, input_tensor1, output_tensor, layer_name, stats_dict)

        # 验证第一次调用结果
        layer_data = stats_dict.get(layer_name, {})
        self.assertEqual(len(layer_data.get('tensor', [])), 1)

        # 第二次调用
        input_tensor2 = torch.tensor([[4.0, 5.0, 6.0]])
        hook(mock_module, input_tensor2, output_tensor, layer_name, stats_dict)

        # 验证累积结果
        layer_data = stats_dict.get(layer_name, {})
        self.assertEqual(len(layer_data.get('tensor', [])), 2)

        # 验证设备信息保持一致
        self.assertEqual(layer_data.get('device'), input_tensor1.device)

    def test_kurtosis_analysis_method_hook_multiple_layers(self):
        """测试KurtosisAnalysisMethod.get_hook多层处理"""
        from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import KurtosisAnalysisMethod

        method = KurtosisAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        mock_module = MagicMock()
        output_tensor = None

        # 处理第一个层
        layer_name1 = 'layer1'
        input_tensor1 = torch.tensor([[1.0, 2.0, 3.0]])
        hook(mock_module, input_tensor1, output_tensor, layer_name1, stats_dict)

        # 处理第二个层
        layer_name2 = 'layer2'
        input_tensor2 = torch.tensor([[4.0, 5.0, 6.0, 7.0]])
        hook(mock_module, input_tensor2, output_tensor, layer_name2, stats_dict)

        # 验证两个层的数据都正确存储
        self.assertIn(layer_name1, stats_dict)
        self.assertIn(layer_name2, stats_dict)

        layer1_data = stats_dict.get(layer_name1, {})
        layer2_data = stats_dict.get(layer_name2, {})
        self.assertEqual(len(layer1_data.get('tensor', [])), 1)
        self.assertEqual(len(layer2_data.get('tensor', [])), 1)

        # 验证设备信息
        self.assertEqual(layer1_data.get('device'), input_tensor1.device)
        self.assertEqual(layer2_data.get('device'), input_tensor2.device)

    def test_kurtosis_analysis_method_hook_sorting_behavior(self):
        """测试KurtosisAnalysisMethod.get_hook的排序行为"""
        from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import KurtosisAnalysisMethod

        method = KurtosisAnalysisMethod(sample_step=10)
        hook = method.get_hook()

        # 创建模拟数据
        stats_dict = {}
        layer_name = 'test_layer'
        input_tensor = torch.tensor([[3.0, 1.0, 5.0, 2.0, 4.0]])  # 无序张量
        output_tensor = None
        mock_module = MagicMock()

        # 调用hook函数
        hook(mock_module, input_tensor, output_tensor, layer_name, stats_dict)

        # 验证排序行为：存储的张量应该是排序后的
        layer_data = stats_dict.get(layer_name, {})
        stored_tensor = layer_data.get('tensor', [torch.tensor([])])[0]
        stored_values = stored_tensor.squeeze().tolist()

        # 验证排序：应该是从小到大排序
        self.assertEqual(stored_values, sorted([3.0, 1.0, 5.0, 2.0, 4.0]))

    def test_attention_mse_set_name_and_hook_when_adapter_valid(self):
        """测试AttentionMSEAnalysisMethod"""
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.impl import AttentionMSEAnalysisMethod
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
            AttentionMSEAnalysisInterface,
        )

        class FakeAdapter(AttentionMSEAnalysisInterface):
            def get_attention_module_cls(self) -> str:
                return 'FakeAttention'

            def get_attention_output_extractor(self):
                return lambda output: output[0] if isinstance(output, tuple) else output

        method = AttentionMSEAnalysisMethod(adapter=FakeAdapter())

        self.assertEqual(method.name, 'mse')
        self.assertEqual(method.adapter.get_attention_module_cls(), 'FakeAttention')
        hook = method.get_hook()
        self.assertTrue(callable(hook))

    def test_attention_mse_raise_unsupported_error_when_adapter_invalid(self):
        """测试AttentionMSEAnalysisMethod要求adapter实现接口"""
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse import AttentionMSEAnalysisMethod

        with self.assertRaises(UnsupportedError):
            AttentionMSEAnalysisMethod(adapter=object())

    def test_attention_mse_return_score_when_inputs_valid(self):
        """测试AttentionMSEAnalysisMethod.compute_score"""
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse import AttentionMSEAnalysisMethod
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
            AttentionMSEAnalysisInterface,
        )

        class FakeAdapter(AttentionMSEAnalysisInterface):
            def get_attention_module_cls(self) -> str:
                return 'FakeAttention'

            def get_attention_output_extractor(self):
                pass

        method = AttentionMSEAnalysisMethod(adapter=FakeAdapter())

        layer_data_before = {
            'attn_output': [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[0.0, 1.0], [2.0, 3.0]])]
        }
        layer_data_after = {
            'attn_output': [torch.tensor([[1.0, 1.0], [2.0, 5.0]]), torch.tensor([[1.0, 1.0], [1.0, 1.0]])]
        }

        expected = (
            torch.stack(
                [
                    torch.nn.functional.mse_loss(
                        layer_data_before['attn_output'][0], layer_data_after['attn_output'][0]
                    ),
                    torch.nn.functional.mse_loss(
                        layer_data_before['attn_output'][1], layer_data_after['attn_output'][1]
                    ),
                ]
            )
            .mean()
            .item()
        )

        score = method.compute_score(layer_data_before, layer_data_after)

        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, expected)

    def test_attention_mse_hook_accumulate_outputs_when_same_layer_called_twice(self):
        """测试AttentionMSEAnalysisMethod.get_hook数据累积行为"""
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse import AttentionMSEAnalysisMethod
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
            AttentionMSEAnalysisInterface,
        )

        class FakeAdapter(AttentionMSEAnalysisInterface):
            def get_attention_module_cls(self) -> str:
                return 'FakeAttention'

            def get_attention_output_extractor(self):
                return lambda output: output[0] if isinstance(output, tuple) else output

        method = AttentionMSEAnalysisMethod(adapter=FakeAdapter())
        hook = method.get_hook()

        stats_dict = {}
        layer_name = 'test_layer'
        mock_module = MagicMock()
        input_tensor = None

        output_tensor1 = (torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), 'ignored')
        hook(mock_module, input_tensor, output_tensor1, layer_name, stats_dict)

        layer_data = stats_dict.get(layer_name, {})
        self.assertEqual(len(layer_data.get('attn_output', [])), 1)

        output_tensor2 = (torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), 'ignored')
        hook(mock_module, input_tensor, output_tensor2, layer_name, stats_dict)

        layer_data = stats_dict.get(layer_name, {})
        self.assertEqual(len(layer_data.get('attn_output', [])), 2)
        self.assertEqual(layer_data['attn_output'][0].device.type, 'cpu')
        self.assertEqual(layer_data['attn_output'][1].device.type, 'cpu')

    def test_attention_mse_hook_store_outputs_when_multiple_layers_given(self):
        """测试AttentionMSEAnalysisMethod.get_hook多层处理"""
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse import AttentionMSEAnalysisMethod
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
            AttentionMSEAnalysisInterface,
        )

        class FakeAdapter(AttentionMSEAnalysisInterface):
            def get_attention_module_cls(self) -> str:
                return 'FakeAttention'

            def get_attention_output_extractor(self):
                return lambda output: output

        method = AttentionMSEAnalysisMethod(adapter=FakeAdapter())
        hook = method.get_hook()

        stats_dict = {}
        mock_module = MagicMock()
        input_tensor = None

        hook(mock_module, input_tensor, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), 'layer1', stats_dict)
        hook(mock_module, input_tensor, torch.tensor([[[5.0, 6.0, 7.0]]]), 'layer2', stats_dict)

        self.assertIn('layer1', stats_dict)
        self.assertIn('layer2', stats_dict)
        self.assertEqual(len(stats_dict['layer1'].get('attn_output', [])), 1)
        self.assertEqual(len(stats_dict['layer2'].get('attn_output', [])), 1)
        self.assertEqual(stats_dict['layer1']['attn_output'][0].shape, torch.Size([2, 2]))
        self.assertEqual(stats_dict['layer2']['attn_output'][0].shape, torch.Size([1, 3]))

    def test_attention_mse_return_match_result_when_module_class_name_checked(self):
        """测试AttentionMSEAnalysisMethod._matches按类名匹配attention模块"""
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.impl import AttentionMSEAnalysisMethod
        from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
            AttentionMSEAnalysisInterface,
        )

        class FakeAdapter(AttentionMSEAnalysisInterface):
            def get_attention_module_cls(self) -> str:
                return 'FakeAttention'

            def get_attention_output_extractor(self):
                return lambda output: output

        class FakeAttention(nn.Module):
            pass

        class OtherModule(nn.Module):
            pass

        method = AttentionMSEAnalysisMethod(adapter=FakeAdapter())

        self.assertTrue(method._matches(FakeAttention()))
        self.assertFalse(method._matches(OtherModule()))

    def test_analysis_method_factory(self):
        """测试AnalysisMethodFactory（unary/binary factories）"""
        from msmodelslim.processor.analysis.unary_operator.metrics.factory import UnaryAnalysisMethodFactory

        # 测试create_method方法
        method = UnaryAnalysisMethodFactory.create_method('std')
        self.assertEqual(method.name, 'std')

        # 测试无效方法名
        with self.assertRaises(UnsupportedError):
            UnaryAnalysisMethodFactory.create_method('invalid_method')

        # 测试register_method方法
        from msmodelslim.processor.analysis.methods_base import LayerAnalysisMethod

        class TestMethod(LayerAnalysisMethod):
            @property
            def name(self):
                return 'test'

            def get_hook(self):
                return lambda: None

            def compute_score(self, layer_data):
                return 0.0

        UnaryAnalysisMethodFactory.register_method('test', TestMethod)
        method = UnaryAnalysisMethodFactory.create_method('test')
        self.assertIsInstance(method, TestMethod)

        # 测试get_supported_methods方法
        supported = UnaryAnalysisMethodFactory.get_supported_methods()
        self.assertIn('std', supported)
        self.assertIn('test', supported)

    def test_kurtosis_function(self):
        """测试kurtosis函数"""
        from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import kurtosis

        # 创建测试张量
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        result = kurtosis(x)
        self.assertIsInstance(result, torch.Tensor)

        # 测试带维度的kurtosis
        x_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = kurtosis(x_2d, dim=0)
        self.assertIsInstance(result, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
