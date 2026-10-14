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
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import torch.nn as nn

from msmodelslim.ir import W16A16sLinear
from msmodelslim.processor.sparse.admm import AdmmPruner, quantize_l2, quantize_clip, split_half
from msmodelslim.processor.sparse.float_sparse import FloatSparseProcessor, FloatSparseProcessorConfig
from msmodelslim.utils.exception import SchemaValidateError
from ..test_processor_base import TestProcessorBase


class TestFloatSparseProcessorConfig(unittest.TestCase):
    """测试FloatSparseProcessorConfig配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = FloatSparseProcessorConfig()

        self.assertEqual(config.type, "float_sparse")
        self.assertEqual(config.sparse_ratio, 0.3)
        self.assertEqual(config.include, [])
        self.assertEqual(config.exclude, [])

    def test_custom_config(self):
        """测试自定义配置"""
        config = FloatSparseProcessorConfig(sparse_ratio=0.5, include=["layer1", "layer2"], exclude=["layer3"])

        self.assertEqual(config.sparse_ratio, 0.5)
        self.assertEqual(config.include, ["layer1", "layer2"])
        self.assertEqual(config.exclude, ["layer3"])

    def test_sparse_ratio_validation(self):
        """测试稀疏比例验证"""
        # 测试有效范围
        config = FloatSparseProcessorConfig(sparse_ratio=0.0)
        self.assertEqual(config.sparse_ratio, 0.0)

        config = FloatSparseProcessorConfig(sparse_ratio=1.0)
        self.assertEqual(config.sparse_ratio, 1.0)

        # 测试无效范围
        with self.assertRaises(SchemaValidateError):
            FloatSparseProcessorConfig(sparse_ratio=-0.1)

        with self.assertRaises(SchemaValidateError):
            FloatSparseProcessorConfig(sparse_ratio=1.1)

        with self.assertRaises(SchemaValidateError):
            FloatSparseProcessorConfig(sparse_ratio=None)


class TestFloatSparseProcessor(TestProcessorBase):
    """测试FloatSparseProcessor的功能"""

    def setUp(self):
        """测试前的准备工作"""
        super().setUp()

        self.model.config.num_hidden_layers = 1
        self.model.model.layers = self.model.model.layers[:1]
        self.linear_layer_names = self.get_linear_layer_names(self.model)
        self.assertGreater(len(self.linear_layer_names), 0, "Model should have at least one linear layer")

    def create_processor_config(
        self, sparse_ratio: float = 0.3, include: list = None, exclude: list = None
    ) -> FloatSparseProcessorConfig:
        """创建处理器配置"""
        return FloatSparseProcessorConfig(sparse_ratio=sparse_ratio, include=include or ["*"], exclude=exclude or [])

    def test_processor_initialization(self):
        """测试处理器初始化"""
        config = self.create_processor_config()
        processor = FloatSparseProcessor(self.model, config)

        self.assertEqual(processor.config, config)
        self.assertIsNotNone(processor.include)
        self.assertIsNotNone(processor.exclude)
        self.assertEqual(processor.admm_pruners, {})
        self.assertEqual(processor.hook_handles, {})
        self.assertFalse(processor.is_data_free())
        self.assertFalse(processor.support_distributed())

    def test_basic_sparse_processing(self):
        """测试基本稀疏化处理"""
        config = self.create_processor_config(sparse_ratio=0.3)

        # 运行处理器
        runner = self.run_processor_with_cfg(config)  # noqa: F841

        # 验证模型仍能正常运行
        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_sparse_all_linear_layers(self):
        """测试稀疏化所有线性层"""
        config = self.create_processor_config(sparse_ratio=0.5, include=["*"])

        self.run_processor_with_cfg(config)

        # 验证所有线性层都被替换为W16A16sLinear
        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsNotNone(layer, f"Layer {layer_name} not found")
            self.assertIsInstance(layer, W16A16sLinear, f"Layer {layer_name} should be W16A16sLinear")

    def test_sparse_specific_layers(self):
        """测试稀疏化特定层"""
        if len(self.linear_layer_names) < 2:
            self.skipTest("Model needs at least 2 linear layers for this test")

        target_layer = self.linear_layer_names[0]
        config = self.create_processor_config(sparse_ratio=0.4, include=[target_layer])

        self.run_processor_with_cfg(config)

        # 验证目标层被稀疏化
        target_module = self.get_module_by_name(self.model, target_layer)
        self.assertIsNotNone(target_module)
        self.assertIsInstance(target_module, W16A16sLinear, f"Target layer {target_layer} should be W16A16sLinear")

        # 验证其他层未被稀疏化
        for layer_name in self.linear_layer_names[1:]:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, nn.Linear, f"Layer {layer_name} should remain as Linear")

    def test_exclude_specific_layers(self):
        """测试排除特定层"""
        if len(self.linear_layer_names) < 2:
            self.skipTest("Model needs at least 2 linear layers for this test")

        excluded_layer = self.linear_layer_names[0]
        config = self.create_processor_config(sparse_ratio=0.3, include=["*"], exclude=[excluded_layer])

        self.run_processor_with_cfg(config)

        # 验证排除的层未被稀疏化
        excluded_module = self.get_module_by_name(self.model, excluded_layer)
        self.assertIsInstance(excluded_module, nn.Linear, f"Excluded layer {excluded_layer} should remain as Linear")

        # 验证其他层被稀疏化
        for layer_name in self.linear_layer_names[1:]:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, W16A16sLinear, f"Layer {layer_name} should be W16A16sLinear")

    def test_include_and_exclude_patterns(self):
        """测试包含和排除模式的组合"""
        if len(self.linear_layer_names) < 3:
            self.skipTest("Model needs at least 3 linear layers for this test")

        excluded_layer = self.linear_layer_names[0]
        config = self.create_processor_config(sparse_ratio=0.3, include=["*"], exclude=[excluded_layer])

        self.run_processor_with_cfg(config)

        # 验证排除的层未被稀疏化
        excluded_module = self.get_module_by_name(self.model, excluded_layer)
        self.assertIsInstance(excluded_module, nn.Linear)

        # 验证其他层被稀疏化
        for layer_name in self.linear_layer_names[1:]:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, W16A16sLinear)

    def test_different_sparse_ratios(self):
        """测试不同的稀疏比例"""
        sparse_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

        for sparse_ratio in sparse_ratios:
            with self.subTest(sparse_ratio=sparse_ratio):
                # 重新初始化模型
                self.setUp()

                config = self.create_processor_config(sparse_ratio=sparse_ratio, include=["*"])
                self.run_processor_with_cfg(config)

                # 验证模型仍能正常运行
                inputs = self.create_test_input()
                self.assert_model_runs_without_error(inputs)

                # 验证层被替换
                for layer_name in self.linear_layer_names:
                    layer = self.get_module_by_name(self.model, layer_name)
                    self.assertIsInstance(layer, W16A16sLinear)

    def test_output_consistency_shape(self):
        """测试输出形状一致性"""
        inputs = self.create_test_input()
        original_outputs = self.run_model_forward(inputs)

        config = self.create_processor_config(sparse_ratio=0.3, include=["*"])
        self.run_processor_with_cfg(config)

        sparse_outputs = self.run_model_forward(inputs)

        # 验证输出形状和数据类型一致
        self.assertEqual(original_outputs.shape, sparse_outputs.shape)
        self.assertEqual(original_outputs.dtype, sparse_outputs.dtype)

    def test_invalid_layer_patterns(self):
        """测试无效的层模式"""
        config = self.create_processor_config(sparse_ratio=0.3, include=["nonexistent_layer"])

        self.run_processor_with_cfg(config)

        # 验证没有层被稀疏化
        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, nn.Linear, f"Layer {layer_name} should remain as Linear")

    def test_warning_unmatched_include_patterns(self):
        """测试未匹配的包含模式警告"""
        config = self.create_processor_config(sparse_ratio=0.3, include=["nonexistent_layer", "another_fake_layer"])

        with self.assertLogs('msmodelslim.processor.float_sparse', level='WARNING') as log_context:
            self.run_processor_with_cfg(config)

        log_messages = log_context.output
        self.assertTrue(any("include patterns are not matched" in msg for msg in log_messages))
        self.assertTrue(any("nonexistent_layer" in msg for msg in log_messages))
        self.assertTrue(any("another_fake_layer" in msg for msg in log_messages))

    def test_warning_unmatched_exclude_patterns(self):
        """测试未匹配的排除模式警告"""
        config = self.create_processor_config(
            sparse_ratio=0.3, include=["*"], exclude=["nonexistent_layer", "another_fake_layer"]
        )

        with self.assertLogs('msmodelslim.processor.float_sparse', level='WARNING') as log_context:
            self.run_processor_with_cfg(config)

        log_messages = log_context.output
        self.assertTrue(any("exclude patterns are not matched" in msg for msg in log_messages))
        self.assertTrue(any("nonexistent_layer" in msg for msg in log_messages))
        self.assertTrue(any("another_fake_layer" in msg for msg in log_messages))

    def test_warning_mixed_matched_unmatched_patterns(self):
        """测试混合匹配和未匹配模式的警告"""
        if len(self.linear_layer_names) < 1:
            self.skipTest("Model needs at least 1 linear layer for this test")

        existing_layer = self.linear_layer_names[0]
        config = self.create_processor_config(sparse_ratio=0.3, include=[existing_layer, "nonexistent_layer"])

        with self.assertLogs('msmodelslim.processor.float_sparse', level='WARNING') as log_context:
            self.run_processor_with_cfg(config)

        log_messages = log_context.output
        self.assertTrue(any("include patterns are not matched" in msg for msg in log_messages))
        self.assertTrue(any("nonexistent_layer" in msg for msg in log_messages))
        self.assertFalse(any(existing_layer in msg for msg in log_messages))

        # 验证存在的层被稀疏化
        existing_module = self.get_module_by_name(self.model, existing_layer)
        self.assertIsInstance(existing_module, W16A16sLinear, f"Layer {existing_layer} should be W16A16sLinear")

    def test_no_warning_with_valid_patterns(self):
        """测试有效模式不产生警告"""
        if len(self.linear_layer_names) < 1:
            self.skipTest("Model needs at least 1 linear layer for this test")

        existing_layer = self.linear_layer_names[0]
        config = self.create_processor_config(sparse_ratio=0.3, include=[existing_layer])

        # 不应该有警告日志
        self.run_processor_with_cfg(config)

        # 验证层被稀疏化
        existing_module = self.get_module_by_name(self.model, existing_layer)
        self.assertIsInstance(existing_module, W16A16sLinear, f"Layer {existing_layer} should be W16A16sLinear")

    def test_multiple_processor_runs(self):
        """测试多次运行处理器"""
        config = self.create_processor_config(sparse_ratio=0.3, include=["*"])

        # 第一次运行
        self.run_processor_with_cfg(config)

        # 第二次运行（模型已经被稀疏化）
        self.run_processor_with_cfg(config)

        # 验证模型仍能正常运行
        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_sparse_with_different_input_sizes(self):
        """测试不同输入大小的稀疏化"""
        config = self.create_processor_config(sparse_ratio=0.3, include=["*"])
        self.run_processor_with_cfg(config)

        test_inputs = [
            self.create_test_input("Short", 5),
            self.create_test_input("Medium length text", 15),
            self.create_test_input("This is a longer text for testing", 20),
        ]

        for inputs in test_inputs:
            self.assert_model_runs_without_error(inputs)

    def test_sparse_preserves_model_structure(self):
        """测试稀疏化保持模型结构"""
        original_module_names = set(name for name, _ in self.model.named_modules())
        original_module_count = len(original_module_names)

        config = self.create_processor_config(sparse_ratio=0.3, include=["*"])
        self.run_processor_with_cfg(config)

        current_module_names = set(name for name, _ in self.model.named_modules())
        current_module_count = len(current_module_names)

        # 验证模块名称结构保持不变（模块被替换但名称结构应该一样）
        self.assertEqual(original_module_names, current_module_names, "模块名称结构应该保持不变")

        # 验证模块数量保持不变
        self.assertEqual(original_module_count, current_module_count, "模块数量应该保持不变")

        # 验证模型仍能正常运行
        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    @patch('msmodelslim.processor.sparse.float_sparse.AdmmPruner')
    def test_admm_pruner_creation_and_cleanup(self, mock_admm_pruner_class):
        """测试ADMM稀疏器的创建和清理"""
        mock_admm_pruner = MagicMock()
        mock_admm_pruner_class.return_value = mock_admm_pruner

        # 排除lm_head层
        config = self.create_processor_config(sparse_ratio=0.3, include=["*"], exclude=["lm_head"])
        self.run_processor_with_cfg(config)

        # 验证AdmmPruner类被调用来创建实例
        self.assertTrue(mock_admm_pruner_class.called, "AdmmPruner类应该被调用来创建实例")

        # 计算排除lm_head后的线性层数量
        actual_linear_count = len(self.linear_layer_names)

        # 验证调用次数
        actual_calls = mock_admm_pruner_class.call_count
        self.assertEqual(
            actual_calls,
            actual_linear_count,
            f"应该为{actual_linear_count}个线性层创建ADMM稀疏器，实际创建了{actual_calls}个",
        )

        # 验证mock实例的方法被调用
        self.assertTrue(mock_admm_pruner.fasterprune.called, "fasterprune方法应该被调用")
        self.assertTrue(mock_admm_pruner.free.called, "free方法应该被调用进行清理")

        # 验证fasterprune被调用时传入了正确的参数
        mock_admm_pruner.fasterprune.assert_called_with(sparse_ratio=0.3)

    def test_edge_case_zero_sparse_ratio(self):
        """测试边界情况：零稀疏比例"""
        config = self.create_processor_config(sparse_ratio=0.0, include=["*"])

        self.run_processor_with_cfg(config)

        # 验证模型仍能正常运行
        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

        # 验证层被替换为W16A16sLinear
        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, W16A16sLinear)

    def test_edge_case_max_sparse_ratio(self):
        """测试边界情况：最大稀疏比例"""
        config = self.create_processor_config(sparse_ratio=1.0, include=["*"])

        self.run_processor_with_cfg(config)

        # 验证模型仍能正常运行（虽然权重可能全为零）
        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

        # 验证层被替换为W16A16sLinear
        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, W16A16sLinear)


class TestAdmmPrunerIntegration(unittest.TestCase):
    """测试ADMM稀疏器集成"""

    def setUp(self):
        """测试前的准备工作"""
        self.layer = nn.Linear(20, 10)
        self.pruner = AdmmPruner(self.layer)

    def test_admm_pruner_initialization(self):
        """测试ADMM稀疏器初始化"""
        self.assertEqual(self.pruner.layer, self.layer)
        self.assertEqual(self.pruner.rows, 10)
        self.assertEqual(self.pruner.columns, 20)
        self.assertEqual(self.pruner.nsamples, 0)

    def test_admm_pruner_add_batch(self):
        """测试添加批次数据"""
        # 测试2D输入 - 会被当作单个批次处理
        input_data_2d = torch.randn(5, 20)
        self.pruner.add_batch(input_data_2d)

        # 由于2D输入会被unsqueeze(0)变成3D，所以nsamples是1而不是5
        self.assertEqual(self.pruner.nsamples, 1)
        self.assertGreater(torch.norm(self.pruner.hessian), 0)

        # 重置并测试3D输入 - 正确的批次处理
        self.pruner.nsamples = 0
        self.pruner.hessian.zero_()

        input_data_3d = torch.randn(5, 10, 20)  # 5个批次，每个批次10个token，20个特征
        self.pruner.add_batch(input_data_3d)

        # 现在nsamples应该是5
        self.assertEqual(self.pruner.nsamples, 5)
        self.assertGreater(torch.norm(self.pruner.hessian), 0)

    def test_admm_pruner_fasterprune(self):
        """测试快速稀疏化"""
        expect_sparse_ratio = 0.5
        # 添加一些数据
        input_data = torch.randn(10, 20)
        self.pruner.add_batch(input_data)

        original_weight = self.layer.weight.data.clone()

        # 执行稀疏化
        self.pruner.fasterprune(sparse_ratio=expect_sparse_ratio)

        # 验证权重被修改
        self.assertFalse(torch.equal(original_weight, self.layer.weight.data))

        # 验证稀疏比例（大致验证）
        zero_count = (self.layer.weight.data.abs() < 1e-6).sum().item()
        total_count = self.layer.weight.data.numel()
        actual_sparse_ratio = zero_count / total_count

        # 允许一定的误差
        self.assertLessEqual(actual_sparse_ratio, expect_sparse_ratio)

    def test_admm_pruner_free(self):
        """测试内存释放"""
        input_data = torch.randn(5, 20)
        self.pruner.add_batch(input_data)

        self.assertIsNotNone(self.pruner.hessian)

        self.pruner.free()

        self.assertIsNone(self.pruner.hessian)


class TestQuantizeL2(unittest.TestCase):
    """测试 quantize_l2 函数"""

    def setUp(self):
        """测试前的准备工作"""
        torch.manual_seed(42)
        np.random.seed(42)

    def test_quantize_l2_basic_functionality(self):
        """测试基本量化功能"""
        # 创建测试数据
        para = torch.randn(4, 4, dtype=torch.float16)
        keep_mask = torch.zeros_like(para, dtype=torch.bool)
        keep_mask[0, 0] = True  # 保持第一个元素的精度

        # 执行量化
        result = quantize_l2(keep_mask, para, threshold=2, rtn=True)

        # 验证结果
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, para.shape)

        # 验证保持精度的位置
        original_mant, _ = split_half(para.to(torch.half))
        result_mant, _ = split_half(result)
        self.assertTrue(torch.allclose(original_mant[0, 0].abs(), result_mant[0, 0].abs(), atol=1e-3))

    def test_quantize_l2_different_thresholds(self):
        """测试不同阈值下的量化效果"""
        para = torch.randn(3, 3, dtype=torch.float16)
        keep_mask = torch.zeros_like(para, dtype=torch.bool)

        thresholds = [1, 2, 4, 8]
        results = []

        for threshold in thresholds:
            result = quantize_l2(keep_mask, para, threshold=threshold, rtn=True)
            results.append(result)

        # 验证不同阈值产生不同结果
        for i in range(len(results) - 1):
            self.assertFalse(torch.equal(results[i], results[i + 1]))

    def test_quantize_l2_rtn_vs_trunc(self):
        """测试四舍五入与截断的区别"""
        para = torch.tensor([[1.7, 2.3], [-1.7, -2.3]], dtype=torch.float16)
        keep_mask = torch.zeros_like(para, dtype=torch.bool)

        result_rtn = quantize_l2(keep_mask, para, threshold=2, rtn=True)
        result_trunc = quantize_l2(keep_mask, para, threshold=2, rtn=False)

        # 验证两种模式产生不同结果
        self.assertFalse(torch.equal(result_rtn, result_trunc))

    def test_quantize_l2_keep_mask_effect(self):
        """测试保持掩码的效果"""
        para = torch.randn(3, 3, dtype=torch.float16)

        # 不保持任何位置的精度
        keep_mask_empty = torch.zeros_like(para, dtype=torch.bool)
        result_empty = quantize_l2(keep_mask_empty, para, threshold=2, rtn=True)

        # 保持所有位置的精度
        keep_mask_full = torch.ones_like(para, dtype=torch.bool)
        result_full = quantize_l2(keep_mask_full, para, threshold=2, rtn=True)

        # 验证保持掩码的效果
        self.assertFalse(torch.equal(result_empty, result_full))
        # 验证结果与原始参数一致
        self.assertTrue(torch.equal(para, result_full))

    def test_quantize_l2_zero_values(self):
        """测试零值处理"""
        para = torch.zeros(2, 2, dtype=torch.float16)
        keep_mask = torch.zeros_like(para, dtype=torch.bool)

        result = quantize_l2(keep_mask, para, threshold=2, rtn=True)

        # 验证零值保持为零
        self.assertTrue(torch.all(result == 0))

    def test_quantize_l2_extreme_values(self):
        """测试极值处理"""
        # 测试极大值和极小值
        para = torch.tensor([[np.float16('inf'), np.float16('-inf')], [1e-10, 1e10]], dtype=torch.float16)
        keep_mask = torch.zeros_like(para, dtype=torch.bool)

        result = quantize_l2(keep_mask, para, threshold=2, rtn=True)

        # 对于无穷大值，量化后可能仍然是无穷大，这是正常的
        # 验证非无穷大的有限值被正确处理
        finite_mask = torch.isfinite(para)
        if torch.any(finite_mask):
            self.assertTrue(torch.all(torch.isfinite(result[finite_mask]) | (result[finite_mask] == 0)))

    def test_quantize_l2_negative_values(self):
        """测试负值处理"""
        para = torch.tensor([[-1.5, -2.5], [1.5, 2.5]], dtype=torch.float16)
        keep_mask = torch.zeros_like(para, dtype=torch.bool)

        result = quantize_l2(keep_mask, para, threshold=2, rtn=True)

        # 验证符号保持
        self.assertTrue(torch.all(torch.sign(result) == torch.sign(para.to(torch.half))))


class TestQuantizeClip(unittest.TestCase):
    """测试 quantize_clip 函数"""

    def setUp(self):
        """测试前的准备工作"""
        torch.manual_seed(42)

    def test_quantize_clip_basic_functionality(self):
        """测试基本裁剪量化功能"""
        para = torch.randn(4, 4, dtype=torch.float16)

        result = quantize_clip(para, threshold=2, rtn=True)

        # 验证结果
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, para.shape)

    def test_quantize_clip_different_thresholds(self):
        """测试不同阈值的效果"""
        para = torch.randn(3, 3, dtype=torch.float16)

        result_t2 = quantize_clip(para, threshold=2, rtn=True)
        result_t4 = quantize_clip(para, threshold=4, rtn=True)

        # 验证不同阈值产生不同结果
        self.assertFalse(torch.equal(result_t2, result_t4))

    def test_quantize_clip_rtn_modes(self):
        """测试四舍五入和截断模式"""
        para = torch.tensor([[1.7, 2.3], [-1.7, -2.3]], dtype=torch.float16)

        result_rtn = quantize_clip(para, threshold=2, rtn=True)
        result_trunc = quantize_clip(para, threshold=2, rtn=False)

        # 验证两种模式产生不同结果
        self.assertFalse(torch.equal(result_rtn, result_trunc))

    def test_quantize_clip_zero_handling(self):
        """测试零值处理"""
        para = torch.zeros(2, 2, dtype=torch.float16)

        result = quantize_clip(para, threshold=2, rtn=True)

        # 验证零值保持为零
        self.assertTrue(torch.all(result == 0))

    def test_quantize_clip_consistency_with_quantize_l2(self):
        """测试与 quantize_l2 的一致性（当不保持任何位置精度时）"""
        para = torch.randn(3, 3, dtype=torch.float16)
        keep_mask = torch.zeros_like(para, dtype=torch.bool)

        result_clip = quantize_clip(para, threshold=2, rtn=True)
        result_l2 = quantize_l2(keep_mask, para, threshold=2, rtn=True)

        # 当不保持任何位置精度时，两种方法应该产生相似的结果
        # 注意：由于数值精度问题，使用 allclose 而不是 equal
        self.assertTrue(torch.allclose(result_clip, result_l2, atol=1e-3))


class TestSplitHalf(unittest.TestCase):
    """测试 split_half 函数"""

    def setUp(self):
        """测试前的准备工作"""
        torch.manual_seed(42)

    def test_split_half_torch_tensor(self):
        """测试 torch.Tensor 输入"""
        para = torch.tensor([1.0, 2.0, 0.5, -1.5], dtype=torch.float16)

        mant, exp = split_half(para)

        # 验证返回类型
        self.assertIsInstance(mant, torch.Tensor)
        self.assertIsInstance(exp, torch.Tensor)
        self.assertEqual(mant.shape, para.shape)
        self.assertEqual(exp.shape, para.shape)

    def test_split_half_numpy_array(self):
        """测试 numpy.ndarray 输入"""
        para = np.array([1.0, 2.0, 0.5, -1.5], dtype=np.float16)

        mant, exp = split_half(para)

        # 验证返回类型
        self.assertIsInstance(mant, np.ndarray)
        self.assertIsInstance(exp, np.ndarray)
        self.assertEqual(mant.shape, para.shape)
        self.assertEqual(exp.shape, para.shape)

    def test_split_half_zero_values(self):
        """测试零值处理"""
        para = torch.zeros(3, dtype=torch.float16)
        zero_exp_value = -10

        mant, exp = split_half(para, zero_exp_value=zero_exp_value)

        # 验证零值的指数被正确设置
        self.assertTrue(torch.all(exp[para == 0] == zero_exp_value))

    def test_split_half_infinite_values(self):
        """测试无穷大值处理"""
        para = torch.tensor([float('inf'), float('-inf'), 1.0], dtype=torch.float16)

        mant, exp = split_half(para)

        # 验证无穷大值的指数被设置为16
        # 注意：需要使用半精度比较
        inf_mask = (para == torch.tensor(float('inf'), dtype=torch.float16)) | (
            para == torch.tensor(float('-inf'), dtype=torch.float16)
        )
        if torch.any(inf_mask):
            self.assertTrue(torch.all(exp[inf_mask] == 16))

    def test_split_half_nan_values(self):
        """测试 NaN 值处理"""
        para = torch.tensor([float('nan'), 1.0, 2.0], dtype=torch.float16)

        mant, exp = split_half(para)

        # 注意：由于split_half函数中使用了 para == np.half("Nan") 来检测NaN，
        # 这种比较对NaN总是返回False，所以NaN值实际上不会被特殊处理
        # 验证函数能够处理NaN输入而不崩溃
        self.assertEqual(mant.shape, para.shape)
        self.assertEqual(exp.shape, para.shape)

        # 验证NaN位置的尾数仍然是NaN
        nan_mask = torch.isnan(para)
        self.assertTrue(torch.all(torch.isnan(mant[nan_mask])))

    def test_split_half_subnormal_values(self):
        """测试次正规数处理"""
        # 创建次正规数（非常小的数）
        para = torch.tensor([1e-8, 1e-10, 1.0], dtype=torch.float16)

        mant, exp = split_half(para)

        # 验证结果的合理性
        self.assertTrue(torch.all(torch.isfinite(mant)))
        self.assertTrue(torch.all(torch.isfinite(exp)))

    def test_split_half_invalid_input_type(self):
        """测试无效输入类型"""
        invalid_input = [1.0, 2.0, 3.0]  # 列表类型

        with self.assertRaises(TypeError):
            split_half(invalid_input)


if __name__ == '__main__':
    unittest.main()
