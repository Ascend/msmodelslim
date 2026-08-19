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

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.quant.linear import LinearProcessorConfig
from ..test_processor_base import TestProcessorBase


class TestLinearQuantProcessor(TestProcessorBase):
    """测试LinearQuantProcessor的功能"""

    def setUp(self):
        """测试前的准备工作"""
        super().setUp()
        self.linear_layer_names = self.get_linear_layer_names(self.model)
        self.assertGreater(len(self.linear_layer_names), 0, "Model should have at least one linear layer")

    def create_basic_qconfig(self, w_bits: int = 8, a_bits: int = 8) -> LinearQConfig:
        """创建基本的量化配置"""
        weight_config = QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax")
        act_config = QConfig(dtype=QDType.INT8, scope=QScope.PER_TENSOR, symmetric=True, method="minmax")
        return LinearQConfig(act=act_config, weight=weight_config)

    def create_processor_config(self, include: list = None, exclude: list = None) -> LinearProcessorConfig:
        """创建处理器配置"""
        if include is None:
            include = ["*"]
        if exclude is None:
            exclude = []
        qconfig = self.create_basic_qconfig()
        return LinearProcessorConfig(
            qconfig=qconfig,
            include=include,
            exclude=exclude,
        )

    def test_basic_quantization(self):
        config = self.create_processor_config()

        self.run_processor_with_cfg(config)

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_quantize_all_linear_layers(self):
        config = self.create_processor_config(include=["*"])

        self.run_processor_with_cfg(config)

        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsNotNone(layer, f"Layer {layer_name} not found")
            self.assertNotIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should be quantized")

    def test_quantize_specific_layers(self):
        if len(self.linear_layer_names) < 2:
            self.skipTest("Model needs at least 2 linear layers for this test")

        target_layer = self.linear_layer_names[0]
        config = self.create_processor_config(include=[target_layer])

        self.run_processor_with_cfg(config)

        target_module = self.get_module_by_name(self.model, target_layer)
        self.assertIsNotNone(target_module)
        self.assertNotIsInstance(target_module, torch.nn.Linear)

        for layer_name in self.linear_layer_names[1:]:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should not be quantized")

    def test_exclude_specific_layers(self):
        if len(self.linear_layer_names) < 2:
            self.skipTest("Model needs at least 2 linear layers for this test")

        excluded_layer = self.linear_layer_names[0]
        config = self.create_processor_config(exclude=[excluded_layer])

        self.run_processor_with_cfg(config)

        excluded_module = self.get_module_by_name(self.model, excluded_layer)
        self.assertIsInstance(excluded_module, torch.nn.Linear, f"Layer {excluded_layer} should not be quantized")

        for layer_name in self.linear_layer_names[1:]:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertNotIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should be quantized")

    def test_include_and_exclude_patterns(self):
        if len(self.linear_layer_names) < 3:
            self.skipTest("Model needs at least 3 linear layers for this test")

        excluded_layer = self.linear_layer_names[0]
        config = self.create_processor_config(include=["*"], exclude=[excluded_layer])

        self.run_processor_with_cfg(config)

        excluded_module = self.get_module_by_name(self.model, excluded_layer)
        self.assertIsInstance(excluded_module, torch.nn.Linear)

        for layer_name in self.linear_layer_names[1:]:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertNotIsInstance(layer, torch.nn.Linear)

    def test_per_channel_quantization(self):
        qconfig_per_channel = LinearQConfig(
            act=QConfig(dtype=QDType.INT8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax"),
            weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = LinearProcessorConfig(qconfig=qconfig_per_channel, include=["*"])

        self.run_processor_with_cfg(config)

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_asymmetric_quantization(self):
        qconfig_asymmetric = LinearQConfig(
            act=QConfig(dtype=QDType.INT8, scope=QScope.PER_TENSOR, symmetric=False, method="minmax"),
            weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = LinearProcessorConfig(qconfig=qconfig_asymmetric, include=["*"])

        self.run_processor_with_cfg(config)

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_output_consistency(self):
        inputs = self.create_test_input()
        original_outputs = self.run_model_forward(inputs)

        config = self.create_processor_config(include=["*"])
        self.run_processor_with_cfg(config)

        quantized_outputs = self.run_model_forward(inputs)

        self.assertEqual(original_outputs.shape, quantized_outputs.shape)

        self.assertEqual(original_outputs.dtype, quantized_outputs.dtype)

    def test_deploy_functionality(self):
        config = self.create_processor_config(include=["*"])
        self.run_processor_with_cfg(config)

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

        has_fake_quant = False
        for module in self.model.modules():
            if isinstance(module, AutoFakeQuantLinear):
                has_fake_quant = True
                break

        self.assertTrue(has_fake_quant, "Model should contain fake quantization layers")

    def test_empty_include_list(self):
        config = self.create_processor_config(include=[])

        self.run_processor_with_cfg(config)

        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should not be quantized")

    def test_invalid_layer_patterns(self):
        config = self.create_processor_config(include=["nonexistent_layer"])

        self.run_processor_with_cfg(config)

        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should not be quantized")

    def test_processor_properties(self):
        processor = self.model.processor if hasattr(self.model, 'processor') else None

        if processor is not None:
            self.assertFalse(processor.is_data_free())

            self.assertTrue(processor.support_distributed())

    def test_multiple_processor_runs(self):
        config = self.create_processor_config(include=["*"])

        self.run_processor_with_cfg(config)

        self.run_processor_with_cfg(config)

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_memory_usage(self):
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()

        config = self.create_processor_config(include=["*"])
        self.run_processor_with_cfg(config)

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            self.assertLess(final_memory, initial_memory * 2, "Memory usage should not double")

    def test_quantization_with_different_input_sizes(self):
        config = self.create_processor_config(include=["*"])
        self.run_processor_with_cfg(config)

        test_inputs = [
            self.create_test_input("Short", 5),
            self.create_test_input("Medium length text", 15),
            self.create_test_input("This is a longer text for testing", 20),
        ]

        for inputs in test_inputs:
            self.assert_model_runs_without_error(inputs)

    def test_quantization_preserves_model_structure(self):
        config = self.create_processor_config(include=["*"])
        self.run_processor_with_cfg(config)

        current_modules = list(self.model.named_modules())

        self.assertGreater(len(current_modules), 0)

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_warning_unmatched_include_patterns(self):
        config = self.create_processor_config(include=["nonexistent_layer", "another_fake_layer"])

        with self.assertLogs('msmodelslim.processor.linear_quant', level='WARNING') as log_context:
            self.run_processor_with_cfg(config)

        log_messages = log_context.output
        self.assertTrue(any("include patterns are not matched" in msg for msg in log_messages))
        self.assertTrue(any("nonexistent_layer" in msg for msg in log_messages))
        self.assertTrue(any("another_fake_layer" in msg for msg in log_messages))

        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should not be quantized")

    def test_warning_unmatched_exclude_patterns(self):
        config = self.create_processor_config(include=["*"], exclude=["nonexistent_layer", "another_fake_layer"])

        with self.assertLogs('msmodelslim.processor.linear_quant', level='WARNING') as log_context:
            self.run_processor_with_cfg(config)

        log_messages = log_context.output
        self.assertTrue(any("exclude patterns are not matched" in msg for msg in log_messages))
        self.assertTrue(any("nonexistent_layer" in msg for msg in log_messages))
        self.assertTrue(any("another_fake_layer" in msg for msg in log_messages))

        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertNotIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should be quantized")

    def test_warning_mixed_matched_unmatched_patterns(self):
        if len(self.linear_layer_names) < 1:
            self.skipTest("Model needs at least 1 linear layer for this test")

        existing_layer = self.linear_layer_names[0]
        config = self.create_processor_config(include=[existing_layer, "nonexistent_layer"])

        with self.assertLogs('msmodelslim.processor.linear_quant', level='WARNING') as log_context:
            self.run_processor_with_cfg(config)

        log_messages = log_context.output
        self.assertTrue(any("include patterns are not matched" in msg for msg in log_messages))
        self.assertTrue(any("nonexistent_layer" in msg for msg in log_messages))
        self.assertFalse(any(existing_layer in msg for msg in log_messages))

        existing_module = self.get_module_by_name(self.model, existing_layer)
        self.assertNotIsInstance(existing_module, torch.nn.Linear, f"Layer {existing_layer} should be quantized")

    def test_warning_with_wildcard_patterns(self):
        config = self.create_processor_config(include=["*"], exclude=["*.nonexistent"])

        with self.assertLogs('msmodelslim.processor.linear_quant', level='WARNING') as log_context:
            self.run_processor_with_cfg(config)

        log_messages = log_context.output
        self.assertEqual(len(log_messages), 1)
        self.assertTrue(any("exclude patterns are not matched" in msg for msg in log_messages))
        self.assertTrue(any("*.nonexistent" in msg for msg in log_messages))

        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertNotIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should be quantized")

    def test_warning_multiple_include_exclude_patterns(self):
        if len(self.linear_layer_names) < 2:
            self.skipTest("Model needs at least 2 linear layers for this test")

        existing_layer = self.linear_layer_names[0]
        config = self.create_processor_config(
            include=[existing_layer, "nonexistent_include_1", "nonexistent_include_2"],
            exclude=["nonexistent_exclude_1", "nonexistent_exclude_2"],
        )

        with self.assertLogs('msmodelslim.processor.linear_quant', level='WARNING') as log_context:
            self.run_processor_with_cfg(config)

        log_messages = log_context.output

        include_warnings = [msg for msg in log_messages if "include patterns are not matched" in msg]
        self.assertEqual(len(include_warnings), 1)
        include_warning = include_warnings[0]
        self.assertIn("nonexistent_include_1", include_warning)
        self.assertIn("nonexistent_include_2", include_warning)
        self.assertNotIn(existing_layer, include_warning)

        exclude_warnings = [msg for msg in log_messages if "exclude patterns are not matched" in msg]
        self.assertEqual(len(exclude_warnings), 1)
        exclude_warning = exclude_warnings[0]
        self.assertIn("nonexistent_exclude_1", exclude_warning)
        self.assertIn("nonexistent_exclude_2", exclude_warning)

        existing_module = self.get_module_by_name(self.model, existing_layer)
        self.assertNotIsInstance(existing_module, torch.nn.Linear, f"Layer {existing_layer} should be quantized")

    def test_no_warning_with_valid_patterns(self):
        if len(self.linear_layer_names) < 1:
            self.skipTest("Model needs at least 1 linear layer for this test")

        existing_layer = self.linear_layer_names[0]
        config = self.create_processor_config(include=[existing_layer])

        self.run_processor_with_cfg(config)

        existing_module = self.get_module_by_name(self.model, existing_layer)
        self.assertNotIsInstance(existing_module, torch.nn.Linear, f"Layer {existing_layer} should be quantized")

        for layer_name in self.linear_layer_names[1:]:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should not be quantized")

    def test_DTS_calibrate_methods_should_exist(self):
        """验证 DTS calibrate 方法存在"""
        from msmodelslim.processor.quant.linear import LinearQuantProcessor

        assert hasattr(LinearQuantProcessor, '_calibrate_shared_data_free_with_dts'), (
            "LinearQuantProcessor should have _calibrate_shared_data_free_with_dts"
        )
        assert hasattr(LinearQuantProcessor, '_dts_calibrate_forward'), (
            "LinearQuantProcessor should have _dts_calibrate_forward"
        )

    def test_preprocess_no_dts_when_not_distributed(self):
        """非分布式下 preprocess 不执行 DTS（不报错）"""
        config = self.create_processor_config(include=["*"])
        self.run_processor_with_cfg(config)
        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_calibrate_forward_calls_weight_quantizer_forward(self):
        """验证 _dts_calibrate_forward 触发 weight_quantizer.forward"""
        from msmodelslim.processor.quant.linear import LinearQuantProcessor

        # 验证方法存在于 LinearQuantProcessor 类上
        assert hasattr(LinearQuantProcessor, '_dts_calibrate_forward'), (
            "LinearQuantProcessor should have _dts_calibrate_forward"
        )

    def test_brace_pattern_include_quantizes_only_matching_layers(self):
        """花括号 include 模式（原 wcmatch BRACE 语义）只量化命中的层。"""
        config = self.create_processor_config(include=["model.layers.{0,1}.mlp.up_proj"])
        self.run_processor_with_cfg(config)

        for layer_idx, expect_quantized in ((0, True), (1, True), (2, False)):
            name = f"model.layers.{layer_idx}.mlp.up_proj"
            layer = self.get_module_by_name(self.model, name)
            if expect_quantized:
                self.assertNotIsInstance(layer, torch.nn.Linear, f"{name} should be quantized")
            else:
                self.assertIsInstance(layer, torch.nn.Linear, f"{name} should not be quantized")

    def test_dts_calibrate_shared_data_free_when_distributed(self):
        """分布式下共享 data-free 权重量化经 DTS 调度完成校准（替换 pygtrie 后的业务级验证）。"""
        from unittest.mock import MagicMock, patch
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer
        from msmodelslim.core.quantizer.linear import LinearQuantizer
        from msmodelslim.processor.quant import linear as linear_module

        config = self.create_processor_config(include=["*"])
        processor = linear_module.LinearQuantProcessor(model=self.model, config=config)
        processor._install_quantizer("", self.model)

        quantizer_names = [
            name for name, submodule in self.model.named_modules() if isinstance(submodule, LinearQuantizer)
        ]
        self.assertGreater(len(quantizer_names), 0)

        processor.dist_helper = MagicMock()
        processor.dist_helper.is_shared.return_value = True

        scheduled: list = []
        original = processor._dts_calibrate_forward

        def _spy(module_name: str):
            scheduled.append(module_name)
            return original(module_name)

        # 用真实函数遮蔽实例方法（MagicMock 无法通过 DTS 的语义哈希校验）
        processor._dts_calibrate_forward = _spy
        with (
            patch.object(linear_module.dist, "get_rank", return_value=0),
            patch.object(linear_module.dist, "get_world_size", return_value=2),
            patch.object(AutoWeightQuantizer, "is_data_free", return_value=True),
        ):
            processor._calibrate_shared_data_free_with_dts("", self.model)

        # 所有共享 data-free 权重量化器都应被 DTS 提交并实际执行校准
        self.assertEqual(sorted(scheduled), sorted(quantizer_names))

        # 校准后每个权重量化器都应已产出量化参数（而非依赖 get_q_param 的惰性触发）
        for name in quantizer_names:
            wq = self.get_module_by_name(self.model, name + ".weight_quantizer")
            self.assertIsNotNone(wq.w_q_param, f"{name} weight quantizer should be calibrated")


if __name__ == '__main__':
    unittest.main()
