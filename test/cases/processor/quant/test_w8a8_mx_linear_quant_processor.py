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
# pylint: disable=attribute-defined-outside-init

import unittest

import torch

from ..test_processor_base import TestProcessorBase
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.processor.quant.linear import LinearProcessorConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig


class TestLinearQuantProcessor(TestProcessorBase):
    """测试LinearQuantProcessor的功能"""

    def create_basic_qconfig(self, w_bits: int = 8, a_bits: int = 8) -> LinearQConfig:
        """创建基本的量化配置"""
        weight_config = QConfig(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True, method="minmax")
        act_config = QConfig(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True, method="minmax")
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

        runner = self.run_processor_with_cfg(config)  # noqa: F841

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_quantize_all_linear_layers(self):
        self.linear_layer_names = self.get_linear_layer_names(self.model)
        config = self.create_processor_config(include=["*"])

        self.run_processor_with_cfg(config)

        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsNotNone(layer, f"Layer {layer_name} not found")
            self.assertNotIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should be quantized")

    def test_quantize_specific_layers(self):
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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

    def test_per_block_quantization(self):
        qconfig_per_block = LinearQConfig(
            act=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True, method="minmax"),
            weight=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True, method="minmax"),
        )
        config = LinearProcessorConfig(qconfig=qconfig_per_block, include=["*"])

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
        self.linear_layer_names = self.get_linear_layer_names(self.model)
        config = self.create_processor_config(include=[])

        self.run_processor_with_cfg(config)

        for layer_name in self.linear_layer_names:
            layer = self.get_module_by_name(self.model, layer_name)
            self.assertIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should not be quantized")

    def test_invalid_layer_patterns(self):
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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
        original_modules = list(self.model.named_modules())  # noqa: F841

        config = self.create_processor_config(include=["*"])
        self.run_processor_with_cfg(config)

        current_modules = list(self.model.named_modules())

        self.assertGreater(len(current_modules), 0)

        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_warning_unmatched_exclude_patterns(self):
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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
        self.linear_layer_names = self.get_linear_layer_names(self.model)
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


if __name__ == '__main__':
    unittest.main()
