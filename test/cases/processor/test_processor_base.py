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
# pylint: disable=expression-not-assigned,no-name-in-module

import unittest
from typing import Tuple, Dict, Optional
from unittest.mock import Mock

import torch
from resources.fake_llama.fake_llama import get_fake_llama_model_and_tokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.generated_runner import GeneratedRunner
from msmodelslim.core.runner.pipeline_parallel_runner import PPRunner
from msmodelslim.processor import AutoProcessorConfig

KEY_INPUT_IDS = "input_ids"
KEY_ATTENTION_MASK = "attention_mask"
STR_TEST_PROMPT = "Hello world"
RETURN_TENSOR_TYPE = "pt"


class TestProcessorBase(unittest.TestCase):
    """处理器测试基类，提供通用的测试方法和工具"""

    @staticmethod
    def init_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """初始化测试模型和分词器"""
        return get_fake_llama_model_and_tokenizer()

    def setUp(self):
        """测试前的准备工作"""
        self.model, self.tokenizer = self.init_model()
        self.original_state_dict = self.model.state_dict().copy()

    def tearDown(self):
        """测试后的清理工作"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def run_processor_with_cfg(self, config: AutoProcessorConfig) -> GeneratedRunner:
        """使用配置运行处理器"""
        # 准备校准数据
        test_prompt = self.tokenizer(STR_TEST_PROMPT, return_tensors=RETURN_TENSOR_TYPE, padding=True, truncation=True)
        dataset_calib = [[test_prompt[KEY_INPUT_IDS], test_prompt[KEY_ATTENTION_MASK]]]

        # 创建mock的PipelineInterface适配器
        mock_adapter = Mock()
        mock_adapter.model = self.model

        # Mock必要的方法
        mock_adapter.init_model.return_value = self.model
        mock_adapter.handle_dataset.return_value = dataset_calib

        # 创建真实的模型层迭代器，用于触发处理器
        def create_model_visit_generator():
            yield ProcessRequest(name="", module=self.model, args=(), kwargs={})

        def create_model_forward_generator():
            # 使用 tokenizer 产生的输入，按整模型签名传参，触发量化层前向
            yield ProcessRequest(
                name="",
                module=self.model,
                args=(),
                kwargs={
                    KEY_INPUT_IDS: test_prompt[KEY_INPUT_IDS],
                    KEY_ATTENTION_MASK: test_prompt[KEY_ATTENTION_MASK],
                },
            )

        mock_adapter.generate_model_visit.return_value = create_model_visit_generator()
        mock_adapter.generate_model_forward.return_value = create_model_forward_generator()
        mock_adapter.enable_kv_cache.return_value = None

        # 创建LayerWiseRunner
        runner = PPRunner(mock_adapter)

        # 添加处理器配置
        runner.add_processor(config)

        # 运行处理器
        runner.run(model=self.model, calib_data=dataset_calib, device=DeviceType.CPU)
        return runner

    def create_test_input(self, text: str = "Hello world", max_length: int = 10) -> Dict[str, torch.Tensor]:
        """创建测试输入"""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
        return inputs

    def run_model_forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """运行模型前向传播"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.logits

    def assert_model_output_shape(self, outputs: torch.Tensor, expected_shape: Tuple[int, ...]):
        """断言模型输出形状"""
        self.assertEqual(outputs.shape, expected_shape)

    def assert_model_output_dtype(self, outputs: torch.Tensor, expected_dtype: torch.dtype):
        """断言模型输出数据类型"""
        self.assertEqual(outputs.dtype, expected_dtype)

    def assert_model_parameters_changed(self, original_state_dict: Dict[str, torch.Tensor]):
        """断言模型参数发生了变化"""
        current_state_dict = self.model.state_dict()
        for key in original_state_dict:
            if key in current_state_dict:
                self.assertFalse(
                    torch.allclose(original_state_dict[key], current_state_dict[key]),
                    f"Parameter {key} should have changed",
                )

    def assert_model_parameters_unchanged(self, original_state_dict: Dict[str, torch.Tensor]):
        """断言模型参数没有发生变化"""
        current_state_dict = self.model.state_dict()
        for key in original_state_dict:
            if key in current_state_dict:
                self.assertTrue(
                    torch.allclose(original_state_dict[key], current_state_dict[key]),
                    f"Parameter {key} should not have changed",
                )

    def assert_linear_layers_quantized(self, model: PreTrainedModel, expected_quantized_layers: list):
        """断言指定的线性层被量化了"""
        for layer_name in expected_quantized_layers:
            layer = self.get_module_by_name(model, layer_name)
            self.assertIsNotNone(layer, f"Layer {layer_name} not found")
            # 检查是否被替换为量化层
            self.assertNotIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should be quantized")

    def assert_linear_layers_not_quantized(self, model: PreTrainedModel, expected_unquantized_layers: list):
        """断言指定的线性层没有被量化"""
        for layer_name in expected_unquantized_layers:
            layer = self.get_module_by_name(model, layer_name)
            self.assertIsNotNone(layer, f"Layer {layer_name} not found")
            # 检查是否仍然是原始线性层
            self.assertIsInstance(layer, torch.nn.Linear, f"Layer {layer_name} should not be quantized")

    def get_module_by_name(self, model: PreTrainedModel, module_name: str) -> Optional[torch.nn.Module]:
        """根据名称获取模块"""
        names = module_name.split('.')
        current_module = model
        for name in names:
            if hasattr(current_module, name):
                current_module = getattr(current_module, name)
            else:
                return None
        return current_module

    def count_linear_layers(self, model: PreTrainedModel) -> int:
        """统计模型中的线性层数量"""
        count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                count += 1
        return count

    def get_linear_layer_names(self, model: PreTrainedModel) -> list:
        """获取模型中所有线性层的名称"""
        linear_names = []
        for name, module in model.named_modules():
            if 'lm_head' in name:
                continue
            if isinstance(module, torch.nn.Linear):
                linear_names.append(name)
        return linear_names

    def assert_model_runs_without_error(self, inputs: Dict[str, torch.Tensor]):
        """断言模型能够正常运行而不出错"""
        try:
            outputs = self.run_model_forward(inputs)
            self.assertIsInstance(outputs, torch.Tensor)
        except Exception as e:
            self.fail(f"Model should run without error, but got: {e}")

    def assert_outputs_close(
        self, outputs1: torch.Tensor, outputs2: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-3
    ):
        """断言两个输出张量接近"""
        self.assertTrue(
            torch.allclose(outputs1, outputs2, rtol=rtol, atol=atol),
            f"Outputs should be close with rtol={rtol}, atol={atol}",
        )

    def assert_outputs_not_close(
        self, outputs1: torch.Tensor, outputs2: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-3
    ):
        """断言两个输出张量不接近"""
        self.assertFalse(
            torch.allclose(outputs1, outputs2, rtol=rtol, atol=atol),
            f"Outputs should not be close with rtol={rtol}, atol={atol}",
        )
