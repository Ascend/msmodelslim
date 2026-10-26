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

import unittest
from unittest.mock import patch

import torch
from torch import nn

from ._linear_quant_processor_contract import LinearQuantProcessorContract
from ..processor_test_helpers import TestProcessorBase
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig, LinearQuantizer
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.quant.linear import LinearProcessorConfig, LinearQuantProcessor


class TestLinearQuantProcessor(LinearQuantProcessorContract, TestProcessorBase):
    """Test the LinearQuantProcessor contract for INT8."""

    weight_dtype = QDType.INT8
    activation_dtype = QDType.INT8
    weight_scope = QScope.PER_CHANNEL
    activation_scope = QScope.PER_TENSOR

    def test_per_channel_quantization(self):
        qconfig_per_channel = LinearQConfig(
            act=QConfig(dtype=QDType.INT8, scope=QScope.PER_TOKEN, symmetric=True, method='minmax'),
            weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method='minmax'),
        )
        config = LinearProcessorConfig(qconfig=qconfig_per_channel, include=['*'])
        self.run_processor_with_cfg(config)
        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_asymmetric_quantization(self):
        qconfig_asymmetric = LinearQConfig(
            act=QConfig(dtype=QDType.INT8, scope=QScope.PER_TENSOR, symmetric=False, method='minmax'),
            weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method='minmax'),
        )
        config = LinearProcessorConfig(qconfig=qconfig_asymmetric, include=['*'])
        self.run_processor_with_cfg(config)
        inputs = self.create_test_input()
        self.assert_model_runs_without_error(inputs)

    def test_preprocess_returns_without_dts_when_distributed_is_not_initialized(self):
        """公开 preprocess 在非分布式环境下只安装量化器，不创建 DTS。"""
        config = self.create_processor_config(include=['*'])
        processor = LinearQuantProcessor(self.model, config)
        request = BatchProcessRequest(name='', module=self.model, datas=[])
        expected_quantized = sum(isinstance(module, nn.Linear) for module in self.model.modules())
        with patch('msmodelslim.processor.quant.linear.DistributedTaskScheduler') as scheduler:
            processor.preprocess(request)
        scheduler.assert_not_called()
        self.assertEqual(
            sum(isinstance(module, LinearQuantizer) for module in self.model.modules()),
            expected_quantized,
        )

    def test_preprocess_calibrates_every_shared_data_free_linear_when_distributed(self):
        """公开 preprocess 在分布式共享路径下对每个 data-free LinearQuantizer 执行 DTS 校准。"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer
        from msmodelslim.processor.quant import linear as linear_module

        config = self.create_processor_config(include=['*'])
        processor = LinearQuantProcessor(self.model, config)
        request = BatchProcessRequest(name='', module=self.model, datas=[])
        submitted = []

        class _RecordingScheduler:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def submit(self, fn, args=(), dependencies=None, parallel=True):
                submitted.append((fn, args, dependencies, parallel))

            def run(self):
                for fn, args, _, _ in submitted:
                    fn(*args)

        with (
            patch.object(linear_module.dist, 'is_initialized', return_value=True),
            patch.object(linear_module, 'DistHelper') as dist_helper_cls,
            patch.object(linear_module, 'DistributedTaskScheduler', side_effect=_RecordingScheduler),
            patch.object(linear_module.dist, 'get_rank', return_value=0),
            patch.object(linear_module.dist, 'get_world_size', return_value=1),
            patch.object(AutoWeightQuantizer, 'is_data_free', return_value=True),
        ):
            dist_helper_cls.return_value.is_shared.return_value = True
            processor.preprocess(request)

        quantizer_names = [name for name, module in self.model.named_modules() if isinstance(module, LinearQuantizer)]
        self.assertGreater(len(quantizer_names), 0)
        self.assertEqual(len(submitted), len(quantizer_names))
        self.assertEqual(sorted(args[0] for _, args, _, _ in submitted), sorted(quantizer_names))
        self.assertTrue(all(parallel for _, _, _, parallel in submitted))
        for name in quantizer_names:
            quantizer = self.get_module_by_name(self.model, name + '.weight_quantizer')
            self.assertIsNotNone(quantizer.w_q_param, f'{name} weight quantizer should be calibrated')

    def test_brace_pattern_include_quantizes_only_matching_layers(self):
        """花括号 include 模式（原 wcmatch BRACE 语义）只量化命中的层。"""
        config = self.create_processor_config(include=['model.layers.{0,1}.mlp.up_proj'])
        self.run_processor_with_cfg(config)
        for layer_idx, expect_quantized in ((0, True), (1, True), (2, False)):
            name = f'model.layers.{layer_idx}.mlp.up_proj'
            layer = self.get_module_by_name(self.model, name)
            if expect_quantized:
                self.assertNotIsInstance(layer, torch.nn.Linear, f'{name} should be quantized')
            else:
                self.assertIsInstance(layer, torch.nn.Linear, f'{name} should not be quantized')


if __name__ == '__main__':
    unittest.main()
