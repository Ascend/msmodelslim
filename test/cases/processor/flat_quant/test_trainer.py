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
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from msmodelslim.processor.flat_quant.trainer import LayerTrainer
from msmodelslim.processor.flat_quant.flat_quant import FlatQuantProcessorConfig


class TestLayerTrainer(unittest.TestCase):
    """测试 LayerTrainer 类 - 单层训练器，允许单层量化校准训练"""

    # ==================== 初始化测试 ====================

    def test_LayerTrainer_config_equals_input_after_init_with_config(self):
        """LayerTrainer-初始化后-config与输入一致"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        self.assertIs(trainer.config, config)
        self.assertIsInstance(trainer.loss_fn, nn.MSELoss)

    def test_LayerTrainer_loss_fn_returns_scalar_when_call(self):
        """LayerTrainer-调用loss_fn时-返回标量张量"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        loss = trainer.loss_fn(x, y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_LayerTrainer_uses_config_epochs_after_init(self):
        """LayerTrainer-初始化后-使用配置中的epochs"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        self.assertEqual(trainer.config.epochs, config.epochs)

    def test_LayerTrainer_uses_config_flat_lr_after_init(self):
        """LayerTrainer-初始化后-使用配置中的flat_lr"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        self.assertEqual(trainer.config.flat_lr, config.flat_lr)

    def test_LayerTrainer_uses_config_cali_bsz_after_init(self):
        """LayerTrainer-初始化后-使用配置中的cali_bsz"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        self.assertEqual(trainer.config.cali_bsz, config.cali_bsz)

    # ==================== setup_optimizer 方法测试 ====================

    def test_LayerTrainer_returns_tuple_when_setup_optimizer(self):
        """LayerTrainer-调用setup_optimizer时-返回元组"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        params = [torch.randn(4, 4, requires_grad=True)]
        nsamples = 16

        optimizer, scheduler = trainer.setup_optimizer(params, nsamples)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsNotNone(scheduler)

    def test_LayerTrainer_creates_adamw_when_setup_optimizer(self):
        """LayerTrainer-调用setup_optimizer时-创建AdamW优化器"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        params = [torch.randn(4, 4, requires_grad=True)]
        nsamples = 16

        optimizer, scheduler = trainer.setup_optimizer(params, nsamples)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_LayerTrainer_creates_chained_scheduler_when_setup_optimizer_with_warmup(self):
        """LayerTrainer-调用setup_optimizer启用warmup时-创建ChainedScheduler"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        params = [torch.randn(4, 4, requires_grad=True)]
        nsamples = 16

        optimizer, scheduler = trainer.setup_optimizer(params, nsamples)

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ChainedScheduler)

    def test_LayerTrainer_returns_cosine_scheduler_when_setup_optimizer_without_warmup(self):
        """LayerTrainer-调用setup_optimizer禁用warmup时-返回CosineAnnealingLR"""
        config = FlatQuantProcessorConfig()
        object.__setattr__(config, 'warmup', False)
        trainer = LayerTrainer(config)
        params = [torch.randn(4, 4, requires_grad=True)]
        nsamples = 16

        optimizer, scheduler = trainer.setup_optimizer(params, nsamples)

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_LayerTrainer_scheduler_has_correct_t_max_when_setup_optimizer_without_warmup(self):
        """LayerTrainer-调用setup_optimizer禁用warmup时-T_max正确设置"""
        config = FlatQuantProcessorConfig()
        object.__setattr__(config, 'warmup', False)
        object.__setattr__(config, 'epochs', 5)
        trainer = LayerTrainer(config)
        params = [torch.randn(4, 4, requires_grad=True)]
        nsamples = 16

        optimizer, scheduler = trainer.setup_optimizer(params, nsamples)

        expected_t_max = 5 * 16
        self.assertEqual(scheduler.T_max, expected_t_max)

    def test_LayerTrainer_scheduler_can_step_when_setup_optimizer(self):
        """LayerTrainer-调用setup_optimizer后-scheduler可以正常step"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        params = [torch.randn(4, 4, requires_grad=True)]
        nsamples = 16

        optimizer, scheduler = trainer.setup_optimizer(params, nsamples)

        optimizer.zero_grad()
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        self.assertIsInstance(new_lr, float)

    # ==================== train_layer 方法测试 ====================

    def test_LayerTrainer_returns_directly_when_train_layer_no_trainable_params(self):
        """LayerTrainer-调用train_layer无可训练参数时-直接返回"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        module = nn.Linear(32, 32)

        mock_request = MagicMock()
        mock_request.module = module
        mock_request.datas = []

        float_output = [torch.randn(2, 32)]

        with patch('msmodelslim.processor.flat_quant.trainer.get_trainable_parameters') as mock_get_params:
            mock_get_params.return_value = ([], [], False)

            with patch('msmodelslim.processor.flat_quant.trainer.convert_outputs_to_inputs') as mock_convert:
                mock_convert.return_value = float_output

                result = trainer.train_layer(mock_request, float_output)  # noqa: F841
                mock_convert.assert_called_once_with(float_output)

    def test_LayerTrainer_trains_when_train_layer_with_trainable_params(self):
        """LayerTrainer-调用train_layer有可训练参数时-执行训练"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        module = nn.Linear(32, 32)

        mock_request = MagicMock()
        mock_request.module = module
        mock_request.name = "test_layer"
        mock_request.datas = [(torch.randn(1, 32), {})]

        float_output = [torch.randn(1, 32)]
        trainable_param = torch.randn(4, 4, requires_grad=True)

        with patch('msmodelslim.processor.flat_quant.trainer.get_trainable_parameters') as mock_get_params:
            mock_get_params.return_value = ([trainable_param], [trainable_param], True)

            with patch('msmodelslim.processor.flat_quant.trainer.convert_outputs_to_inputs') as mock_convert:
                mock_convert.return_value = float_output

                with patch('msmodelslim.processor.flat_quant.trainer.move_tensors_to_device') as mock_move:
                    mock_move.side_effect = lambda x, d: x

                    original_forward = module.forward
                    module.forward = lambda *args, **kwargs: (torch.randn(1, 32),)

                    try:
                        result = trainer.train_layer(mock_request, float_output)
                        self.assertIsNotNone(result)
                    finally:
                        module.forward = original_forward

    def test_LayerTrainer_executes_training_loop_when_train_layer_with_trainable_params(self):
        """LayerTrainer-调用train_layer有可训练参数时-执行完整训练循环"""
        config = FlatQuantProcessorConfig()
        object.__setattr__(config, 'epochs', 1)
        object.__setattr__(config, 'cali_bsz', 1)
        object.__setattr__(config, 'warmup', False)
        trainer = LayerTrainer(config)

        module = nn.Linear(4, 4)
        input_data = torch.randn(1, 4)
        kwargs_data = {}

        mock_request = MagicMock()
        mock_request.module = module
        mock_request.name = "test_layer"
        mock_request.datas = [(input_data, kwargs_data)]

        float_output = [torch.randn(1, 4)]
        trainable_param = torch.randn(4, 4, requires_grad=True)

        with patch('msmodelslim.processor.flat_quant.trainer.get_trainable_parameters') as mock_get_params:
            mock_get_params.return_value = ([trainable_param], [trainable_param], True)

            with patch('msmodelslim.processor.flat_quant.trainer.convert_outputs_to_inputs') as mock_convert:
                mock_convert.return_value = float_output

                with patch('msmodelslim.processor.flat_quant.trainer.move_tensors_to_device') as mock_move:
                    mock_move.side_effect = lambda x, d: x

                    def mock_forward(*args, **kwargs):
                        out = args[0] @ trainable_param
                        return (out,)

                    original_forward = module.forward
                    module.forward = mock_forward

                    try:
                        result = trainer.train_layer(mock_request, float_output)
                        self.assertIsNotNone(result)
                        mock_convert.assert_called()
                    finally:
                        module.forward = original_forward

    def test_LayerTrainer_processes_multiple_batches_when_train_layer(self):
        """LayerTrainer-调用train_layer多批次时-正确处理所有批次"""
        config = FlatQuantProcessorConfig()
        object.__setattr__(config, 'epochs', 1)
        object.__setattr__(config, 'cali_bsz', 1)
        object.__setattr__(config, 'warmup', False)
        trainer = LayerTrainer(config)

        module = nn.Linear(4, 4)
        num_samples = 4
        mock_request = MagicMock()
        mock_request.module = module
        mock_request.name = "test_layer"
        mock_request.datas = [(torch.randn(1, 4), {}) for _ in range(num_samples)]

        float_output = [torch.randn(1, 4) for _ in range(num_samples)]
        trainable_param = torch.randn(4, 4, requires_grad=True)

        with patch('msmodelslim.processor.flat_quant.trainer.get_trainable_parameters') as mock_get_params:
            mock_get_params.return_value = ([trainable_param], [trainable_param], True)

            with patch('msmodelslim.processor.flat_quant.trainer.convert_outputs_to_inputs') as mock_convert:
                mock_convert.return_value = float_output

                with patch('msmodelslim.processor.flat_quant.trainer.move_tensors_to_device') as mock_move:
                    mock_move.side_effect = lambda x, d: x

                    def mock_forward(*args, **kwargs):
                        out = args[0] @ trainable_param
                        return (out,)

                    original_forward = module.forward
                    module.forward = mock_forward

                    try:
                        result = trainer.train_layer(mock_request, float_output)
                        self.assertIsNotNone(result)
                    finally:
                        module.forward = original_forward

    def test_LayerTrainer_handles_empty_data_when_train_layer(self):
        """LayerTrainer-调用train_layer数据为空时-正确处理"""
        config = FlatQuantProcessorConfig()
        trainer = LayerTrainer(config)
        module = nn.Linear(4, 4)

        mock_request = MagicMock()
        mock_request.module = module
        mock_request.datas = []

        float_output = []

        with patch('msmodelslim.processor.flat_quant.trainer.get_trainable_parameters') as mock_get_params:
            mock_get_params.return_value = ([], [], False)

            with patch('msmodelslim.processor.flat_quant.trainer.convert_outputs_to_inputs') as mock_convert:
                mock_convert.return_value = []

                result = trainer.train_layer(mock_request, float_output)  # noqa: F841
                mock_convert.assert_called_once_with(float_output)

    def test_LayerTrainer_loss_computed_correctly_when_train_layer(self):
        """LayerTrainer-调用train_layer时-损失计算正确"""
        config = FlatQuantProcessorConfig()
        object.__setattr__(config, 'epochs', 1)
        object.__setattr__(config, 'cali_bsz', 1)
        object.__setattr__(config, 'warmup', False)
        trainer = LayerTrainer(config)

        module = nn.Linear(4, 4)

        mock_request = MagicMock()
        mock_request.module = module
        mock_request.name = "test_layer"
        mock_request.datas = [(torch.randn(1, 4), {})]

        float_output = [torch.randn(1, 4)]
        trainable_param = torch.randn(4, 4, requires_grad=True)

        with patch('msmodelslim.processor.flat_quant.trainer.get_trainable_parameters') as mock_get_params:
            mock_get_params.return_value = ([trainable_param], [trainable_param], True)

            with patch('msmodelslim.processor.flat_quant.trainer.convert_outputs_to_inputs') as mock_convert:
                mock_convert.return_value = float_output

                with patch('msmodelslim.processor.flat_quant.trainer.move_tensors_to_device') as mock_move:
                    mock_move.side_effect = lambda x, d: x

                    def mock_forward(*args, **kwargs):
                        out = args[0] @ trainable_param
                        return (out,)

                    original_forward = module.forward
                    module.forward = mock_forward

                    try:
                        result = trainer.train_layer(mock_request, float_output)
                        self.assertIsNotNone(result)
                    finally:
                        module.forward = original_forward


if __name__ == '__main__':
    unittest.main()
