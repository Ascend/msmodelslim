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

from msmodelslim.processor.quant.autoround_utils.trainer import BlockQuantTrainer
from msmodelslim.processor.quant.autoround_utils.sign_sgd import SignSGD


class TestBlockQuantTrainerInit(unittest.TestCase):
    """测试 BlockQuantTrainer 初始化"""

    def test_should_initialize_with_default_values(self):
        """测试默认值初始化"""
        trainer = BlockQuantTrainer()

        self.assertEqual(trainer.batch_size, 1)
        self.assertEqual(trainer.iters, 2)
        self.assertFalse(trainer.enable_minmax_tuning)
        self.assertTrue(trainer.enable_quanted_input)
        self.assertTrue(trainer.enable_ema_loss)

    def test_should_initialize_with_custom_values(self):
        """测试自定义值初始化"""
        trainer = BlockQuantTrainer(
            batch_size=2,
            iters=10,
            enable_minmax_tuning=True,
            enable_quanted_input=False,
            enable_ema_loss=False,
            lr=0.01,
            minmax_lr=0.001,
            sampler='sequential',
            seqlen=1024,
            gradient_accumulate_steps=2,
            batch_dim=1,
            amp=True,
            amp_dtype=torch.bfloat16,
            not_use_best_mse=True,
            dynamic_max_gap=5,
            infer_bs_coeff=2,
            shared_cache_keys=("key1", "key2"),
        )

        self.assertEqual(trainer.batch_size, 2)
        self.assertEqual(trainer.iters, 10)
        self.assertTrue(trainer.enable_minmax_tuning)
        self.assertFalse(trainer.enable_quanted_input)
        self.assertFalse(trainer.enable_ema_loss)
        self.assertEqual(trainer.lr, 0.01)
        self.assertEqual(trainer.minmax_lr, 0.001)
        self.assertEqual(trainer.sampler, 'sequential')
        self.assertEqual(trainer.seqlen, 1024)
        self.assertEqual(trainer.gradient_accumulate_steps, 2)
        self.assertEqual(trainer.batch_dim, 1)
        self.assertTrue(trainer.amp)
        self.assertEqual(trainer.amp_dtype, torch.bfloat16)
        self.assertTrue(trainer.not_use_best_mse)
        self.assertEqual(trainer.dynamic_max_gap, 5)
        self.assertEqual(trainer.infer_bs_coeff, 2)
        self.assertEqual(trainer.shared_cache_keys, ("key1", "key2"))

    def test_should_set_lr_based_on_iters(self):
        """测试 lr 基于 iters 设置"""
        trainer = BlockQuantTrainer(iters=100)
        self.assertEqual(trainer.lr, 1.0 / 100)

    def test_should_use_default_lr_when_iters_is_zero(self):
        """测试 iters=0 时使用默认 lr"""
        trainer = BlockQuantTrainer(iters=0)
        self.assertEqual(trainer.lr, 5e-3)

    def test_should_use_default_iters_when_negative(self):
        """测试负 iters 使用默认值"""
        trainer = BlockQuantTrainer(iters=-1)
        # 负 iters 时，lr 应该基于 200 计算
        self.assertEqual(trainer.lr, 1.0 / 200)


class TestBlockQuantTrainerStaticMethods(unittest.TestCase):
    """测试 BlockQuantTrainer 静态方法"""

    def test_get_optimizer_should_return_signsgd(self):
        """测试 get_optimizer 返回 SignSGD"""
        optimizer = BlockQuantTrainer.get_optimizer()
        self.assertEqual(optimizer, SignSGD)

    def test_scale_loss_and_backward_should_scale_loss(self):
        """测试 scale_loss_and_backward 缩放损失"""
        loss = torch.tensor(1.0, requires_grad=True)
        scaled_loss = BlockQuantTrainer.scale_loss_and_backward(loss)

        self.assertEqual(scaled_loss.item(), 1000.0)
        self.assertIsNotNone(loss.grad)

    def test_step_should_call_optimizer_step(self):
        """测试 step 调用优化器 step"""
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1)
        lr_schedule = MagicMock()

        # 计算梯度
        loss = param.sum()
        loss.backward()

        BlockQuantTrainer.step(optimizer, lr_schedule)

        lr_schedule.step.assert_called_once()

    def test_collect_best_params_should_collect_from_wrapper(self):
        """测试 collect_best_params 从 wrapper 收集参数"""
        # 创建模拟的 wrapper
        wrapper = MagicMock()
        wrapper.orig_layer = MagicMock()
        wrapper.params = {
            "value": MagicMock(data=torch.tensor([1.0])),
            "min_scale": MagicMock(data=torch.tensor([2.0])),
        }

        block = MagicMock()
        block.named_modules.return_value = [("layer1", wrapper)]

        params = BlockQuantTrainer.collect_best_params(block)

        self.assertIn("layer1", params)
        self.assertIn("value", params["layer1"])
        self.assertIn("min_scale", params["layer1"])

    def test_collect_best_params_should_return_empty_for_no_wrapper(self):
        """测试没有 wrapper 时返回空"""
        block = MagicMock()
        block.named_modules.return_value = [("layer1", nn.Linear(4, 2))]

        params = BlockQuantTrainer.collect_best_params(block)

        self.assertEqual(params, {})


class TestBlockQuantTrainerSamplingInputs(unittest.TestCase):
    """测试 BlockQuantTrainer sampling_inputs 方法"""

    def test_should_sample_inputs(self):
        """测试采样输入"""
        trainer = BlockQuantTrainer()

        input_ids = [torch.randn(1, 4, 8) for _ in range(5)]
        input_others = {
            "positional_inputs": [torch.randn(1, 4) for _ in range(5)],
            "attention_mask": [torch.randn(1, 4) for _ in range(5)],
        }
        indices = [0, 2, 4]

        current_input_ids, current_input_others = trainer.sampling_inputs(
            input_ids, input_others, indices, batch_dim=0, share_cache_keys=()
        )

        self.assertIsInstance(current_input_ids, torch.Tensor)
        self.assertIn("positional_inputs", current_input_others)

    def test_should_handle_single_index(self):
        """测试单个索引"""
        trainer = BlockQuantTrainer()

        input_ids = [torch.randn(1, 4, 8) for _ in range(5)]
        input_others = {
            "positional_inputs": [torch.randn(1, 4) for _ in range(5)],
            "attention_mask": [torch.randn(1, 4) for _ in range(5)],
        }
        indices = [2]

        current_input_ids, current_input_others = trainer.sampling_inputs(
            input_ids, input_others, indices, batch_dim=0, share_cache_keys=()
        )

        self.assertEqual(len(current_input_ids), 1)

    def test_should_share_cache_keys(self):
        """测试共享缓存键"""
        trainer = BlockQuantTrainer()

        input_ids = [torch.randn(1, 4, 8) for _ in range(5)]
        input_others = {
            "positional_inputs": [torch.randn(1, 4) for _ in range(5)],
            "cache_position": torch.randn(4),
        }
        indices = [0, 1]

        current_input_ids, current_input_others = trainer.sampling_inputs(
            input_ids, input_others, indices, batch_dim=0, share_cache_keys=("cache_position",)
        )

        # 共享键应该直接传递
        self.assertIn("cache_position", current_input_others)


class TestBlockQuantTrainerToDevice(unittest.TestCase):
    """测试 BlockQuantTrainer to_device 方法"""

    def test_should_move_tensor_to_device(self):
        """测试移动张量到设备"""
        trainer = BlockQuantTrainer()

        inputs = torch.randn(2, 4)
        result = trainer.to_device(inputs, "cpu")

        self.assertEqual(result.device, torch.device("cpu"))

    def test_should_move_list_to_device(self):
        """测试移动列表到设备"""
        trainer = BlockQuantTrainer()

        inputs = [torch.randn(2, 4), torch.randn(2, 4)]
        result = trainer.to_device(inputs, "cpu")

        self.assertEqual(len(result), 2)
        for item in result:
            self.assertEqual(item.device, torch.device("cpu"))

    def test_should_move_dict_to_device(self):
        """测试移动字典到设备"""
        trainer = BlockQuantTrainer()

        inputs = {"key1": torch.randn(2, 4), "key2": torch.randn(2, 4)}
        result = trainer.to_device(inputs, "cpu")

        self.assertIn("key1", result)
        self.assertIn("key2", result)

    def test_should_handle_none_inputs(self):
        """测试处理 None 输入"""
        trainer = BlockQuantTrainer()

        result = trainer.to_device(None, "cpu")
        self.assertIsNone(result)

    def test_should_handle_string_inputs(self):
        """测试处理字符串输入"""
        trainer = BlockQuantTrainer()

        result = trainer.to_device("test", "cpu")
        self.assertEqual(result, "test")


class TestBlockQuantTrainerAdvanced(unittest.TestCase):
    """测试 BlockQuantTrainer 高级功能"""

    def test_should_handle_custom_lr_scheduler(self):
        """测试自定义 lr_scheduler"""
        lr_scheduler = MagicMock()
        trainer = BlockQuantTrainer(lr_scheduler=lr_scheduler)
        self.assertEqual(trainer.lr_scheduler, lr_scheduler)

    def test_should_set_lr_multiplier(self):
        """测试 lr_multiplier"""
        trainer = BlockQuantTrainer()
        self.assertEqual(trainer.lr_multiplier, 5)

    def test_should_set_dynamic_max_gap(self):
        """测试 dynamic_max_gap"""
        trainer = BlockQuantTrainer(dynamic_max_gap=10)
        self.assertEqual(trainer.dynamic_max_gap, 10)

    def test_should_set_infer_bs_coeff(self):
        """测试 infer_bs_coeff"""
        trainer = BlockQuantTrainer(infer_bs_coeff=2)
        self.assertEqual(trainer.infer_bs_coeff, 2)

    def test_should_set_amp_dtype(self):
        """测试 amp_dtype"""
        trainer = BlockQuantTrainer(amp_dtype=torch.bfloat16)
        self.assertEqual(trainer.amp_dtype, torch.bfloat16)

    def test_should_set_not_use_best_mse(self):
        """测试 not_use_best_mse"""
        trainer = BlockQuantTrainer(not_use_best_mse=True)
        self.assertTrue(trainer.not_use_best_mse)

    def test_should_set_shared_cache_keys(self):
        """测试 shared_cache_keys"""
        trainer = BlockQuantTrainer(shared_cache_keys=("key1", "key2"))
        self.assertEqual(trainer.shared_cache_keys, ("key1", "key2"))

    def test_step_should_zero_grad(self):
        """测试 step 清除梯度"""
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1)
        lr_schedule = MagicMock()

        # 计算梯度
        loss = param.sum()
        loss.backward()

        BlockQuantTrainer.step(optimizer, lr_schedule)

        # 验证梯度被清除
        self.assertTrue(param.grad is None or torch.all(param.grad == 0))

    def test_collect_best_params_should_deep_copy_data(self):
        """测试 collect_best_params 深拷贝数据"""
        wrapper = MagicMock()
        wrapper.orig_layer = MagicMock()
        original_data = torch.tensor([1.0, 2.0])
        wrapper.params = {
            "value": MagicMock(data=original_data),
        }

        block = MagicMock()
        block.named_modules.return_value = [("layer1", wrapper)]

        params = BlockQuantTrainer.collect_best_params(block)

        # 修改原始数据不应该影响收集的参数
        original_data[0] = 100.0
        self.assertNotEqual(params["layer1"]["value"][0].item(), 100.0)


if __name__ == '__main__':
    unittest.main()
