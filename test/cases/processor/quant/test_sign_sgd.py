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

from msmodelslim.processor.quant.autoround_utils.sign_sgd import SignSGD


class TestSignSGD(unittest.TestCase):
    """测试 SignSGD 优化器"""

    def test_creation_with_valid_params(self):
        """测试使用有效参数创建"""
        param = torch.randn(10, requires_grad=True)
        optimizer = SignSGD([param], lr=0.1)
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.defaults['lr'], 0.1)

    def test_creation_with_momentum(self):
        """测试带动量创建"""
        param = torch.randn(10, requires_grad=True)
        optimizer = SignSGD([param], lr=0.1, momentum=0.9)
        self.assertEqual(optimizer.defaults['momentum'], 0.9)

    def test_creation_with_weight_decay(self):
        """测试带权重衰减创建"""
        param = torch.randn(10, requires_grad=True)
        optimizer = SignSGD([param], lr=0.1, weight_decay=0.01)
        self.assertEqual(optimizer.defaults['weight_decay'], 0.01)

    def test_creation_with_nesterov(self):
        """测试带 Nesterov 动量创建"""
        param = torch.randn(10, requires_grad=True)
        optimizer = SignSGD([param], lr=0.1, momentum=0.9, nesterov=True)
        self.assertTrue(optimizer.defaults['nesterov'])

    def test_should_raise_for_negative_lr(self):
        """测试负学习率抛出异常"""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            SignSGD([param], lr=-0.1)

    def test_should_raise_for_negative_momentum(self):
        """测试负动量抛出异常"""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            SignSGD([param], lr=0.1, momentum=-0.1)

    def test_should_raise_for_negative_weight_decay(self):
        """测试负权重衰减抛出异常"""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            SignSGD([param], lr=0.1, weight_decay=-0.01)

    def test_should_raise_for_nesterov_with_zero_momentum(self):
        """测试 Nesterov 动量与零动量抛出异常"""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            SignSGD([param], lr=0.1, momentum=0, nesterov=True)

    def test_should_raise_for_nesterov_with_dampening(self):
        """测试 Nesterov 动量与非零阻尼抛出异常"""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            SignSGD([param], lr=0.1, momentum=0.9, dampening=0.1, nesterov=True)

    def test_step_should_update_params(self):
        """测试 step 更新参数"""
        param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1)

        # 计算梯度
        loss = param.sum()
        loss.backward()

        # 记录更新前的值
        old_param = param.clone()

        # 执行更新
        optimizer.step()

        # 验证参数已更新（sign 梯度下降，梯度为正时参数减小）
        self.assertTrue(torch.all(param < old_param))

    def test_step_with_sign_gradient_should_use_sign(self):
        """测试 step 使用梯度的符号"""
        param = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1)

        # 梯度为 [1, 1, 1]
        loss = param.sum()
        loss.backward()

        old_param = param.clone()
        optimizer.step()

        # sign(梯度) = [1, 1, 1]，参数应该减少 lr * 1
        expected = old_param - 0.1
        self.assertTrue(torch.allclose(param, expected))

    def test_step_with_zero_gradient_should_not_change(self):
        """测试零梯度时参数不变"""
        param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1)

        # 不计算梯度
        optimizer.step()

        # 参数应该不变
        self.assertTrue(torch.allclose(param, torch.tensor([1.0, 2.0, 3.0])))

    def test_step_with_momentum_should_accumulate(self):
        """测试动量累积"""
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1, momentum=0.9)

        # 第一步
        loss = param.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 第二步
        loss = param.sum()
        loss.backward()
        optimizer.step()

        # 动量应该累积
        self.assertIsNotNone(optimizer.state[param].get('momentum_buffer'))

    def test_step_with_maximize_should_increase_params(self):
        """测试 maximize 模式增加参数"""
        param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1, maximize=True)

        loss = param.sum()
        loss.backward()

        old_param = param.clone()
        optimizer.step()

        # maximize 模式下，参数应该增加
        self.assertTrue(torch.all(param > old_param))

    def test_step_with_weight_decay_should_apply_decay(self):
        """测试权重衰减"""
        param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1, weight_decay=0.01)

        loss = param.sum()
        loss.backward()

        old_param = param.clone()
        optimizer.step()

        # 权重衰减会增加梯度，导致参数变化更大
        self.assertTrue(torch.all(param < old_param))

    def test_zero_grad_should_clear_gradients(self):
        """测试 zero_grad 清除梯度"""
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1)

        loss = param.sum()
        loss.backward()

        self.assertIsNotNone(param.grad)
        optimizer.zero_grad()
        # zero_grad 后梯度应该为 None 或 0
        if param.grad is not None:
            self.assertTrue(torch.all(param.grad == 0))

    def test_setstate_should_set_defaults(self):
        """测试 __setstate__ 设置默认值"""
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = SignSGD([param], lr=0.1)

        # 模拟旧状态
        state = {'state': {}, 'param_groups': [{'params': [param]}]}
        optimizer.__setstate__(state)

        self.assertFalse(optimizer.param_groups[0]['nesterov'])
        self.assertFalse(optimizer.param_groups[0]['maximize'])
        self.assertIsNone(optimizer.param_groups[0]['foreach'])
        self.assertFalse(optimizer.param_groups[0]['differentiable'])


if __name__ == '__main__':
    unittest.main()
