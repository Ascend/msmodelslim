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
# pylint: disable=ungrouped-imports

import unittest

import pytest
import numpy as np
from mindspore import Tensor, context
from mindspore import dtype as mstype

from ascend_utils.mindspore.knowledge_distill.distill_losses_func_ms import (
    REDUCE_NONE,
    REDUCE_MEAN,
    REDUCE_SUM,
    mse_loss_ms,
    update_logits_by_temperature_ms,
    KDMse,
    KLDivLoss,
    Mse,
    KDCrossEntropy,
    HiddenMse,
    MMD,
    DISTILL_LOSS_FUNC_MS,
)
from ascend_utils.mindspore.knowledge_distill.distill_losses_func_ms import (
    HardKDCrossEntropy,  # noqa: F401
)  # noqa: F401
from mindspore import nn, ops  # noqa: F401
import mindspore  # noqa: F401


class TestMSELossMS:
    """测试mse_loss_ms函数"""

    @staticmethod
    def setup_method():
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_mse_loss_none_reduction():
        """测试无缩减的MSE损失"""
        state_s = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)
        state_t = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)

        loss = mse_loss_ms(state_s, state_t, reduction=REDUCE_NONE)

        expected = Tensor(np.array([0.0, 0.0, 0.0]), dtype=mstype.float32)
        assert np.allclose(loss.asnumpy(), expected.asnumpy())

    @staticmethod
    def test_mse_loss_mean_reduction():
        """测试均值缩减的MSE损失"""
        state_s = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)
        state_t = Tensor(np.array([2.0, 3.0, 4.0]), dtype=mstype.float32)

        loss = mse_loss_ms(state_s, state_t, reduction=REDUCE_MEAN)

        # (1^2 + 1^2 + 1^2) / 3 = 1.0
        expected = 1.0
        assert np.allclose(loss.asnumpy(), expected)

    @staticmethod
    def test_mse_loss_sum_reduction():
        """测试求和缩减的MSE损失"""
        state_s = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)
        state_t = Tensor(np.array([2.0, 3.0, 4.0]), dtype=mstype.float32)

        loss = mse_loss_ms(state_s, state_t, reduction=REDUCE_SUM)

        # 1^2 + 1^2 + 1^2 = 3.0
        expected = 3.0
        assert np.allclose(loss.asnumpy(), expected)

    @staticmethod
    def test_mse_loss_2d_tensor():
        """测试2D张量的MSE损失"""
        state_s = Tensor(np.random.randn(2, 3).astype(np.float32))
        state_t = Tensor(np.random.randn(2, 3).astype(np.float32))

        loss = mse_loss_ms(state_s, state_t, reduction=REDUCE_MEAN)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()


class TestUpdateLogitsByTemperatureMS:
    """测试update_logits_by_temperature_ms函数"""

    @staticmethod
    def setup_method():
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_update_logits_scalar_temperature():
        """测试标量温度参数"""
        logits_s = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)
        logits_t = Tensor(np.array([2.0, 3.0, 4.0]), dtype=mstype.float32)
        temperature = 2.0

        beta_s, beta_t = update_logits_by_temperature_ms(logits_s, logits_t, temperature)

        expected_s = Tensor(np.array([0.5, 1.0, 1.5]), dtype=mstype.float32)
        expected_t = Tensor(np.array([1.0, 1.5, 2.0]), dtype=mstype.float32)

        assert np.allclose(beta_s.asnumpy(), expected_s.asnumpy())
        assert np.allclose(beta_t.asnumpy(), expected_t.asnumpy())

    @staticmethod
    def test_update_logits_tensor_temperature():
        """测试张量温度参数"""
        logits_s = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=mstype.float32)
        logits_t = Tensor(np.array([[2.0, 3.0], [4.0, 5.0]]), dtype=mstype.float32)
        temperature = Tensor(np.array([2.0, 3.0]), dtype=mstype.float32)

        beta_s, beta_t = update_logits_by_temperature_ms(logits_s, logits_t, temperature)

        assert beta_s.shape == logits_s.shape
        assert beta_t.shape == logits_t.shape


class TestKDMse(unittest.TestCase):
    """测试KDMse类"""

    def setUp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        self.kd_mse = KDMse()

    def test_construct_default_temperature(self):
        """测试默认温度的前向计算"""
        logits_s = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)
        logits_t = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)

        loss = self.kd_mse.construct(logits_s, logits_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        assert np.allclose(loss.asnumpy(), 0.0)

    def test_construct_custom_temperature(self):
        """测试自定义温度的前向计算"""
        logits_s = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)
        logits_t = Tensor(np.array([2.0, 3.0, 4.0]), dtype=mstype.float32)
        temperature = 2.0

        loss = self.kd_mse.construct(logits_s, logits_t, temperature)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()

    def test_construct_2d_tensor(self):
        """测试2D张量的前向计算"""
        logits_s = Tensor(np.random.randn(2, 3).astype(np.float32))
        logits_t = Tensor(np.random.randn(2, 3).astype(np.float32))

        loss = self.kd_mse.construct(logits_s, logits_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()


class TestKLDivLoss(unittest.TestCase):
    """测试KLDivLoss类"""

    def setUp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        self.kl_div = KLDivLoss()

    def test_construct_default_temperature(self):
        """测试默认温度的前向计算"""
        logits_s = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=mstype.float32)
        logits_t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=mstype.float32)

        loss = self.kl_div.construct(logits_s, logits_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        # 当两个分布相同时，KL散度应该接近0
        assert loss.asnumpy() >= 0.0

    def test_construct_custom_temperature(self):
        """测试自定义温度的前向计算"""
        logits_s = Tensor(np.random.randn(2, 3).astype(np.float32))
        logits_t = Tensor(np.random.randn(2, 3).astype(np.float32))
        temperature = 2.0

        loss = self.kl_div.construct(logits_s, logits_t, temperature)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        assert loss.asnumpy() >= 0.0

    def test_construct_different_dtypes(self):
        """测试不同数据类型的输入"""
        logits_s = Tensor(np.array([[1, 2], [3, 4]]), dtype=mstype.int32)
        logits_t = Tensor(np.array([[1, 2], [3, 4]]), dtype=mstype.int32)

        loss = self.kl_div.construct(logits_s, logits_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()


class TestMse(unittest.TestCase):
    """测试Mse类"""

    def setUp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        self.mse = Mse()

    def test_construct(self):
        """测试前向计算"""
        t_pred = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)
        s_pred = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)

        loss = self.mse.construct(t_pred, s_pred)

        assert loss == 0.0


class TestKDCrossEntropy(unittest.TestCase):
    """测试KDCrossEntropy类"""

    def setUp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        self.kd_ce = KDCrossEntropy()

    def test_construct_default_temperature(self):
        """测试默认温度的前向计算"""
        # 注意：这里使用较小的值避免softmax数值不稳定
        logits_s = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]]), dtype=mstype.float32)
        logits_t = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]]), dtype=mstype.float32)

        loss = self.kd_ce.construct(logits_s, logits_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        # 当两个分布相同时，交叉熵应该接近熵
        assert loss.asnumpy() >= 0.0

    def test_construct_custom_temperature(self):
        """测试自定义温度的前向计算"""
        logits_s = Tensor(np.random.randn(2, 3).astype(np.float32))
        logits_t = Tensor(np.random.randn(2, 3).astype(np.float32))
        temperature = 2.0

        loss = self.kd_ce.construct(logits_s, logits_t, temperature)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        assert loss.asnumpy() >= 0.0

    def test_construct_different_dtypes(self):
        """测试不同数据类型的输入"""
        logits_s = Tensor(np.array([[1, 2], [3, 4]]), dtype=mstype.int32)
        logits_t = Tensor(np.array([[1, 2], [3, 4]]), dtype=mstype.int32)

        loss = self.kd_ce.construct(logits_s, logits_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()


class TestHiddenMse(unittest.TestCase):
    """测试HiddenMse类"""

    def setUp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        self.hidden_mse = HiddenMse()

    def test_construct_no_mask(self):
        """测试无掩码的前向计算"""
        state_s = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)
        state_t = Tensor(np.array([1.0, 2.0, 3.0]), dtype=mstype.float32)

        loss = self.hidden_mse.construct(state_s, state_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        assert np.allclose(loss.asnumpy(), 0.0)


class TestMMD(unittest.TestCase):
    """测试MMD类"""

    def setUp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    def test_construct_no_mask_2d(self):
        """测试2D张量无掩码的前向计算"""
        batch_size = 2
        mmd = MMD(batch_size)

        state_s = Tensor(np.random.randn(4, 8).astype(np.float32))
        state_t = Tensor(np.random.randn(4, 8).astype(np.float32))

        loss = mmd.construct(state_s, state_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        assert loss.asnumpy() >= 0.0

    def test_construct_no_mask_3d(self):
        """测试3D张量无掩码的前向计算"""
        batch_size = 2
        mmd = MMD(batch_size)

        state_s = Tensor(np.random.randn(2, 5, 8).astype(np.float32))
        state_t = Tensor(np.random.randn(2, 5, 8).astype(np.float32))

        loss = mmd.construct(state_s, state_t)

        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        assert loss.asnumpy() >= 0.0


class TestDistillLossFuncMS(unittest.TestCase):
    """测试DISTILL_LOSS_FUNC_MS字典"""

    def test_distill_loss_func_dict(self):
        """测试蒸馏损失函数字典"""
        assert "KDCrossEntropy" in DISTILL_LOSS_FUNC_MS
        assert DISTILL_LOSS_FUNC_MS["KDCrossEntropy"] == KDCrossEntropy

        # 测试可以实例化
        loss_func = DISTILL_LOSS_FUNC_MS["KDCrossEntropy"]()
        assert isinstance(loss_func, KDCrossEntropy)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    def test_multiple_loss_functions(self):
        """测试多个损失函数的一致性"""
        logits_s = Tensor(np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]), dtype=mstype.float32)
        logits_t = Tensor(np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]), dtype=mstype.float32)

        # 测试多个损失函数
        kd_mse = KDMse()
        kl_div = KLDivLoss()
        kd_ce = KDCrossEntropy()

        loss1 = kd_mse.construct(logits_s, logits_t)
        loss2 = kl_div.construct(logits_s, logits_t)
        loss3 = kd_ce.construct(logits_s, logits_t)

        # 所有损失都应该是非负的
        assert loss1.asnumpy() >= 0.0
        assert loss2.asnumpy() >= 0.0
        assert loss3.asnumpy() >= 0.0

        # 所有损失都应该是标量
        assert loss1.shape == ()
        assert loss2.shape == ()
        assert loss3.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
