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
# pylint: disable=super-with-arguments,use-implicit-booleaness-not-comparison,useless-parent-delegation

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, context
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal

from ascend_utils.common.knowledge_distill.utils import replace_module  # noqa: F401
from ascend_utils.mindspore.knowledge_distill.distill_losses_manager_ms import (
    SaveOutputShapeModule,
    GetOutputShapeModule,
    SaveOutputModule,
    GetOutputModule,
    LossModuleBase,
    LossModule,
    DistillLossesManagerBase,
    DistillLossesManager,
)


class SimpleNet(nn.Cell):
    """简单的测试网络"""

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, has_bias=True, weight_init=Normal(0.02))
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class MultiOutputNet(nn.Cell):
    """多输出测试网络"""

    def __init__(self):
        super(MultiOutputNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, has_bias=True, weight_init=Normal(0.02))
        self.conv2 = nn.Conv2d(16, 32, 3, has_bias=True, weight_init=Normal(0.02))

    def construct(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x1, x2


class SimpleLoss(nn.Cell):
    """简单的损失函数"""

    def __init__(self):
        super(SimpleLoss, self).__init__()

    def construct(self, s_output, t_output, temperature=None):
        if temperature:
            return mindspore.ops.abs(s_output - t_output).mean() * temperature
        return mindspore.ops.abs(s_output - t_output).mean()


class TestConfig:
    """测试配置类"""

    def __init__(self):
        self.inter_matches = []
        self.output_matches = []
        self.hard_label_loss_weight = 0.5
        self.model_parallel = False
        self.output_replace_idx = 0


class TestSaveOutputShapeModule:
    """测试SaveOutputShapeModule类"""

    @staticmethod
    def setup_method():
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_initialization():
        """测试初始化"""
        module = nn.ReLU()
        save_module = SaveOutputShapeModule(module, "test", 0)

        assert save_module.name == "test"
        assert save_module.module == module
        assert save_module.output_idx == 0

    @staticmethod
    def test_construct_single_output():
        """测试单输出构造"""
        module = nn.ReLU()
        save_module = SaveOutputShapeModule(module, "test", None)
        input_tensor = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))

        output = save_module(input_tensor)

        # 检查输出是否正确
        assert output.shape == (2, 3, 4, 5)

    @staticmethod
    def test_construct_multi_output():
        """测试多输出构造"""
        module = MultiOutputNet()
        save_module = SaveOutputShapeModule(module, "test", 1)
        input_tensor = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))

        output = save_module(input_tensor)

        # 检查第二个输出的形状
        assert len(output) == 2
        assert output[1].shape == (2, 32, 8, 8)


class TestGetOutputShapeModule:
    """测试GetOutputShapeModule类"""

    @staticmethod
    def setup_method():
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_get_output_shape():
        """测试获取输出形状"""
        module = nn.ReLU()
        save_module = SaveOutputShapeModule(module, "test", None)
        get_module = GetOutputShapeModule(save_module)

        # 先运行一次以记录形状
        input_tensor = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
        save_module(input_tensor)

        output_shape = get_module.get_output_shape()

        # 检查输出形状是否正确
        assert output_shape is not None


class TestSaveOutputModule:
    """测试SaveOutputModule类"""

    @staticmethod
    def setup_method():
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_initialization():
        """测试初始化"""
        module = nn.ReLU()
        output_shape = (2, 3, 4, 5)
        save_module = SaveOutputModule(module, "test", None, output_shape)

        assert save_module.name == "test"
        assert save_module.module == module
        assert save_module.output_idx is None


class TestGetOutputModule:
    """测试GetOutputModule类"""

    @staticmethod
    def setup_method():
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_get_output():
        """测试获取输出"""
        module = nn.ReLU()
        output_shape = (2, 3, 4, 5)
        save_module = SaveOutputModule(module, "test", None, output_shape)
        get_module = GetOutputModule(save_module)

        # 测试获取输出
        output = get_module._get_output()
        assert output is not None


class TestLossModuleBase:
    """测试LossModuleBase类"""

    @staticmethod
    def setup_method():
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_initialization():
        """测试初始化"""
        match = {
            "loss_func": [
                {"func_instance": SimpleLoss(), "func_weight": 1.0, "temperature": None},
                {"func_instance": SimpleLoss(), "func_weight": 0.5, "temperature": 2.0},
            ],
            "s_output_idx": 0,
            "t_output_idx": 0,
        }

        loss_module = LossModuleBase(match)

        assert len(loss_module.modules) == 2
        assert len(loss_module.loss_scale_weights) == 2
        assert len(loss_module.temperatures) == 2
        assert loss_module.s_output_idx == 0
        assert loss_module.t_output_idx == 0
        assert loss_module.loss_scale_weights[0] == 1.0
        assert loss_module.loss_scale_weights[1] == 0.5
        assert loss_module.temperatures[1] == 2.0

    @staticmethod
    def test_calc_loss():
        """测试损失计算"""
        match = {
            "loss_func": [{"func_instance": SimpleLoss(), "func_weight": 1.0, "temperature": None}],
            "s_output_idx": 0,
            "t_output_idx": 0,
        }

        loss_module = LossModuleBase(match)

        s_output = Tensor(np.ones((2, 3), dtype=np.float32))
        t_output = Tensor(np.zeros((2, 3), dtype=np.float32))

        loss = loss_module.calc_loss(s_output, t_output)

        # 绝对差值的平均值应为1.0
        expected_loss = 1.0
        assert abs(loss.asnumpy() - expected_loss) < 1e-6

    @staticmethod
    def test_calc_loss_with_temperature():
        """测试带温度参数的损失计算"""
        match = {
            "loss_func": [{"func_instance": SimpleLoss(), "func_weight": 1.0, "temperature": 2.0}],
            "s_output_idx": 0,
            "t_output_idx": 0,
        }

        loss_module = LossModuleBase(match)

        s_output = Tensor(np.ones((2, 3), dtype=np.float32))
        t_output = Tensor(np.zeros((2, 3), dtype=np.float32))

        loss = loss_module.calc_loss(s_output, t_output)

        # 绝对差值的平均值乘以温度2.0应为2.0
        expected_loss = 2.0
        assert abs(loss.asnumpy() - expected_loss) < 1e-6


class TestLossModule:
    """测试LossModule类"""

    @staticmethod
    def setup_method():
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_construct_with_modules():
        """测试使用模块的构造方法"""
        # 创建保存输出模块
        module = nn.ReLU()
        output_shape = (2, 3)
        s_save_module = SaveOutputModule(module, "s_test", None, output_shape)
        t_save_module = SaveOutputModule(module, "t_test", None, output_shape)

        match = {
            "loss_func": [{"func_instance": SimpleLoss(), "func_weight": 1.0, "temperature": None}],
            "s_output_idx": 0,
            "t_output_idx": 0,
        }

        loss_module = LossModule(match, t_save_module, s_save_module)

        # 传入虚拟输出
        dummy_output = (Tensor(np.zeros((1, 1), dtype=np.float32)),)
        loss = loss_module(dummy_output, dummy_output)

        # 检查是否有损失输出
        assert loss is not None

    @staticmethod
    def test_construct_with_direct_outputs():
        """测试直接使用输出的构造方法"""
        match = {
            "loss_func": [{"func_instance": SimpleLoss(), "func_weight": 1.0, "temperature": None}],
            "s_output_idx": 0,
            "t_output_idx": 0,
        }

        loss_module = LossModule(match)

        s_output = (Tensor(np.ones((2, 3), dtype=np.float32)), Tensor(np.zeros((1, 1), dtype=np.float32)))
        t_output = (Tensor(np.zeros((2, 3), dtype=np.float32)), Tensor(np.zeros((1, 1), dtype=np.float32)))

        loss = loss_module(s_output, t_output)

        # 损失应为1.0
        expected_loss = 1.0
        assert abs(loss.asnumpy() - expected_loss) < 1e-6


class TestDistillLossesManagerBase:
    """测试DistillLossesManagerBase类"""

    @staticmethod
    def setup_method():
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_initialization():
        manager = DistillLossesManagerBase(TestConfig())

        assert manager.inter_matches == []
        assert manager.output_matches == []
        assert manager.hard_label_loss_weight == 0.5
        assert manager.output_replace_idx == 0

    @staticmethod
    def test_index_able():
        manager = DistillLossesManagerBase(TestConfig())

        # 测试列表
        result = manager.index_able([1, 2, 3])
        assert result == [1, 2, 3]

        # 测试元组
        result = manager.index_able((1, 2, 3))
        assert result == (1, 2, 3)

        # 测试单个元素
        result = manager.index_able(5)
        assert result == [5]

    @staticmethod
    def test_try_merge_to_ori_output():
        """测试合并到原始输出"""
        config = TestConfig()
        config.output_replace_idx = 0
        manager = DistillLossesManagerBase(config)

        distill_loss = Tensor(2.0, dtype=mstype.float32)
        s_outputs = [Tensor(1.0, dtype=mstype.float32), Tensor(3.0, dtype=mstype.float32)]

        result = manager.try_merge_to_ori_output(distill_loss, s_outputs)

        # 检查结果不为None
        assert result is not None

    @staticmethod
    def test_add_t_loss():
        """测试添加教师损失"""
        config = TestConfig()
        config.output_replace_idx = 0
        manager = DistillLossesManagerBase(config)

        t_outputs = Tensor(2.0, dtype=mstype.float32)
        s_outputs = [Tensor(1.0, dtype=mstype.float32), Tensor(3.0, dtype=mstype.float32)]

        result = manager.add_t_loss(t_outputs, s_outputs)

        # 检查结果不为None
        assert result is not None


class TestDistillLossesManager:
    """测试DistillLossesManager类"""

    @staticmethod
    def setup_method():
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    @staticmethod
    def test_initialization():
        """测试初始化"""
        config = TestConfig()
        s_model = SimpleNet()
        t_model = SimpleNet()
        manager = DistillLossesManager(config, s_model, t_model)

        assert isinstance(manager.s_name2module, dict)
        assert isinstance(manager.t_name2module, dict)
        assert manager.loss_modules == []
        assert manager.get_output_shape_modules == []

    @staticmethod
    def test_compute_loss_ms():
        """测试计算损失（MindSpore模式）"""
        # 添加输出匹配
        output_match = {
            "loss_func": [{"func_instance": SimpleLoss(), "func_weight": 1.0, "temperature": None}],
            "s_output_idx": 0,
            "t_output_idx": 0,
        }
        config = TestConfig()
        s_model = SimpleNet()
        t_model = SimpleNet()
        config.output_matches = [output_match]

        manager = DistillLossesManager(config, s_model, t_model)

        s_output = Tensor(np.ones((2, 3), dtype=np.float32))
        t_output = Tensor(np.zeros((2, 3), dtype=np.float32))

        total_loss, _ = manager.compute_loss_ms(s_output, t_output)

        # 检查损失计算是否成功
        assert total_loss is not None

    @staticmethod
    def test_get_module_output_shapes():
        """测试获取模块输出形状"""
        # 添加中间匹配
        inter_match = {"t_module": "conv", "s_module": "conv", "s_output_idx": 0, "t_output_idx": 0}
        config = TestConfig()
        s_model = SimpleNet()
        t_model = SimpleNet()
        config.inter_matches = [inter_match]

        manager = DistillLossesManager(config, s_model, t_model)

        # 运行一次以记录形状
        input_tensor = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        s_model(input_tensor)
        t_model(input_tensor)

        output_shapes = manager.get_module_output_shapes()

        # 应该包含教师和学生模型的形状
        assert len(output_shapes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
