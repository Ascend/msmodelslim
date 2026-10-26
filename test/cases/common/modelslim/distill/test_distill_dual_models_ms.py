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

import logging
import unittest
from unittest.mock import Mock, patch
import pytest
import numpy as np
from mindspore import Tensor, nn, context
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal

# 导入被测试的类
from ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms import DistillDualModelsMs


class SimpleStudentModel(nn.Cell):
    """简单的学生模型用于测试"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, has_bias=True, weight_init=Normal(0.02))
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class SimpleTeacherModel(nn.Cell):
    """简单的教师模型用于测试"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, has_bias=True, weight_init=Normal(0.02))
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class MultiOutputModel(nn.Cell):
    """多输出模型用于测试"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, has_bias=True, weight_init=Normal(0.02))
        self.conv2 = nn.Conv2d(16, 32, 3, has_bias=True, weight_init=Normal(0.02))

    def construct(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x1, x2


class TestConfig:
    """测试配置类"""

    def __init__(self):
        self.hard_label_loss_weight = 0.5
        self.inter_matches = []
        self.output_matches = []
        self.model_parallel = False
        self.output_replace_idx = 0


class TestDistillDualModelsMs(unittest.TestCase):
    """测试DistillDualModelsMs类"""

    def setUp(self):
        """设置测试环境"""
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        self.config = TestConfig()
        self.student_model = SimpleStudentModel()
        self.teacher_model = SimpleTeacherModel()
        self.output_shapes = [(-1, 16, 6, 6), (-1, 32, 4, 4)]

    @patch('ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms.DistillLossesManager')
    @patch('ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms.logging')
    def test_initialization(self, mock_logging, mock_distill_manager):
        """测试初始化"""
        # 模拟DistillLossesManager实例
        mock_manager_instance = Mock()
        mock_distill_manager.return_value = mock_manager_instance

        # 创建测试实例
        distill_model = DistillDualModelsMs(self.config, self.student_model, self.teacher_model, self.output_shapes)

        # 验证属性设置
        assert distill_model.student_model == self.student_model
        assert distill_model.teacher_model == self.teacher_model
        assert distill_model.hard_label_loss_weight == 0.5
        assert distill_model.distill_loss == 0

        # 验证DistillLossesManager被正确初始化
        mock_distill_manager.assert_called_once_with(
            self.config, self.student_model, self.teacher_model, self.output_shapes
        )

        # 验证日志记录
        mock_logging.info.assert_called_with("DistillDualModels inited.")

    @patch('ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms.DistillLossesManager')
    def test_construct_basic(self, mock_distill_manager):
        """测试基础前向计算"""
        # 设置模拟
        mock_manager_instance = Mock()
        mock_manager_instance.construct.return_value = (Tensor(3.0, dtype=mstype.float32), {})
        mock_distill_manager.return_value = mock_manager_instance

        # 创建测试实例
        distill_model = DistillDualModelsMs(self.config, self.student_model, self.teacher_model, self.output_shapes)

        # 模拟输入数据
        test_input = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))

        # 执行前向计算
        result = distill_model.construct(test_input)

        # 验证结果
        assert isinstance(result, Tensor)
        assert result.asnumpy() == 3.0

        # 验证损失管理器被调用
        mock_manager_instance.construct.assert_called_once()

    @patch('ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms.DistillLossesManager')
    def test_get_student_model(self, mock_distill_manager):
        """测试获取学生模型"""
        mock_manager_instance = Mock()
        mock_distill_manager.return_value = mock_manager_instance

        distill_model = DistillDualModelsMs(self.config, self.student_model, self.teacher_model, self.output_shapes)

        # 获取学生模型
        result = distill_model.get_student_model()

        # 验证模块恢复被调用
        mock_manager_instance.restore_modules.assert_called_once_with(self.student_model, self.teacher_model)

        # 验证返回正确的学生模型
        assert result == self.student_model

    @patch('ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms.DistillLossesManager')
    def test_construct_with_multi_output_models(self, mock_distill_manager):
        """测试多输出模型的前向计算"""
        mock_manager_instance = Mock()
        mock_distill_manager.return_value = mock_manager_instance

        # 使用多输出模型
        multi_student = MultiOutputModel()
        multi_teacher = MultiOutputModel()

        mock_manager_instance.construct.return_value = (Tensor(1.5, dtype=mstype.float32), {})

        distill_model = DistillDualModelsMs(self.config, multi_student, multi_teacher, self.output_shapes)

        # 执行前向计算
        test_input = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        result = distill_model.construct(test_input)

        # 验证损失管理器被正确调用
        mock_manager_instance.construct.assert_called_once()

        # 验证返回损失值
        assert result.asnumpy() == 1.5

    @patch('ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms.DistillLossesManager')
    def test_property_access(self, mock_distill_manager):
        """测试属性访问"""
        distill_model = DistillDualModelsMs(self.config, self.student_model, self.teacher_model, self.output_shapes)

        # 验证属性可访问
        assert distill_model.student_model == self.student_model
        assert distill_model.teacher_model == self.teacher_model
        assert distill_model.hard_label_loss_weight == 0.5

    @patch('ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms.DistillLossesManager')
    def test_edge_cases(self, mock_distill_manager):
        """测试边界情况"""
        # 测试空输出形状
        distill_model = DistillDualModelsMs(self.config, self.student_model, self.teacher_model, output_shapes=None)

        assert distill_model.distill_losses_manager is mock_distill_manager.return_value

        # 测试零hard_label_loss_weight
        self.config.hard_label_loss_weight = 0.0
        distill_model = DistillDualModelsMs(self.config, self.student_model, self.teacher_model, self.output_shapes)

        assert distill_model.hard_label_loss_weight == 0.0

    @patch('ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms.DistillLossesManager')
    def test_construct_with_loss_details(self, mock_distill_manager):
        """测试带详细损失信息的前向计算"""
        mock_manager_instance = Mock()
        mock_distill_manager.return_value = mock_manager_instance

        # 模拟返回详细损失信息
        loss_details = {"kd_loss": Tensor(1.0, dtype=mstype.float32), "hard_loss": Tensor(0.5, dtype=mstype.float32)}
        mock_manager_instance.construct.return_value = (Tensor(1.5, dtype=mstype.float32), loss_details)

        distill_model = DistillDualModelsMs(self.config, self.student_model, self.teacher_model, self.output_shapes)

        test_input = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        result = distill_model.construct(test_input)

        # 验证总损失正确
        assert result.asnumpy() == 1.5


if __name__ == "__main__":
    # 设置日志级别避免测试时输出过多信息
    logging.getLogger().setLevel(logging.ERROR)

    pytest.main([__file__, "-v", "-s"])
