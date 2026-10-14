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
# pylint: disable=consider-using-from-import,super-with-arguments

# 导入相关依赖
import mindspore.nn as nn
from mindspore.ops import operations as P

from msmodelslim import set_logger_level
from msmodelslim import logger as msmodelslim_logger  # noqa: F401
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from msmodelslim.common.knowledge_distill.knowledge_distill import get_distill_model


class ConvBNReLUConfig:
    def __init__(self, in_channel, out_channel, kernel_size, stride, depth_wise, activation='relu6'):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth_wise = depth_wise
        self.activation = activation

    def build(self):
        output = [
            nn.Conv2d(
                self.in_channel,
                self.out_channel,
                self.kernel_size,
                self.stride,
                pad_mode="same",
                group=1 if not self.depth_wise else self.in_channel,
            ),
            nn.BatchNorm2d(self.out_channel),
        ]
        if self.activation:
            output.append(nn.get_activation(self.activation))
        return nn.SequentialCell(output)


def conv_bn_relu(config: ConvBNReLUConfig):
    return config.build()


# 加载预训练模型
class TestNetMindSpore(nn.Cell):
    def __init__(self, class_num=10, features_only=False):
        super(TestNetMindSpore, self).__init__()
        self.features_only = features_only
        cnn1 = [
            conv_bn_relu(ConvBNReLUConfig(3, 32, 3, 2, False)),
        ]
        cnn2 = [
            conv_bn_relu(ConvBNReLUConfig(32, 32, 3, 2, False)),
        ]
        self.network1 = nn.SequentialCell(cnn1)
        self.network2 = nn.SequentialCell(cnn2)
        self.fc = nn.Dense(32, class_num)

    def construct(self, x):
        output = x
        output = self.network1(output)
        output = self.network2(output)
        output = P.ReduceMean()(output, (2, 3))
        output = self.fc(output)
        return output


# 创建模型
model = TestNetMindSpore()

# 根据实际情况配置
set_logger_level("info")
# 创建配置类
distill_config = KnowledgeDistillConfig()
# 添加软标签
distill_config.add_output_soft_label(
    {
        "t_output_idx": 1,
        "s_output_idx": 1,
        "loss_func": [{"func_name": "KDCrossEntropy", "func_weight": 1, "temperature": 1}],
    }
)

# 获得知识蒸馏模型
distill_model = get_distill_model(model, model, distill_config)
# 获取知识蒸馏后的学生模型
student_model = distill_model.get_student_model()
