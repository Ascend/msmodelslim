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

import mindspore.nn as nn
from mindspore.ops import operations as P


def conv_bn_relu(
    in_channel,
    out_channel,
    kernel_size,
    stride,
    depth_wise,
):
    output = [
        nn.Conv2d(
            in_channel, out_channel, kernel_size, stride, pad_mode="same", group=1 if not depth_wise else in_channel
        ),
        nn.BatchNorm2d(out_channel),
    ]

    output.append(nn.get_activation('relu6'))
    return nn.SequentialCell(output)


class TestNetMindSpore(nn.Cell):
    """
    MobileNet V1 backbone
    """

    def __init__(self, class_num=10, features_only=False):
        super(TestNetMindSpore, self).__init__()
        self.features_only = features_only
        cnn1 = [
            conv_bn_relu(3, 32, 3, 2, False),
        ]
        cnn2 = [
            conv_bn_relu(32, 32, 3, 2, False),
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


class TestNetMindSpore2(nn.Cell):
    """
    MobileNet V1 backbone
    """

    def __init__(self, class_num=10, features_only=False):
        super(TestNetMindSpore2, self).__init__()
        self.features_only = features_only
        cnn = [
            conv_bn_relu(3, 32, 3, 2, False),
        ]
        self.backbone = nn.SequentialCell(cnn)
        self.fc = nn.Dense(32, class_num)

    def construct(self, x):
        output = x
        output = self.backbone(output)
        output = P.ReduceMean()(output, (2, 3))
        output = self.fc(output)
        return output


class TestNetMindSpore3(nn.Cell):
    """
    MobileNet V1 backbone
    """

    def __init__(self, class_num=10, features_only=False):
        super().__init__()
        self.features_only = features_only
        cnn = [
            conv_bn_relu(3, 32, 3, 2, False),
        ]
        self.backbone = nn.SequentialCell(cnn)
        self.fc = nn.Dense(32, class_num)

    def construct(self, x, y):
        output = x
        output = self.backbone(output)
        output = P.ReduceMean()(output, (2, 3))
        output = self.fc(output)
        return output


class TestNetMindSpore4(nn.Cell):
    """
    TestNet
    """

    def __init__(self, class_num=10):
        super().__init__()
        self.backbone = nn.SequentialCell([conv_bn_relu(3, 64, 3, 2, False), conv_bn_relu(64, 32, 3, 1, False)])
        self.fc = nn.Dense(32, class_num)

    def forward(self, x):
        x = self.backbone(x)
        x = P.ReduceMean(keep_dims=True)(x)
        x = x.Flatten()
        x = self.fc(x)
        return x


def get_model():
    return TestNetMindSpore(class_num=10)


class SampleModel(nn.Cell):
    def __init__(self, num_class=10, num_channel=3):
        super(SampleModel, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 16, 3, pad_mode='valid')
        self.conv2 = nn.Conv2d(16, 16, 3, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 3 * 3, 10)
        self.fc2 = nn.Dense(10, num_class)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        y = self.relu(x)
        return y


class LrdSampleNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.embedding = nn.SequentialCell(
            [
                nn.Embedding(16, 32),
                nn.Dense(32, 64),
                nn.ReLU(),
            ]
        )

        self.feature = nn.SequentialCell(
            [
                nn.Conv2d(64, 128, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(128, 64, 1, has_bias=True),
                nn.ReLU(),
            ]
        )
        self.pool = nn.AvgPool2d((7, 7), stride=2)
        self.flatten = nn.Flatten()
        self.inner = nn.Dense(64 * 5 * 5, 512)
        self.classifier = nn.SequentialCell(
            [
                nn.Dense(512, 256),
                nn.Dense(256, 10),
            ]
        )

    def construct(self, inputs):
        shortcut = self.embedding(inputs)
        shortcut = shortcut.transpose([0, 3, 1, 2])
        next_node = self.feature(shortcut)
        next_node = next_node + shortcut
        next_node = self.pool(next_node)
        next_node = self.flatten(next_node)
        next_node = self.inner(next_node)
        next_node = self.classifier(next_node)
        return next_node


class MsTeacherModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.teacher_fc = nn.Dense(1, 1)

    def construct(self, inputs):
        logits = self.teacher_fc(inputs)
        loss = logits / 2
        return loss


class MsStudentModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.student_fc = nn.Dense(1, 1)

    def construct(self, inputs):
        logits = self.student_fc(inputs)
        loss = logits / 2
        return loss
