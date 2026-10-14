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
# pylint: disable=consider-using-from-import,redefined-outer-name

import pytest
import torch
import torch.nn as nn

from msmodelslim.pytorch.prune.prune_torch import PruneTorch
from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook


class MyConvTestNet(nn.Module):
    def __init__(self, groups=1) -> None:
        super().__init__()
        self.features = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2, groups=groups)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.features(x_in)
        return x_out


class MyLinearTestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Linear(12, 24)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.features(x_in)
        return x_out


class MyBNTestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2, groups=1)
        self.batch_normal = nn.BatchNorm2d(64)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.features(x_in)
        x_out = self.batch_normal(x_out)
        return x_out


@pytest.fixture(scope="module")
def inputs_like_img():
    return torch.ones([1, 4, 22, 22]).type(torch.float32)


@pytest.fixture(scope="module")
def inputs_of_linear():
    return torch.ones([1, 12]).type(torch.float32)


@pytest.fixture(scope="function")
def dag_conv(inputs_like_img):
    model = MyConvTestNet()
    return DagTorchHook(model, inputs_like_img)


@pytest.fixture(scope="function")
def dag_bn(inputs_like_img):
    model = MyBNTestNet()
    return DagTorchHook(model, inputs_like_img)


@pytest.fixture(scope="function")
def dag_conv_group2(inputs_like_img):
    model = MyConvTestNet(2)
    return DagTorchHook(model, inputs_like_img)


@pytest.fixture(scope="function")
def dag_linear(inputs_of_linear):
    model = MyLinearTestNet()
    return DagTorchHook(model, inputs_of_linear)


class TestPruneConv2d:
    def test_prune_input_given_channel_4_delete_123_when_any_pass(self, dag_conv, inputs_like_img):
        input_ori = inputs_like_img.clone()
        input_ori[:, [1, 2, 3], :, :] = 0
        ori_output = dag_conv.network(input_ori)

        assert dag_conv.dag_node_list[0].node.in_channels == 4

        prune_torch = PruneTorch(dag_conv)
        dag_conv._inputs = input_ori[:, [0], :, :]
        prune_torch.prune_by_desc(
            {
                "features": {
                    "input": (1, '-' * 1 + 'x' * 3),
                },
            }
        )
        pruned_output = prune_torch.network(inputs_like_img[:, [0], :, :])

        assert (torch.round(ori_output * 1000) == torch.round(pruned_output * 1000)).sum() == ori_output.numel()
        assert dag_conv.dag_node_list[0].node.in_channels == 1

    def test_prune_output_given_channel_64_delete_32to63_when_any_pass(self, dag_conv, inputs_like_img):
        ori_output = dag_conv.network(inputs_like_img)

        assert dag_conv.get_node_by_name('features').node.out_channels == 64

        prune_torch = PruneTorch(dag_conv)
        prune_torch.prune_by_desc(
            {
                "features": {
                    "output": (32, '-' * 32 + 'x' * 32),
                },
            }
        )
        pruned_output = prune_torch.network(inputs_like_img)

        assert (
            torch.round(ori_output[:, 0:32, :, :] * 1000) == torch.round(pruned_output * 1000)
        ).sum() == pruned_output.numel()
        assert dag_conv.get_node_by_name('features').node.out_channels == 32

    def test_prune_input_with_group_given_channel_4_delete_13_when_any_pass(self, dag_conv_group2, inputs_like_img):
        input_ori = inputs_like_img.clone()
        input_ori[:, [1, 3], :, :] = 0
        ori_output = dag_conv_group2.network(input_ori)

        assert dag_conv_group2.dag_node_list[0].node.in_channels == 4

        prune_torch = PruneTorch(dag_conv_group2)
        dag_conv_group2._inputs = input_ori[:, [0, 1], :, :]
        prune_torch.prune_by_desc(
            {
                "features": {
                    "input": (1, '-' * 1 + 'x' * 1),
                },
            }
        )
        pruned_output = prune_torch.network(inputs_like_img[:, [0, 2], :, :])

        assert (torch.round(ori_output * 1000) == torch.round(pruned_output * 1000)).sum() == ori_output.numel()
        assert dag_conv_group2.dag_node_list[0].node.in_channels == 2

    def test_prune_linear_input_given_channel_12_delete_6to11_when_any_pass(self, dag_linear, inputs_of_linear):
        input_ori = inputs_of_linear.clone()
        input_ori[:, 6:12] = 0
        ori_output = dag_linear.network(input_ori)

        assert dag_linear.dag_node_list[0].node.in_features == 12

        prune_torch = PruneTorch(dag_linear)
        dag_linear._inputs = input_ori[:, 0:6]
        prune_torch.prune_by_desc(
            {
                "features": {
                    "input": (6, '-' * 6 + 'x' * 6),
                },
            }
        )
        pruned_output = prune_torch.network(input_ori[:, 0:6])

        assert (torch.round(ori_output * 1000) == torch.round(pruned_output * 1000)).sum() == ori_output.numel()
        assert dag_linear.dag_node_list[0].node.in_features == 6

    def test_prune_linear_output_given_channel_24_delete_11to24_when_any_pass(self, dag_linear, inputs_of_linear):
        ori_output = dag_linear.network(inputs_of_linear)

        assert dag_linear.dag_node_list[0].node.out_features == 24

        prune_torch = PruneTorch(dag_linear)
        prune_torch.prune_by_desc(
            {
                "features": {
                    "output": (12, '-' * 12 + 'x' * 12),
                },
            }
        )
        pruned_output = prune_torch.network(inputs_of_linear)

        assert (
            torch.round(ori_output[:, 0:12] * 1000) == torch.round(pruned_output * 1000)
        ).sum() == pruned_output.numel()
        assert dag_linear.dag_node_list[0].node.out_features == 12

    def test_prune_bn_given_channel_24_delete_11to24_when_any_pass(self, dag_bn, inputs_like_img):
        ori_output = dag_bn.network(inputs_like_img)

        assert dag_bn.get_node_by_name("batch_normal").node.num_features == 64

        prune_torch = PruneTorch(dag_bn)
        prune_torch.prune_by_desc(
            {
                "features": {
                    "output": (32, '-' * 32 + 'x' * 32),
                },
                "batch_normal": {
                    "input": (32, '-' * 32 + 'x' * 32),
                },
            }
        )
        pruned_output = prune_torch.network(inputs_like_img)

        assert (
            torch.round(ori_output[:, 0:32, :, :] * 1000) == torch.round(pruned_output * 1000)
        ).sum() == pruned_output.numel()
        assert dag_bn.get_node_by_name("batch_normal").node.num_features == 32
