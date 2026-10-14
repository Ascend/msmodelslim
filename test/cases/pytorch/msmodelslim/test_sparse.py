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
# pylint: disable=redefined-outer-name

import torch
import pytest
from torch import nn

from ascend_utils.common.utils import count_parameters
from msmodelslim.pytorch import sparse


@pytest.fixture(scope="function")
def width_model():
    yield nn.Sequential(
        nn.Conv2d(3, 32, 1, 1),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 32),
        nn.Linear(32, 10),
    )


@pytest.fixture(scope="function")
def width_optimizer(width_model):
    yield torch.optim.SGD(width_model.parameters(), lr=0.1)


@pytest.fixture(scope="function")
def depth_model():
    yield nn.Sequential(
        nn.Conv2d(3, 32, 1, 1, bias=False),
        nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.Conv2d(64, 32, 1, 1, bias=False)),
        nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.Conv2d(64, 32, 1, 1, bias=False)),
        nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.Conv2d(64, 32, 1, 1, bias=False)),
        nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.Conv2d(64, 32, 1, 1, bias=False)),
        nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.Conv2d(64, 32, 1, 1, bias=False)),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10, bias=False),
    )


@pytest.fixture(scope="function")
def depth_optimizer(depth_model):
    yield torch.optim.SGD(depth_model.parameters(), lr=0.1)


def test_sparse_model_width_given_valid_when_eval_then_pass(width_model, width_optimizer):
    steps_per_epoch, epochs_each_stage = 2, [2, 2, 1]
    oring_model_params = count_parameters(width_model)  # 10826
    model = sparse.sparse_model_width(
        width_model, width_optimizer, steps_per_epoch=steps_per_epoch, epochs_each_stage=epochs_each_stage
    )
    model.eval()
    model(torch.ones([1, 3, 32, 32]))
    assert count_parameters(model) == oring_model_params

    model.train()
    model(torch.ones([1, 3, 32, 32]))
    assert count_parameters(model) == 794


def test_sparse_model_width_given_valid_when_train_then_pass(width_model, width_optimizer):
    steps_per_epoch, epochs_each_stage = 3, [2, 3, 1]
    oring_model_params = count_parameters(width_model)  # 10826
    model = sparse.sparse_model_width(
        width_model, width_optimizer, steps_per_epoch=steps_per_epoch, epochs_each_stage=epochs_each_stage
    )
    model.train()
    for _ in range(steps_per_epoch * sum(epochs_each_stage)):
        width_optimizer.zero_grad()
        output = model(torch.ones([1, 3, 32, 32]))
        loss = torch.mean(output)
        loss.backward()
        width_optimizer.step()
    assert count_parameters(model) == oring_model_params


def test_sparse_model_width_given_invalid_model_when_any_then_error(width_model, width_optimizer):
    with pytest.raises(TypeError):
        sparse.sparse_model_width([1, 2, 3], width_optimizer, steps_per_epoch=2, epochs_each_stage=[1, 1])


def test_sparse_model_width_given_invalid_optimizer_when_any_then_error(width_model):
    with pytest.raises(TypeError):
        sparse.sparse_model_width(width_model, 42, steps_per_epoch=2, epochs_each_stage=[1, 1])


def test_sparse_model_width_given_invalid_steps_per_epoch_when_any_then_error(width_model, width_optimizer):
    with pytest.raises(TypeError):
        sparse.sparse_model_width(width_model, width_optimizer, steps_per_epoch=2.0, epochs_each_stage=[1, 1])


def test_sparse_model_width_given_invalid_epochs_each_stage_when_any_then_error(width_model, width_optimizer):
    with pytest.raises(TypeError):
        sparse.sparse_model_width(width_model, width_optimizer, steps_per_epoch=2.0, epochs_each_stage=[1])

    with pytest.raises(TypeError):
        sparse.sparse_model_width(width_model, width_optimizer, steps_per_epoch=2.0, epochs_each_stage=[-1, 1])


def test_sparse_model_depth_given_valid_when_eval_then_pass(depth_model, depth_optimizer):
    steps_per_epoch, epochs_each_stage = 2, [2, 2, 1]
    oring_model_params = count_parameters(depth_model)  # 20896
    model = sparse.sparse_model_depth(
        depth_model, depth_optimizer, steps_per_epoch=steps_per_epoch, epochs_each_stage=epochs_each_stage
    )
    model.eval()
    model(torch.ones([1, 3, 32, 32]))
    assert count_parameters(model) == oring_model_params

    model.train()
    model(torch.ones([1, 3, 32, 32]))
    assert count_parameters(model) == 8608


def test_sparse_model_depth_given_valid_when_train_then_pass(depth_model, depth_optimizer):
    steps_per_epoch, epochs_each_stage = 3, [2, 3, 1]
    oring_model_params = count_parameters(depth_model)  # 20896
    model = sparse.sparse_model_depth(
        depth_model, depth_optimizer, steps_per_epoch=steps_per_epoch, epochs_each_stage=epochs_each_stage
    )
    model.train()
    for _ in range(steps_per_epoch * sum(epochs_each_stage)):
        depth_optimizer.zero_grad()
        output = model(torch.ones([1, 3, 24, 24]))
        loss = torch.mean(output)
        loss.backward()
        depth_optimizer.step()
    assert count_parameters(model) == oring_model_params
