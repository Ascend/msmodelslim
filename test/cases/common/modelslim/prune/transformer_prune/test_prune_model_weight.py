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
# pylint: disable=redefined-outer-name,useless-object-inheritance

import os

import stat  # noqa: F401
from resources.sample_net_prune import TorchPrunedModel
from resources.sample_net_prune import TorchOriModel
from resources.sample_net_prune import MsPrunedModel
from resources.sample_net_prune import MsOriModel

import pytest
import torch
import mindspore

from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    pre_device_target = mindspore.get_context('device_target')
    mindspore.set_context(device_target='CPU')  # NPU will be rather slow
    yield
    mindspore.set_context(device_target=pre_device_target)  # Set back


@pytest.fixture()
def prune_config():
    config = PruneConfig()
    config.set_steps(['prune_blocks', 'prune_bert_intra_block'])
    config.add_blocks_params(r'fc(\d+)', {1: 2})
    yield config


@pytest.fixture()
def torch_pruned_model():
    yield TorchPrunedModel()


@pytest.fixture()
def torch_ori_weight_path():
    weight_file_path = "model_weights.pth"
    torch_ori_model = TorchOriModel()
    torch.save(torch_ori_model.state_dict(), weight_file_path)
    os.chmod(weight_file_path, int("640", 8))
    yield weight_file_path
    if os.path.exists(weight_file_path):
        os.remove(weight_file_path)


@pytest.fixture()
def ms_pruned_model():
    yield MsPrunedModel()


@pytest.fixture()
def ms_ori_weight_path():
    weight_file_path = "model_weights.ckpt"
    ms_ori_model = MsOriModel()
    mindspore.save_checkpoint(ms_ori_model, weight_file_path)
    os.chmod(weight_file_path, int("640", 8))
    yield weight_file_path
    if os.path.exists(weight_file_path):
        os.remove(weight_file_path)


class TestPruneModelWeight(object):
    @pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
    def test_prune_model_weight_given_valid_when_pytorch_then_pass(
        self, torch_ori_weight_path, torch_pruned_model, prune_config
    ):
        prune_model_weight(torch_pruned_model, prune_config, torch_ori_weight_path)

    def test_prune_model_weight_given_valid_when_mindspore_then_pass(
        self, ms_ori_weight_path, ms_pruned_model, prune_config
    ):
        prune_model_weight(ms_pruned_model, prune_config, ms_ori_weight_path)
