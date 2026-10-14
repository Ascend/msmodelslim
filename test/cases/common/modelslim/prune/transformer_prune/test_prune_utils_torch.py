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
# pylint: disable=consider-using-dict-items

import pytest
import torch
import os
import shutil  # noqa: F401
from unittest.mock import MagicMock, patch
from ascend_utils.common.security.pytorch import safe_torch_load  # noqa: F401
from ascend_utils.common.prune.transformer_prune.prune_utils_torch import PruneUtilsTorch, QKV_NUMS  # noqa: F401

# Mock data paths
FAKE_WEIGHT_PATH = "./fake_weight.pth"
FAKE_INVALID_PATH = "./invalid_weight.pth"

# Prepare mock state dicts
STATE_DICT = {
    "bert.encoder.layer.0.attention.self.query.weight": torch.randn(768, 768),
    "bert.encoder.layer.0.attention.self.key.weight": torch.randn(768, 768),
    "bert.encoder.layer.1.attention.self.query.weight": torch.randn(768, 768),
    "bert.encoder.layer.1.attention.self.key.weight": torch.randn(768, 768),
    "bert.pooler.dense.weight": torch.randn(768, 768),
}

MODEL_STATE_DICT = {
    "bert.encoder.layer.0.attention.self.query.weight": torch.randn(512, 768),
    "bert.encoder.layer.0.attention.self.key.weight": torch.randn(512, 768),
    "bert.encoder.layer.0.attention.output.LayerNorm.weight": torch.randn(512),
    "bert.encoder.layer.1.attention.self.query.weight": torch.randn(512, 768),
    "bert.encoder.layer.1.attention.self.key.weight": torch.randn(512, 768),
    "bert.pooler.dense.weight": torch.randn(768, 768),
}


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: create fake weight file
    torch.save({"state_dict": STATE_DICT}, FAKE_WEIGHT_PATH)

    yield

    # Teardown: remove fake files
    if os.path.exists(FAKE_WEIGHT_PATH):
        os.remove(FAKE_WEIGHT_PATH)


def test_find_linear_diff_dim_given_2d_weights_when_shapes_mismatch_dim0_then_return_0():
    w1 = torch.randn(768, 768)
    w2 = torch.randn(512, 768)
    diff_axis = PruneUtilsTorch.find_linear_diff_dim(w1, w2)
    assert diff_axis == 0


def test_find_linear_diff_dim_given_2d_weights_when_shapes_mismatch_dim1_then_return_1():
    w1 = torch.randn(768, 768)
    w2 = torch.randn(768, 512)
    diff_axis = PruneUtilsTorch.find_linear_diff_dim(w1, w2)
    assert diff_axis == 1


def test_find_linear_diff_dim_given_invalid_dims_when_called_then_raise_exception():
    w1 = torch.randn(3, 3, 3)
    w2 = torch.randn(1, 1, 1)
    with pytest.raises(Exception):
        PruneUtilsTorch.find_linear_diff_dim(w1, w2)


def test_get_state_dict_given_valid_path_when_file_exists_then_returns_dict():
    ckpt = PruneUtilsTorch.get_state_dict(FAKE_WEIGHT_PATH)
    assert "bert.encoder.layer.0.attention.self.query.weight" in ckpt


def test_get_state_dict_given_invalid_path_when_file_not_exists_then_raise_error():
    with pytest.raises(FileNotFoundError):
        PruneUtilsTorch.get_state_dict(FAKE_INVALID_PATH)


@patch("ascend_utils.common.security.pytorch.torch.load")
def test_get_state_dict_given_invalid_checkpoint_structure_when_called_then_returns_model(mock_load):
    mock_load.return_value = {"model": STATE_DICT}
    ckpt = PruneUtilsTorch.get_state_dict(FAKE_WEIGHT_PATH)
    assert "bert.encoder.layer.0.attention.self.query.weight" in ckpt


def test_prune_bert_intra_block_torch_given_model_and_state_dict_when_called_then_returns_pruned_dict():
    model = MagicMock()
    model.state_dict.return_value = MODEL_STATE_DICT
    pruner = PruneUtilsTorch()
    pruned_dict = pruner.prune_bert_intra_block_torch(model, STATE_DICT.copy(), {})
    for k in pruned_dict:
        assert pruned_dict[k].shape == model.state_dict()[k].shape


def test_prune_vit_intra_block_given_missing_qkv_keys_when_called_then_raise_exception():
    model = MagicMock()
    model.state_dict.return_value = MODEL_STATE_DICT
    pruner = PruneUtilsTorch()
    with pytest.raises(Exception):
        pruner.prune_vit_intra_block(model, STATE_DICT.copy(), {})


def test_prune_vit_intra_block_given_unsupported_dim_when_called_then_raise_exception():
    model = MagicMock()
    model.state_dict.return_value = {**MODEL_STATE_DICT, "extra.tensor": torch.randn(2, 2, 2)}
    model_config = {"qkv_keys": ["query", "key"]}
    pruner = PruneUtilsTorch()
    with pytest.raises(NotImplementedError):
        pruner.prune_vit_intra_block(model, {**STATE_DICT, "extra.tensor": torch.randn(3, 3, 3)}, model_config)


def test_prune_combined_qkv_weight_given_invalid_axis_when_called_then_raise_exception():
    name = "bert.encoder.layer.0.attention.self.query.weight"
    model_weight = torch.randn(768, 768)
    st_weight = torch.randn(768, 768)
    pruner = PruneUtilsTorch()
    with pytest.raises(Exception):
        pruner.prune_combined_qkv_weight(name, {}, model_weight, st_weight)


def test_prune_combined_qkv_weight_given_unsupported_dim_when_called_then_raise_exception():
    name = "bert.encoder.layer.0.attention.self.query.weight"
    model_weight = torch.randn(2, 2, 2)
    st_weight = torch.randn(3, 3, 3)
    pruner = PruneUtilsTorch()
    with pytest.raises(NotImplementedError):
        pruner.prune_combined_qkv_weight(name, {}, model_weight, st_weight)
