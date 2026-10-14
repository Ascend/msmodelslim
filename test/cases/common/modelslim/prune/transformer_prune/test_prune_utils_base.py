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

import pytest
import torch
import re  # Required for prune_state_dict_blocks  # noqa: F401
from ascend_utils.common.prune.transformer_prune.prune_utils_base import PruneUtilsBase


# Sample data for testing
FLAT_DICT = {1: 'a', 2: 'b', 3: 'c'}
FLAT_DICT_REVERSED = {'a': '1', 'b': '2', 'c': '3'}

STATE_DICT_BERT = {
    'bert.encoder.layer.0.attention.self.query.weight': torch.randn(768, 768),
    'bert.encoder.layer.0.attention.self.key.weight': torch.randn(768, 768),
    'bert.encoder.layer.1.attention.self.query.weight': torch.randn(768, 768),
    'bert.encoder.layer.1.attention.self.key.weight': torch.randn(768, 768),
    'bert.pooler.dense.weight': torch.randn(768, 768),
}

MODEL_STATE_DICT_BERT = {
    'bert.encoder.layer.0.attention.self.query.weight': torch.randn(512, 768),
    'bert.encoder.layer.0.attention.self.key.weight': torch.randn(512, 768),
    'bert.encoder.layer.0.attention.output.LayerNorm.weight': torch.randn(512),
    'bert.encoder.layer.1.attention.self.query.weight': torch.randn(512, 768),
    'bert.encoder.layer.1.attention.self.key.weight': torch.randn(512, 768),
    'bert.pooler.dense.weight': torch.randn(768, 768),
}


class MockParameter:
    def __init__(self, tensor, name):
        self.tensor = tensor
        self.name = name

    def __repr__(self):
        return f"MockParameter({self.name})"


# Test flip_dict static method
def test_flip_dict_given_dict_when_called_then_returns_reversed():
    result = PruneUtilsBase.flip_dict(FLAT_DICT)
    assert result == FLAT_DICT_REVERSED


# Test prune_bert_intra_block - same shape
def test_prune_bert_intra_block_given_same_shape_when_called_then_no_change():
    state_dict = {
        'weight': torch.randn(768, 768),
    }
    model_state_dict = {
        'weight': torch.randn(768, 768),
    }
    PruneUtilsBase.prune_bert_intra_block(model_state_dict, state_dict)
    assert state_dict['weight'].shape == (768, 768)


# Test prune_bert_intra_block - unsupported dim
def test_prune_bert_intra_block_given_unsupported_dim_then_raise_error():
    state_dict = {
        'tensor': torch.randn(2, 2, 2),
    }
    model_state_dict = {
        'tensor': torch.randn(1, 1, 1),
    }
    with pytest.raises(NotImplementedError):
        PruneUtilsBase.prune_bert_intra_block(model_state_dict, state_dict)


# Test prune_bert_intra_block - convert to parameter
def test_prune_bert_intra_block_given_is_parameter_true_then_converted():
    state_dict = {
        'weight': torch.randn(768, 768),
    }
    model_state_dict = {
        'weight': torch.randn(512, 768),
    }
    PruneUtilsBase.prune_bert_intra_block(model_state_dict, state_dict, is_parameter=True, parameter=MockParameter)
    assert isinstance(state_dict['weight'], MockParameter)


# Test prune_blocks - missing prune_blocks_params
def test_prune_blocks_given_missing_params_when_called_then_raise_exception():
    model_config = {}
    with pytest.raises(Exception) as exc_info:
        PruneUtilsBase().prune_blocks(None, None, model_config)
    assert "prune_blocks failed. prune_blocks_params cannot be None" in str(exc_info.value)


# Test prune_state_dict_blocks - no match
def test_prune_state_dict_blocks_given_no_match_pattern_when_called_then_keep_original():
    params = [{'pattern': r'not_matching\.pattern\.(\d+)\.', 'layer_id_map': {0: 0}}]
    new_state_dict = PruneUtilsBase().prune_state_dict_blocks(STATE_DICT_BERT, params)
    assert len(new_state_dict) == len(STATE_DICT_BERT)


# Test prune_state_dict_blocks - multiple patterns
def test_prune_state_dict_blocks_given_multiple_patterns_when_called_then_all_applied():
    params = [
        {'pattern': r'bert\.encoder\.layer\.(\d+)\.', 'layer_id_map': {0: 0}},
        {'pattern': r'bert\.encoder\.layer\.(\d+)\.attention\.self\.key\.weight', 'layer_id_map': {1: 1}},
    ]
    new_state_dict = PruneUtilsBase().prune_state_dict_blocks(STATE_DICT_BERT, params)
    assert 'bert.encoder.layer.0.attention.self.query.weight' in new_state_dict
    assert 'bert.encoder.layer.1.attention.self.key.weight' in new_state_dict
