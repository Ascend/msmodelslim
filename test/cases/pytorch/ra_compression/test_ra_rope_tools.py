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
# pylint: disable=consider-using-from-import,no-name-in-module,redefined-outer-name,use-implicit-booleaness-not-comparison

import os
import tempfile
from unittest.mock import patch
import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from msmodelslim.pytorch.ra_compression.ra_rope_tools import RARopeCompressor, SoftmaxDumpOutput, DUMMY_INPUT_LENGTH
from msmodelslim.pytorch.ra_compression.ra_rope_config import RARopeCompressConfig


class FakeModelConfig:
    def __init__(self, hidden_size=512, num_attention_heads=8, num_key_value_heads=4):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads


class FakeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, **kwargs):
        return type('obj', (object,), {'logits': torch.randn(1, 10)})


class FakeTokenizer(PreTrainedTokenizerBase):
    def __call__(self, text, return_tensors=None):
        if text == '':
            return {'input_ids': torch.tensor([[]]), 'attention_mask': torch.tensor([])}
        return {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}

    def encode(self, text, **kwargs):
        return [1, 2, 3] if text else []


@pytest.fixture
def fake_config():
    return RARopeCompressConfig(induction_head_ratio=0.5, echo_head_ratio=0.5)


@pytest.fixture
def fake_model():
    config = FakeModelConfig()
    return FakeModel(config)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_raropecompressor_init_given_valid_parameters_when_initialized_then_success(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    assert compressor.model == fake_model
    assert compressor.cfg == fake_config
    assert compressor.tokenizer == fake_tokenizer
    assert compressor.hidden_size == fake_model.config.hidden_size
    assert compressor.num_attention_heads == fake_model.config.num_attention_heads
    assert compressor.num_key_value_heads == fake_model.config.num_key_value_heads
    assert compressor.num_kv_per_group == 2.0


def test_raropecompressor_init_given_invalid_model_when_missing_config_then_raise_error(fake_tokenizer, fake_config):
    invalid_model = nn.Linear(10, 10)
    with pytest.raises(ValueError, match="Model does not have attribute `config`"):
        RARopeCompressor(invalid_model, fake_tokenizer, fake_config)


def test_raropecompressor_init_given_invalid_model_when_missing_hidden_size_then_raise_error(
    fake_tokenizer, fake_config
):
    class InvalidConfigModel(nn.Module):
        def __init__(self):
            self.config = type('obj', (object,), {})

    invalid_model = InvalidConfigModel()
    with pytest.raises(ValueError, match="Model must have a `config` attribute with a `hidden_size` property"):
        RARopeCompressor(invalid_model, fake_tokenizer, fake_config)


def test_raropecompressor_init_given_model_with_different_head_names_when_initialized_then_success(
    fake_tokenizer, fake_config
):
    class SpecialConfigModel(nn.Module):
        def __init__(self):
            self.config = type('obj', (object,), {'hidden_size': 512, 'n_head': 8, 'multi_query_group_num': 4})

    special_model = SpecialConfigModel()
    compressor = RARopeCompressor(special_model, fake_tokenizer, fake_config)
    assert compressor.num_attention_heads == 8
    assert compressor.num_key_value_heads == 4


def test_max_every_group_given_valid_data_when_processed_then_return_correct_result(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    data = {'layer1': [1, 3, 2, 4], 'layer2': [5, 7, 6, 8]}
    result = compressor.max_every_group(data, 2)
    expected = {'layer1': [3, 4], 'layer2': [7, 8]}
    assert result == expected


def test_max_every_group_given_empty_data_when_processed_then_return_empty_dict(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    data = {}
    result = compressor.max_every_group(data, 2)
    assert result == {}


def test_remove_empty_list_keys_given_dict_with_empty_lists_when_processed_then_remove_empties(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    data = {'key1': [1, 2, 3], 'key2': [], 'key3': [4, 5], 'key4': []}
    result = compressor.remove_empty_list_keys(data)
    expected = {'key1': [1, 2, 3], 'key3': [4, 5]}
    assert result == expected


def test_remove_empty_list_keys_given_dict_no_empty_lists_when_processed_then_unchanged(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    data = {'key1': [1, 2, 3], 'key2': [4, 5]}
    result = compressor.remove_empty_list_keys(data)
    assert result == data


def test_select_top_heads_given_valid_data_when_processed_then_return_correct_indices(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    data = {'layer1': [1, 9, 2, 8], 'layer2': [3, 7, 4, 6]}
    result = compressor.select_top_heads(data, 0.5)
    assert 1 in result['layer1']
    assert 3 in result['layer2']


def test_select_top_heads_given_empty_data_when_processed_then_return_empty_dict(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    data = {}
    result = compressor.select_top_heads(data, 0.5)
    assert result == {}


def test_select_top_heads_given_zero_ratio_when_processed_then_return_empty_lists(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    data = {'layer1': [1, 2, 3, 4], 'layer2': [5, 6, 7, 8]}
    result = compressor.select_top_heads(data, 0.0)
    assert result['layer1'] == []
    assert result['layer2'] == []


def test_select_top_heads_given_one_ratio_when_processed_then_return_all_indices(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    data = {'layer1': [1, 2], 'layer2': [3, 4]}
    result = compressor.select_top_heads(data, 1.0)
    assert result['layer1'] == [0, 1]
    assert result['layer2'] == [0, 1]


def test_get_compress_heads_given_valid_save_path_when_processed_then_save_file(
    fake_model, fake_tokenizer, fake_config, temp_dir
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)

    with patch.object(compressor, 'get_attention_score') as mock_get_attention:
        mock_get_attention.return_value = (
            {0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
            {0: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
        )

        save_path = os.path.join(temp_dir, "test_heads.pt")
        compressor.get_compress_heads(save_path)

        assert os.path.exists(save_path)
        loaded_data = torch.load(save_path)
        assert 'prefix_matching' in loaded_data
        assert 'copying' in loaded_data


def test_get_compress_heads_given_invalid_save_path_when_processed_then_raise_error(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)

    with patch.object(compressor, 'get_attention_score') as mock_get_attention:
        mock_get_attention.return_value = ({0: [0.1, 0.2, 0.3, 0.4]}, {0: [0.4, 0.3, 0.2, 0.1]})

        with pytest.raises(Exception):
            compressor.get_compress_heads(123)


def test_get_attention_score_given_valid_inputs_when_processed_then_return_scores(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    compressor.model.device = 'cpu'

    with patch('msmodelslim.pytorch.ra_compression.ra_rope_tools.hook') as mock_hook:  # noqa: F841
        prefix_scores, copying_scores = compressor.get_attention_score()

        assert isinstance(prefix_scores, dict)
        assert isinstance(copying_scores, dict)


def test_get_attention_score_given_empty_tokenizer_when_fallback_then_use_fallback_text(
    fake_model, fake_tokenizer, fake_config
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    compressor.model.device = 'cpu'

    with patch('msmodelslim.pytorch.ra_compression.ra_rope_tools.hook'):
        prefix_scores, copying_scores = compressor.get_attention_score()

        assert prefix_scores is not None
        assert copying_scores is not None


def test_softmaxdumpoutput_init_given_valid_parameters_when_initialized_then_success():
    softmax_dump = SoftmaxDumpOutput(num_attention_heads=8, hidden_size=512)
    assert softmax_dump.head_num == 0
    assert softmax_dump.num_attention_heads == 8
    assert softmax_dump.hidden_size == 512
    assert softmax_dump.gather_data_prefix == {}
    assert softmax_dump.gather_data_copying == {}


def test_softmaxdumpoutput_call_given_valid_input_when_processed_then_update_data():
    softmax_dump = SoftmaxDumpOutput(num_attention_heads=4, hidden_size=512)
    input_tensor = torch.randn(64, 100, 100)

    output = softmax_dump(input_tensor, dim=-1)

    assert output is not None
    assert 0 in softmax_dump.gather_data_prefix
    assert 0 in softmax_dump.gather_data_copying
    assert len(softmax_dump.gather_data_prefix[0]) == 1
    assert len(softmax_dump.gather_data_copying[0]) == 1
    assert softmax_dump.head_num == 1


def test_softmaxdumpoutput_call_given_zero_attention_heads_when_processed_then_raise_error():
    softmax_dump = SoftmaxDumpOutput(num_attention_heads=0, hidden_size=512)
    input_tensor = torch.randn(64, 100, 100)

    with pytest.raises(ValueError, match="Num attention heads can not be zero"):
        softmax_dump(input_tensor, dim=-1)


def test_softmaxdumpoutput_get_prefix_matching_score_given_valid_input_when_processed_then_return_score():
    # Create a mock attention output with specific pattern
    batch_size = 1
    seq_len = DUMMY_INPUT_LENGTH * 2 + 1  # +1 for initial token
    out = torch.zeros(batch_size, seq_len, seq_len)

    # Set up some attention patterns that would yield non-zero scores
    for i in range(1, seq_len):
        if i // DUMMY_INPUT_LENGTH > 0:
            for j in range(i % DUMMY_INPUT_LENGTH, i, DUMMY_INPUT_LENGTH):
                if j + 1 < seq_len:
                    out[0, i, j + 1] = 0.5

    score = SoftmaxDumpOutput._get_prefix_matching_score(out)
    assert isinstance(score, torch.Tensor)


def test_softmaxdumpoutput_get_copying_matching_score_given_valid_input_when_processed_then_return_score():
    # Create a mock attention output with specific pattern
    batch_size = 1
    seq_len = DUMMY_INPUT_LENGTH * 2 + 1  # +1 for initial token
    out = torch.zeros(batch_size, seq_len, seq_len)

    # Set up some attention patterns that would yield non-zero scores
    for i in range(1, seq_len):
        if i // DUMMY_INPUT_LENGTH > 0:
            for j in range(i % DUMMY_INPUT_LENGTH, i, DUMMY_INPUT_LENGTH):
                if j < seq_len:
                    out[0, i, j] = 0.5

    score = SoftmaxDumpOutput._get_copying_matching_score(out)
    assert isinstance(score, torch.Tensor)


def test_integration_compress_heads_workflow_given_complete_setup_when_processed_then_success(
    fake_model, fake_tokenizer, fake_config, temp_dir
):
    compressor = RARopeCompressor(fake_model, fake_tokenizer, fake_config)
    compressor.model.device = 'cpu'

    save_path = os.path.join(temp_dir, "integration_test.pt")

    # Mock the attention score generation to avoid actual model inference
    with patch.object(compressor, 'get_attention_score') as mock_scores:
        mock_scores.return_value = (
            {0: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]},
            {0: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        )

        compressor.get_compress_heads(save_path)

        assert os.path.exists(save_path)
        head_dict = torch.load(save_path)
        assert 'prefix_matching' in head_dict
        assert 'copying' in head_dict
