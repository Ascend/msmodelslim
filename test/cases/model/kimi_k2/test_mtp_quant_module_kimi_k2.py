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

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.model.kimi_k2 import mtp_quant_module as target
from msmodelslim.model.kimi_k2.mtp_quant_module import (
    DeepseekV3RMSNorm,
    MTPLayer,
    SharedHead,
    get_mtp_layer,
    remove_zero_and_shift,
    wrap_mtp_decoder,
)


def test_remove_zero_and_shift_given_matrix_with_zeros_when_called_then_shift_and_pad_last_column():
    matrix = torch.tensor(
        [
            [1, 2, 0, 3, 4],
            [5, 0, 6, 7, 8],
            [0, 9, 10, 11, 12],
        ]
    )
    result = remove_zero_and_shift(matrix)
    expected = torch.tensor(
        [
            [1, 2, 3, 4, 0],
            [5, 6, 7, 8, 0],
            [9, 10, 11, 12, 0],
        ]
    )
    assert result.shape == matrix.shape
    assert torch.equal(result, expected)


def test_remove_zero_and_shift_given_single_row_with_zero_when_called_then_correct_padding():
    matrix = torch.tensor([[1, 0, 2, 3]])
    result = remove_zero_and_shift(matrix)
    expected = torch.tensor([[1, 2, 3, 0]])
    assert torch.equal(result, expected)


def test_remove_zero_and_shift_given_int_dtype_when_called_then_preserve_dtype():
    matrix = torch.tensor([[1, 0, 2, 3]], dtype=torch.long)
    result = remove_zero_and_shift(matrix)
    assert result.dtype == matrix.dtype


def test_remove_zero_and_shift_given_two_columns_when_called_then_output_same_shape():
    matrix = torch.tensor([[0, 5], [3, 0]])
    result = remove_zero_and_shift(matrix)
    expected = torch.tensor([[5, 0], [3, 0]])
    assert result.shape == matrix.shape
    assert torch.equal(result, expected)


def test_remove_zero_and_shift_given_cpu_tensor_when_called_then_preserve_device():
    matrix = torch.tensor([[1, 0, 2]], device="cpu")
    result = remove_zero_and_shift(matrix)
    assert result.device == matrix.device


def test_deepseek_v3_rms_norm_given_hidden_size_when_initialized_then_weight_is_ones():
    norm = DeepseekV3RMSNorm(64, eps=1e-6)
    assert norm.weight.shape == (64,)
    assert torch.allclose(norm.weight, torch.ones(64))
    assert norm.variance_epsilon == 1e-6


def test_deepseek_v3_rms_norm_given_default_eps_when_initialized_then_default_value_used():
    norm = DeepseekV3RMSNorm(32)
    assert norm.variance_epsilon == 1e-6


def test_deepseek_v3_rms_norm_given_input_when_forward_then_return_same_shape():
    norm = DeepseekV3RMSNorm(32)
    hidden_states = torch.randn(2, 5, 32)
    out = norm(hidden_states)
    assert out.shape == hidden_states.shape


def test_deepseek_v3_rms_norm_given_bfloat16_input_when_forward_then_preserve_input_dtype():
    norm = DeepseekV3RMSNorm(32)
    hidden_states = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    out = norm(hidden_states)
    assert out.dtype == torch.float32
    assert out.shape == hidden_states.shape


def test_deepseek_v3_rms_norm_given_float32_input_when_forward_then_preserve_float32():
    norm = DeepseekV3RMSNorm(16)
    hidden_states = torch.randn(1, 3, 16, dtype=torch.float32)
    out = norm(hidden_states)
    assert out.dtype == torch.float32


def test_shared_head_given_config_when_initialized_then_has_norm_and_linear_head(dummy_config):
    head = SharedHead(dummy_config)
    assert isinstance(head.norm, DeepseekV3RMSNorm)
    assert isinstance(head.head, nn.Linear)
    assert head.head.in_features == dummy_config.hidden_size
    assert head.head.out_features == dummy_config.vocab_size
    assert head.head.bias is None


def test_shared_head_given_hidden_states_when_forward_then_return_logits_with_expected_shape(dummy_config):
    head = SharedHead(dummy_config)
    hidden_states = torch.randn(2, 5, dummy_config.hidden_size)
    logits = head(hidden_states)
    assert logits.shape == (2, 5, dummy_config.vocab_size)


def test_mtp_layer_given_config_when_initialized_then_has_all_components(dummy_config):
    mtp = MTPLayer(dummy_config)
    assert isinstance(mtp.enorm, DeepseekV3RMSNorm)
    assert isinstance(mtp.hnorm, DeepseekV3RMSNorm)
    assert isinstance(mtp.shared_head, SharedHead)
    assert isinstance(mtp.eh_proj, nn.Linear)
    assert isinstance(mtp.embed_tokens, nn.Embedding)
    assert mtp.eh_proj.in_features == dummy_config.hidden_size * 2
    assert mtp.eh_proj.out_features == dummy_config.hidden_size
    assert mtp.eh_proj.bias is None
    assert mtp.embed_tokens.num_embeddings == dummy_config.vocab_size
    assert mtp.embed_tokens.embedding_dim == dummy_config.hidden_size
    assert mtp.embed_tokens.padding_idx == dummy_config.pad_token_id


def test_mtp_layer_given_config_when_initialized_then_rmsnorm_eps_uses_config_value(dummy_config):
    dummy_config.rms_norm_eps = 1e-5
    mtp = MTPLayer(dummy_config)
    assert mtp.enorm.variance_epsilon == 1e-5
    assert mtp.hnorm.variance_epsilon == 1e-5
    assert mtp.shared_head.norm.variance_epsilon == 1e-5


def test_get_mtp_layer_given_safetensor_file_when_called_then_return_mtp_layer(dummy_config, tmp_path):
    safetensor_path = os.path.join(tmp_path, "model-00163-of-000163.safetensors")
    mock_weights = {
        "model.layers.61.enorm.weight": torch.ones(dummy_config.hidden_size),
        "model.layers.61.hnorm.weight": torch.ones(dummy_config.hidden_size),
        "model.layers.61.eh_proj.weight": torch.ones((dummy_config.hidden_size, dummy_config.hidden_size * 2)),
        "model.layers.61.embed_tokens.weight": torch.ones((dummy_config.vocab_size, dummy_config.hidden_size)),
        "model.layers.61.shared_head.head.weight": torch.ones((dummy_config.vocab_size, dummy_config.hidden_size)),
        "model.layers.61.shared_head.norm.weight": torch.ones(dummy_config.hidden_size),
    }

    with (
        patch.object(target, "load_file", return_value=mock_weights),
        patch.object(target, "get_valid_read_path", return_value=safetensor_path),
        patch.object(target, "get_logger"),
    ):
        result = get_mtp_layer(dummy_config, tmp_path)

    assert isinstance(result, MTPLayer)
    assert torch.equal(result.enorm.weight, mock_weights["model.layers.61.enorm.weight"])


def test_get_mtp_layer_given_extra_keys_in_weights_when_called_then_ignored(dummy_config, tmp_path):
    safetensor_path = os.path.join(tmp_path, "model-00163-of-000163.safetensors")
    mock_weights = {
        "model.layers.61.enorm.weight": torch.ones(dummy_config.hidden_size),
        "model.layers.61.hnorm.weight": torch.ones(dummy_config.hidden_size),
        "model.layers.61.eh_proj.weight": torch.ones((dummy_config.hidden_size, dummy_config.hidden_size * 2)),
        "model.layers.61.embed_tokens.weight": torch.ones((dummy_config.vocab_size, dummy_config.hidden_size)),
        "model.layers.61.shared_head.head.weight": torch.ones((dummy_config.vocab_size, dummy_config.hidden_size)),
        "model.layers.61.shared_head.norm.weight": torch.ones(dummy_config.hidden_size),
        "model.layers.61.unexpected.weight": torch.ones(2),
        "model.layers.61.another.weight": torch.ones(2),
    }

    with (
        patch.object(target, "load_file", return_value=mock_weights),
        patch.object(target, "get_valid_read_path", return_value=safetensor_path),
        patch.object(target, "get_logger"),
    ):
        result = get_mtp_layer(dummy_config, tmp_path)

    assert isinstance(result, MTPLayer)


def test_get_mtp_layer_given_no_matching_keys_when_called_then_strict_missing_keys_raised(dummy_config, tmp_path):
    safetensor_path = os.path.join(tmp_path, "model-00163-of-000163.safetensors")
    mock_weights = {"model.layers.61.unrelated.weight": torch.ones(2)}

    with (
        patch.object(target, "load_file", return_value=mock_weights),
        patch.object(target, "get_valid_read_path", return_value=safetensor_path),
        patch.object(target, "get_logger"),
    ):
        with pytest.raises(RuntimeError):
            get_mtp_layer(dummy_config, tmp_path)


def test_wrap_mtp_decoder_given_mtp_decoder_and_mtp_layer_when_called_then_assign_attributes():
    class _FakeDecoder(nn.Module):
        pass

    mtp_decoder = _FakeDecoder()
    mtp_layer = SimpleNamespace(
        enorm=nn.Module(),
        hnorm=nn.Module(),
        shared_head=nn.Module(),
        eh_proj=nn.Module(),
        embed_tokens=nn.Module(),
    )

    with patch.object(target, "get_logger"):
        wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_layer=mtp_layer)

    assert mtp_decoder.enorm is mtp_layer.enorm
    assert mtp_decoder.hnorm is mtp_layer.hnorm
    assert mtp_decoder.shared_head is mtp_layer.shared_head
    assert mtp_decoder.eh_proj is mtp_layer.eh_proj
    assert mtp_decoder.embed_tokens is mtp_layer.embed_tokens


def test_wrap_mtp_decoder_given_dummy_decoder_when_called_then_all_attributes_replaced():
    class _FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enorm = "old_enorm"
            self.hnorm = "old_hnorm"
            self.shared_head = "old_shared_head"
            self.eh_proj = "old_eh_proj"
            self.embed_tokens = "old_embed_tokens"

    mtp_decoder = _FakeDecoder()
    mtp_layer = SimpleNamespace(
        enorm="new_enorm",
        hnorm="new_hnorm",
        shared_head="new_shared_head",
        eh_proj="new_eh_proj",
        embed_tokens="new_embed_tokens",
    )

    with patch.object(target, "get_logger"):
        wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_layer=mtp_layer)

    assert mtp_decoder.enorm == "new_enorm"
    assert mtp_decoder.hnorm == "new_hnorm"
    assert mtp_decoder.shared_head == "new_shared_head"
    assert mtp_decoder.eh_proj == "new_eh_proj"
    assert mtp_decoder.embed_tokens == "new_embed_tokens"
