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
# pylint: disable=consider-using-from-import,redefined-outer-name,useless-parent-delegation

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn

from msmodelslim.pytorch.ra_compression.ra_tools import RACompressor
from msmodelslim.pytorch.ra_compression.ra_config import RACompressConfig


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_attention_heads
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])


class SequentialModel(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_attention_heads
        self.blocks = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size))


class InvalidModel(nn.Module):
    def __init__(self):
        super().__init__()
        # No config attribute


class NoHiddenSizeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        # No hidden_size attribute


class NoAttentionHeadsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = 768
        # No num_attention_heads attribute


class ZeroAttentionHeadsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = 768
        self.config.num_attention_heads = 0


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def valid_config():
    return RACompressConfig(theta=0.00001, alpha=100)


class TestRACompressor:
    @staticmethod
    def test_init_given_valid_parameters_when_all_checks_pass_then_success(valid_config):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)
        assert compressor.model == model
        assert compressor.cfg == valid_config
        assert compressor.hidden_size == 768
        assert compressor.head_dim == 64

    @staticmethod
    def test_init_given_invalid_model_when_not_nn_module_then_fail(valid_config):
        with pytest.raises(TypeError):
            RACompressor("not_a_model", valid_config)

    @staticmethod
    def test_init_given_invalid_config_when_not_ra_config_then_fail():
        model = SimpleModel()
        with pytest.raises(TypeError):
            RACompressor(model, "invalid_config")

    @staticmethod
    def test_init_given_invalid_model_when_no_config_then_fail(valid_config):
        model = InvalidModel()
        with pytest.raises(ValueError, match="Model does not have attribute `config`"):
            RACompressor(model, valid_config)

    @staticmethod
    def test_init_given_invalid_model_when_no_hidden_size_then_fail(valid_config):
        model = NoHiddenSizeModel()
        with pytest.raises(ValueError, match="Model must have a `config` attribute with a `hidden_size` property"):
            if hasattr(model.config, 'hidden_size'):
                delattr(model.config, 'hidden_size')
            RACompressor(model, valid_config)

    @staticmethod
    def test_init_given_invalid_model_when_no_num_attention_heads_then_fail(valid_config):
        model = NoAttentionHeadsModel()
        with pytest.raises(
            ValueError, match="Model must have a `config` attribute with a `num_attention_heads` property"
        ):
            if hasattr(model.config, 'num_attention_heads'):
                delattr(model.config, 'num_attention_heads')
            RACompressor(model, valid_config)

    @staticmethod
    def test_init_given_invalid_model_when_zero_num_attention_heads_then_fail(valid_config):
        model = ZeroAttentionHeadsModel()
        with pytest.raises(
            ValueError, match="Model must have a `config` attribute with a `num_attention_heads` property"
        ):
            RACompressor(model, valid_config)

    @staticmethod
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.get_wins')
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.RACompressor._get_qk_weight_and_reshape_by_num_heads')
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.RACompressor._get_interleave')
    def test_get_alibi_windows_given_valid_path_when_all_operations_succeed_then_success(
        mock_get_interleave, mock_get_qk_weights, mock_get_wins, valid_config, temp_dir
    ):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        mock_get_qk_weights.return_value = [(torch.randn(768, 12, 64), torch.randn(768, 12, 64))]
        mock_get_interleave.return_value = [0.5] * 12
        mock_get_wins.return_value = [1.0, 2.0, 3.0]

        save_path = os.path.join(temp_dir, "test_windows.pt")
        compressor.get_alibi_windows(save_path)

        assert os.path.exists(save_path)
        loaded_wins = torch.load(save_path)
        assert torch.equal(loaded_wins, torch.tensor([1.0, 2.0, 3.0]))

    @staticmethod
    def test_get_alibi_windows_given_invalid_path_when_path_not_string_then_fail(valid_config):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        with pytest.raises(TypeError):
            compressor.get_alibi_windows(123)

    @staticmethod
    def test_get_attention_mlp_blocks_given_model_with_module_list_when_found_then_return_module_list(valid_config):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        result = compressor._get_attention_mlp_blocks(model)
        assert result is model.layers

    @staticmethod
    def test_get_attention_mlp_blocks_given_model_with_sequential_when_found_then_return_sequential(valid_config):
        model = SequentialModel()
        compressor = RACompressor(model, valid_config)

        result = compressor._get_attention_mlp_blocks(model)
        assert result is model.blocks

    @staticmethod
    def test_split_qkv_weight_to_query_key_given_single_weight_when_qkv_fused_then_return_query_key(valid_config):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        qkv_weight = torch.randn(2304, 768)  # 768*3 = 2304
        weights = [qkv_weight]

        q_weight, k_weight = compressor._split_qkv_weight_to_query_key(weights, 12)
        assert q_weight.shape == (768, 12, 64)
        assert k_weight.shape == (768, 12, 64)

    @staticmethod
    def test_split_qkv_weight_to_query_key_given_single_weight_when_shape_invalid_then_fail(valid_config):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        qkv_weight = torch.randn(2305, 768)  # Not divisible by 3
        weights = [qkv_weight]

        with pytest.raises(ValueError, match="QKV fused weight first dimension must be divisible by 3"):
            compressor._split_qkv_weight_to_query_key(weights, 12)

    @staticmethod
    def test_split_qkv_weight_to_query_key_given_two_weights_when_q_kv_fused_then_return_query_key(valid_config):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        q_weight = torch.randn(768, 768)
        kv_weight = torch.randn(1536, 768)  # 768*2 = 1536
        weights = [q_weight, kv_weight]

        q_result, k_result = compressor._split_qkv_weight_to_query_key(weights, 12)
        assert q_result.shape == (768, 12, 64)
        assert k_result.shape == (768, 12, 64)

    @staticmethod
    def test_split_qkv_weight_to_query_key_given_two_weights_when_kv_shape_invalid_then_fail(valid_config):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        q_weight = torch.randn(768, 768)
        kv_weight = torch.randn(1537, 768)  # Not divisible by 2
        weights = [q_weight, kv_weight]

        with pytest.raises(ValueError, match="KV fused weight first dimension must be divisible by 2"):
            compressor._split_qkv_weight_to_query_key(weights, 12)

    @staticmethod
    def test_split_qkv_weight_to_query_key_given_three_weights_when_q_k_v_separate_then_return_query_key(valid_config):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        q_weight = torch.randn(768, 768)
        k_weight = torch.randn(768, 768)
        v_weight = torch.randn(768, 768)
        weights = [q_weight, k_weight, v_weight]

        q_result, k_result = compressor._split_qkv_weight_to_query_key(weights, 12)
        assert q_result.shape == (768, 12, 64)
        assert k_result.shape == (768, 12, 64)

    @staticmethod
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.RACompressor._get_attention_mlp_blocks')
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.RACompressor._get_qkv_name')
    def test_get_qk_weight_and_reshape_by_num_heads_given_valid_model_when_weights_found_then_return_qk_list(
        mock_get_qkv_name, mock_get_attention_mlp_blocks, valid_config
    ):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        mock_blocks = MagicMock()
        mock_get_attention_mlp_blocks.return_value = mock_blocks
        mock_get_qkv_name.return_value = ["q", "k", "v"]

        q_weight = torch.randn(768, 768)
        k_weight = torch.randn(768, 768)
        v_weight = torch.randn(768, 768)

        mock_blocks.state_dict.return_value = {"0.q.weight": q_weight, "0.k.weight": k_weight, "0.v.weight": v_weight}

        result = compressor._get_qk_weight_and_reshape_by_num_heads()
        assert len(result) == 1
        q_result, k_result = result[0]
        assert q_result.shape == (768, 12, 64)
        assert k_result.shape == (768, 12, 64)

    @staticmethod
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.RACompressor._get_attention_mlp_blocks')
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.RACompressor._get_qkv_name')
    def test_get_qk_weight_and_reshape_by_num_heads_given_invalid_model_when_no_linear_nodes_then_fail(
        mock_get_qkv_name, mock_get_attention_mlp_blocks, valid_config
    ):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        mock_blocks = MagicMock()
        mock_get_attention_mlp_blocks.return_value = mock_blocks
        mock_get_qkv_name.return_value = []  # No linear nodes

        with pytest.raises(ValueError, match="Found no Linear node in model attention block"):
            compressor._get_qk_weight_and_reshape_by_num_heads()

    @staticmethod
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.RACompressor._get_attention_mlp_blocks')
    @patch('msmodelslim.pytorch.ra_compression.ra_tools.RACompressor._get_qkv_name')
    def test_get_qk_weight_and_reshape_by_num_heads_given_invalid_model_when_too_many_linears_then_fail(
        mock_get_qkv_name, mock_get_attention_mlp_blocks, valid_config
    ):
        model = SimpleModel()
        compressor = RACompressor(model, valid_config)

        mock_blocks = MagicMock()
        mock_get_attention_mlp_blocks.return_value = mock_blocks
        mock_get_qkv_name.return_value = ["q", "k", "v", "extra"]  # Too many linear nodes

        with pytest.raises(ValueError, match="Found 4 Linears node in model attention block, should be <= 3"):
            compressor._get_qk_weight_and_reshape_by_num_heads()

    @staticmethod
    def test_get_interleave_given_power_of_2_when_n_is_power_of_2_then_return_correct_list(valid_config):
        model = SimpleModel(num_attention_heads=8)  # 8 is power of 2
        compressor = RACompressor(model, valid_config)

        result = compressor._get_interleave(8)
        assert len(result) == 8
        assert all(isinstance(x, float) for x in result)

    @staticmethod
    def test_get_interleave_given_non_power_of_2_when_n_not_power_of_2_then_return_correct_list(valid_config):
        model = SimpleModel(num_attention_heads=12)  # 12 is not power of 2
        compressor = RACompressor(model, valid_config)

        result = compressor._get_interleave(12)
        assert len(result) == 12
        assert all(isinstance(x, float) for x in result)

    @staticmethod
    def test_get_interleave_given_small_number_when_n_is_small_then_return_correct_list(valid_config):
        model = SimpleModel(num_attention_heads=1)
        compressor = RACompressor(model, valid_config)

        result = compressor._get_interleave(1)
        assert len(result) == 1
        assert all(isinstance(x, float) for x in result)
