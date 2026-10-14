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

from unittest.mock import Mock

import pytest

from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.model.wan2_2.base_model_adapter import Wan2_2BaseModelAdapter
from msmodelslim.utils.exception import SchemaValidateError


def _make_base_adapter():
    adapter = Wan2_2BaseModelAdapter.__new__(Wan2_2BaseModelAdapter)
    adapter.model_type = "wan2_2"
    adapter.model_path = "/tmp/fake"
    adapter._expert_adapters = {}
    adapter.low_noise_model = None
    adapter.high_noise_model = None
    adapter.transformer = None
    return adapter


class TestWan2_2BaseModelAdapter:
    # ===== handle_dataset =====

    @pytest.fixture
    def adapter(self):
        return _make_base_adapter()

    def test_handle_dataset_returns_empty_list_when_dataset_is_none(self, adapter):
        result = adapter.handle_dataset(None)
        assert result == []

    def test_handle_dataset_returns_validated_list_when_dataset_is_single_vlm_calib_sample(self, adapter):
        sample = VlmCalibSample(text="hello")
        result = adapter.handle_dataset(sample)
        assert result == [sample]

    def test_handle_dataset_returns_validated_list_when_dataset_is_vlm_calib_sample_list(self, adapter):
        samples = [VlmCalibSample(text="a"), VlmCalibSample(text="b")]
        result = adapter.handle_dataset(samples)
        assert result == samples

    def test_handle_dataset_passes_through_when_dataset_is_tensor_data_list(self, adapter):
        tensor_data = [[1, 2, 3], [4, 5, 6]]
        result = adapter.handle_dataset(tensor_data)
        assert result == tensor_data

    def test_handle_dataset_passes_through_when_dataset_is_dict_item_list(self, adapter):
        dict_data = [{"key": "val1"}, {"key": "val2"}]
        result = adapter.handle_dataset(dict_data)
        assert result == dict_data

    def test_handle_dataset_returns_empty_list_when_dataset_is_empty_list(self, adapter):
        result = adapter.handle_dataset([])
        assert result == []

    def test_handle_dataset_passes_through_when_dataset_is_mock_object_list(self, adapter):
        mock_data = [Mock(), Mock()]
        result = adapter.handle_dataset(mock_data)
        assert result == mock_data

    def test_handle_dataset_calls_validate_calib_samples_when_dataset_is_vlm_calib_sample_list(self, adapter):
        validated = [VlmCalibSample(text="ok")]
        adapter.validate_calib_samples = Mock(return_value=validated)
        samples = [VlmCalibSample(text="ok")]
        result = adapter.handle_dataset(samples)
        adapter.validate_calib_samples.assert_called_once_with(samples)
        assert result == validated

    def test_handle_dataset_raises_schema_validate_error_when_dataset_is_str(self, adapter):
        with pytest.raises(SchemaValidateError, match="handle_dataset expects dataset to be a list, got str"):
            adapter.handle_dataset("not_a_list")

    def test_handle_dataset_raises_schema_validate_error_when_dataset_is_int(self, adapter):
        with pytest.raises(SchemaValidateError, match="handle_dataset expects dataset to be a list, got int"):
            adapter.handle_dataset(42)

    def test_handle_dataset_raises_schema_validate_error_when_dataset_is_dict(self, adapter):
        with pytest.raises(SchemaValidateError, match="handle_dataset expects dataset to be a list, got dict"):
            adapter.handle_dataset({"key": "val"})

    # ===== get_adapter_config_for_subgraph =====

    def test_get_adapter_config_for_subgraph_returns_empty_list_when_num_layers_is_zero(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(0)
        assert result == []

    def test_get_adapter_config_for_subgraph_returns_seven_configs_when_num_layers_is_one(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        assert len(result) == 7

    def test_get_adapter_config_for_subgraph_returns_fourteen_configs_when_num_layers_is_two(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(2)
        assert len(result) == 14

    def test_get_adapter_config_for_subgraph_returns_adapter_config_instances_when_normal_input(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        for cfg in result:
            assert isinstance(cfg, AdapterConfig)

    def test_get_adapter_config_for_subgraph_sets_norm_linear_when_all_configs(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        for cfg in result:
            assert cfg.subgraph_type == "norm-linear"

    def test_get_adapter_config_for_subgraph_includes_self_attn_qkv_targets_when_layer_zero(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        qkv_config = result[0]
        assert isinstance(qkv_config.mapping, MappingConfig)
        assert qkv_config.mapping.targets == [
            "blocks.0.self_attn.q",
            "blocks.0.self_attn.k",
            "blocks.0.self_attn.v",
        ]

    def test_get_adapter_config_for_subgraph_includes_cross_attn_q_target_when_layer_zero(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        cross_q_config = result[1]
        assert cross_q_config.mapping.targets == ["blocks.0.cross_attn.q"]

    def test_get_adapter_config_for_subgraph_includes_cross_attn_kv_targets_when_layer_zero(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        cross_kv_config = result[2]
        assert cross_kv_config.mapping.targets == ["blocks.0.cross_attn.k", "blocks.0.cross_attn.v"]

    def test_get_adapter_config_for_subgraph_includes_self_attn_o_target_when_layer_zero(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        o_self_config = result[3]
        assert o_self_config.mapping.targets == ["blocks.0.self_attn.o"]

    def test_get_adapter_config_for_subgraph_includes_cross_attn_o_target_when_layer_zero(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        o_cross_config = result[4]
        assert o_cross_config.mapping.targets == ["blocks.0.cross_attn.o"]

    def test_get_adapter_config_for_subgraph_includes_ffn_up_target_when_layer_zero(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        up_config = result[5]
        assert up_config.mapping.targets == ["blocks.0.ffn.0"]

    def test_get_adapter_config_for_subgraph_includes_ffn_down_target_when_layer_zero(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        down_config = result[6]
        assert down_config.mapping.targets == ["blocks.0.ffn.2"]

    def test_get_adapter_config_for_subgraph_uses_layer_idx_one_when_second_layer(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(2)
        layer1_qkv_config = result[7]
        assert layer1_qkv_config.mapping.targets == [
            "blocks.1.self_attn.q",
            "blocks.1.self_attn.k",
            "blocks.1.self_attn.v",
        ]

    def test_get_adapter_config_for_subgraph_sets_source_none_when_all_mappings(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        for cfg in result:
            assert cfg.mapping.source is None

    def test_get_adapter_config_for_subgraph_sets_fusion_none_when_all_configs(self, adapter):
        result = adapter.get_adapter_config_for_subgraph(1)
        for cfg in result:
            assert cfg.fusion is None
