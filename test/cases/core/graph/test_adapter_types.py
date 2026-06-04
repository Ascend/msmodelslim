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

from msmodelslim.core.graph.adapter_types import (
    SUPPORTED_SUBGRAPH_TYPES,
    AdapterConfig,
    FusionConfig,
    MappingConfig,
)


class TestFusionConfig:
    """Tests for FusionConfig validation."""

    def test_fusion_config_accepts_none_type_when_default(self):
        """场景：fusion_type=none。预期：构造成功。"""
        cfg = FusionConfig(fusion_type="none")
        assert cfg.fusion_type == "none"

    def test_fusion_config_accepts_qkv_when_heads_provided(self):
        """场景：qkv 且 heads 齐全。预期：构造成功。"""
        cfg = FusionConfig(fusion_type="qkv", num_attention_heads=8, num_key_value_heads=2)
        assert cfg.num_attention_heads == 8

    def test_fusion_config_raises_value_error_when_qkv_missing_heads(self):
        """场景：qkv 缺少 heads。预期：ValueError。"""
        with pytest.raises(ValueError, match="num_attention_heads"):
            FusionConfig(fusion_type="qkv", num_attention_heads=None, num_key_value_heads=None)

    def test_fusion_config_raises_value_error_when_kv_missing_custom_config(self):
        """场景：kv 无 custom_config。预期：ValueError。"""
        with pytest.raises(ValueError, match="custom_config"):
            FusionConfig(fusion_type="kv", num_attention_heads=8)

    def test_fusion_config_raises_value_error_when_custom_missing_config(self):
        """场景：custom 无 custom_config。预期：ValueError。"""
        with pytest.raises(ValueError, match="custom_config"):
            FusionConfig(fusion_type="custom")

    def test_fusion_config_raises_value_error_when_unsupported_type(self):
        """场景：未知 fusion_type。预期：ValueError。"""
        with pytest.raises(ValueError, match="不支持的融合类型"):
            FusionConfig(fusion_type="unknown_type")


class TestAdapterConfig:
    """Tests for AdapterConfig validation."""

    def test_adapter_config_accepts_valid_subgraph_type_when_mapping_present(self):
        """场景：合法 subgraph_type 与 mapping。预期：构造成功。"""
        for subgraph_type in SUPPORTED_SUBGRAPH_TYPES:
            cfg = AdapterConfig(
                subgraph_type=subgraph_type,
                mapping=MappingConfig(targets=["layer2"], source="layer1"),
            )
            assert cfg.subgraph_type == subgraph_type

    def test_adapter_config_raises_value_error_when_subgraph_type_none(self):
        """场景：subgraph_type 为 None。预期：ValueError。"""
        with pytest.raises(ValueError, match="subgraph_type is required"):
            AdapterConfig(subgraph_type=None, mapping=MappingConfig(targets=["a"]))

    def test_adapter_config_raises_value_error_when_subgraph_type_unsupported(self):
        """场景：不支持的 subgraph_type。预期：ValueError。"""
        with pytest.raises(ValueError, match="不是支持的子图类型"):
            AdapterConfig(
                subgraph_type="invalid-type",
                mapping=MappingConfig(targets=["a"]),
            )


class TestMappingConfig:
    """Tests for MappingConfig."""

    def test_mapping_config_accepts_optional_source_when_none(self):
        """场景：source 为 None（非融合）。预期：构造成功。"""
        cfg = MappingConfig(targets=["layer2"], source=None)
        assert cfg.source is None
        assert cfg.targets == ["layer2"]
