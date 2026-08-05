#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for core.graph.subgraph_builder."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from torch import nn

from unittest.mock import patch

from msmodelslim.core.graph.adapter_types import AdapterConfig, FusionConfig, MappingConfig
from msmodelslim.core.graph.subgraph_builder import build_subgraph_from_adapter


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)
        self.up = nn.Linear(4, 8)
        self.down = nn.Linear(8, 4)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = TinyBlock()
        self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)


class TestBuildSubgraphFromAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyModel()

    def test_missing_ov_path_returns_none_without_raise(self):
        cfg = AdapterConfig(
            subgraph_type="ov",
            mapping=MappingConfig(
                source="block0.missing_v",
                targets=["block0.o_proj"],
            ),
        )
        built = build_subgraph_from_adapter(self.model, cfg)
        self.assertIsNone(built)

    def test_standard_ov_builds(self):
        cfg = AdapterConfig(
            subgraph_type="ov",
            mapping=MappingConfig(
                source="block0.v_proj",
                targets=["block0.o_proj"],
            ),
        )
        built = build_subgraph_from_adapter(self.model, cfg)
        self.assertIsNotNone(built)
        self.assertEqual(built.linear_names, ["block0.o_proj"])

    def test_non_fusion_missing_target_returns_none(self):
        cfg = AdapterConfig(
            subgraph_type="norm-linear",
            mapping=MappingConfig(source=None, targets=["block0.no_such"]),
        )
        built = build_subgraph_from_adapter(self.model, cfg)
        self.assertIsNone(built)

    def test_ov_missing_heads_returns_none(self):
        model = TinyModel()
        model.config = SimpleNamespace()  # no head fields
        cfg = AdapterConfig(
            subgraph_type="ov",
            mapping=MappingConfig(
                source="block0.v_proj",
                targets=["block0.o_proj"],
            ),
        )
        built = build_subgraph_from_adapter(model, cfg)
        self.assertIsNone(built)

    def test_kv_fusion_missing_custom_keys_returns_none(self):
        cfg = AdapterConfig(
            subgraph_type="ov",
            mapping=MappingConfig(
                source="block0.v_proj",
                targets=["block0.o_proj"],
            ),
            fusion=FusionConfig(
                fusion_type="kv",
                num_attention_heads=4,
                custom_config={"v_head_dim": 128},
            ),
        )
        with patch("msmodelslim.processor.anti_outlier.common.VirtualVModuleFromKVFused") as mock_v:
            built = build_subgraph_from_adapter(self.model, cfg)
        self.assertIsNone(built)
        mock_v.assert_not_called()


if __name__ == "__main__":
    unittest.main()
