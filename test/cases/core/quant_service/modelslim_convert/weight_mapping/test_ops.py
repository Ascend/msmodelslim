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

msmodelslim.core.quant_service.modelslim_convert.weight_mapping.ops 模块的单元测试
"""

import pytest

from msmodelslim.core.convert.catalog import TensorCatalog, TensorEntry
from msmodelslim.core.convert.config import WeightMappingRule, WeightOpConfig
from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.ops import apply_preprocess_ops


def _make_catalog(keys: list[str]) -> TensorCatalog:
    catalog = TensorCatalog()
    for key in keys:
        catalog.add(TensorEntry(key=key, shard="s0", dtype="bf16", shape=(4, 4)))
    return catalog


class TestApplyPreprocessOps:
    """测试 apply_preprocess_ops 函数"""

    def test_apply_preprocess_ops_rename_key_when_rename_op_given(self):
        catalog = _make_catalog(["ab_layer.weight"])
        rule = WeightMappingRule(
            id="r1",
            source_patterns=["ab_(.*)"],
            target_patterns=["cd_\\1"],
            ops=[WeightOpConfig(type="rename")],
        )
        result = apply_preprocess_ops(catalog, rule)
        assert "cd_layer.weight" in list(catalog.keys())
        assert result.added_keys

    def test_apply_preprocess_ops_split_gate_up_when_chunk_op_given(self):
        fused_key = "model.layers.0.mlp.experts.gate_up_proj"
        catalog = _make_catalog([fused_key])
        rule = WeightMappingRule(
            id="split",
            source_patterns=[fused_key],
            target_patterns=["gate", "up"],
            ops=[
                WeightOpConfig(
                    type="split_fused_gate_up",
                    params={"split_dim": 1, "projections": ["gate_proj", "up_proj"]},
                ),
            ],
            reversible=True,
        )
        result = apply_preprocess_ops(catalog, rule, num_experts=2)
        assert fused_key not in list(catalog.keys())
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" in list(catalog.keys())
        assert "model.layers.0.mlp.experts.1.up_proj.weight" in list(catalog.keys())
        assert len(result.restore_rules) == 1

    def test_apply_preprocess_ops_raise_error_when_op_unknown(self):
        catalog = _make_catalog(["x"])
        rule = WeightMappingRule(
            id="bad",
            source_patterns=["x"],
            target_patterns=["y"],
            ops=[WeightOpConfig(type="unknown")],
        )
        with pytest.raises(NotImplementedError):
            apply_preprocess_ops(catalog, rule)

    def test_apply_preprocess_ops_raise_error_when_num_experts_missing(self):
        catalog = _make_catalog(["model.layers.0.mlp.experts.gate_up_proj"])
        rule = WeightMappingRule(
            id="split",
            source_patterns=["model.layers.0.mlp.experts.gate_up_proj"],
            target_patterns=["a", "b"],
            ops=[WeightOpConfig(type="split_fused_gate_up", params={})],
        )
        with pytest.raises(ValueError, match="num_experts"):
            apply_preprocess_ops(catalog, rule)
