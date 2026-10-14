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

msmodelslim.core.quant_service.modelslim_convert.impl.preprocess 模块的单元测试
"""

import json
from pathlib import Path

from msmodelslim.core.convert.catalog import TensorCatalog, TensorEntry
from msmodelslim.core.convert.config import ConvertConfig, WeightMappingRule, WeightOpConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.quant_service.modelslim_convert.impl.preprocess import PreprocessExecutor


class TestPreprocessExecutor:
    """测试 PreprocessExecutor 类"""

    def test_run_apply_rules_and_build_dependency_map_when_rules_given(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"num_experts": 4}),
            encoding="utf-8",
        )
        raw = TensorCatalog()
        raw.add(
            TensorEntry(
                key="model.layers.0.mlp.experts.gate_up_proj",
                shard="s0",
                dtype="bf16",
                shape=(4, 8, 16),
            ),
        )
        config = ConvertConfig(
            model_path=str(model_dir),
            save_path=str(tmp_path / "out"),
            preprocess_rules=[
                WeightMappingRule(
                    id="split",
                    source_patterns=["model.layers.0.mlp.experts.gate_up_proj"],
                    target_patterns=["gate", "up"],
                    ops=[
                        WeightOpConfig(
                            type="split_fused_gate_up",
                            params={"split_dim": 1, "projections": ["gate_proj", "up_proj"]},
                        ),
                    ],
                ),
            ],
        )
        context = ConvertContext(config=config)
        result = PreprocessExecutor().run(context, raw, config.preprocess_rules)
        assert "model.layers.0.mlp.experts.gate_up_proj" not in list(result.catalog.keys())
        assert len(result.applied_rules) == 1
