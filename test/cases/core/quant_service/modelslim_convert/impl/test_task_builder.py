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

msmodelslim.core.quant_service.modelslim_convert.impl.task_builder 模块的单元测试
"""

from torch import nn

from msmodelslim.core.convert.catalog import PreprocessResult, TensorCatalog, TensorEntry
from msmodelslim.core.convert.config import ConvertConfig, ConvertRule
from msmodelslim.core.convert.edges import RouteConstraints
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.impl.task_builder import DefaultIRTaskBuilder
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear


class TestDefaultIRTaskBuilder:
    """测试 DefaultIRTaskBuilder 类"""

    def test_build_create_ir_tasks_when_linear_modules_match_rules(self):
        root = nn.Module()
        linear = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={
                "weight": TensorRef("weight", "layers.0.q_proj.weight", "s0", "bf16", (2, 2)),
            },
            source_ir=SourceIR(kind=IRKind.FLOAT),
        )
        layers = nn.Module()
        layer0 = nn.Module()
        layer0.add_module("q_proj", linear)
        layers.add_module("0", layer0)
        root.add_module("layers", layers)

        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="layers.0.q_proj.weight", shard="s0", dtype="bf16", shape=(2, 2)))
        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            convert_rules=[
                ConvertRule(match="layers.*.q_proj", target_ir=IRKind.W8A8_MXFP8),
            ],
        )
        context = ConvertContext(config=config)
        context.preprocess_result = PreprocessResult(catalog=catalog)

        tasks = DefaultIRTaskBuilder().build(context, root, catalog)
        assert len(tasks) == 1
        assert tasks[0].module_path == "layers.0.q_proj"
        assert tasks[0].target_ir == IRKind.W8A8_MXFP8
        assert isinstance(tasks[0].route_constraints, RouteConstraints)
        assert tasks[0].route_constraints.explicit_route == [IRKind.FLOAT, IRKind.W8A8_MXFP8]

    def test_build_set_fp8_to_float_auto_route_when_source_is_fp8_block(self):
        root = nn.Module()
        linear = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={
                "weight": TensorRef("weight", "layers.0.q_proj.weight", "s0", "float8_e4m3fn", (2, 2)),
                "weight_scale_inv": TensorRef(
                    "weight_scale_inv",
                    "layers.0.q_proj.weight_scale_inv",
                    "s0",
                    "bf16",
                    (2, 1),
                ),
            },
            source_ir=SourceIR(kind=IRKind.FP8_BLOCK),
        )
        layers = nn.Module()
        layer0 = nn.Module()
        layer0.add_module("q_proj", linear)
        layers.add_module("0", layer0)
        root.add_module("layers", layers)

        catalog = TensorCatalog()
        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            convert_rules=[
                ConvertRule(match="layers.*.q_proj", target_ir=IRKind.FLOAT, route="auto"),
            ],
        )
        context = ConvertContext(config=config)
        context.preprocess_result = PreprocessResult(catalog=catalog)

        tasks = DefaultIRTaskBuilder().build(context, root, catalog)
        assert tasks[0].route_constraints.explicit_route == [IRKind.FP8_BLOCK, IRKind.FLOAT]
