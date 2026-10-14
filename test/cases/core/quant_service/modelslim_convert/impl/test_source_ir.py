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

msmodelslim.core.quant_service.modelslim_convert.impl.source_ir 模块的单元测试
"""

from msmodelslim.core.convert.config import ConvertConfig, ModuleRule
from msmodelslim.core.convert.types import IRKind, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.impl.source_ir import (
    bind_tensors_for_module,
    infer_source_ir,
)
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear


class TestInferSourceIr:
    """测试 infer_source_ir 函数"""

    def test_infer_source_ir_return_fp8_when_module_rule_declares_fp8(self):
        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={"weight": TensorRef("weight", "layers.0.q_proj.weight", "s0", "bf16", (2, 2))},
        )
        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            module_rules=[
                ModuleRule(match="layers.*.q_proj", source_format="fp8_block", source_ir=IRKind.FP8_BLOCK),
            ],
        )
        result = infer_source_ir(mod, config)
        assert result.kind == IRKind.FP8_BLOCK

    def test_infer_source_ir_return_fp8_when_scale_inv_in_bindings(self):
        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={
                "weight": TensorRef("weight", "layers.0.q_proj.weight", "s0", "bf16", (2, 2)),
                "weight_scale_inv": TensorRef(
                    "weight_scale_inv", "layers.0.q_proj.weight_scale_inv", "s0", "bf16", (1, 1)
                ),
            },
        )
        config = ConvertConfig(model_path="/m", save_path="/o")
        result = infer_source_ir(mod, config)
        assert result.kind == IRKind.FP8_BLOCK

    def test_infer_source_ir_return_float_when_weight_only_binding(self):
        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={"weight": TensorRef("weight", "layers.0.q_proj.weight", "s0", "bf16", (2, 2))},
        )
        config = ConvertConfig(model_path="/m", save_path="/o")
        result = infer_source_ir(mod, config)
        assert result.kind == IRKind.FLOAT


class TestBindTensorsForModule:
    """测试 bind_tensors_for_module 函数"""

    def test_bind_tensors_for_module_resolve_weight_key_when_in_catalog(self):
        rule = ModuleRule(match="layers.*", tensor_map={"weight": "{module}.weight"})
        keys = {"layers.0.q_proj.weight", "layers.1.q_proj.weight"}
        bindings = bind_tensors_for_module("layers.0.q_proj", rule, keys)
        assert "weight" in bindings
        assert bindings["weight"].key == "layers.0.q_proj.weight"
