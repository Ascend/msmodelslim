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

msmodelslim.core.quant_service.modelslim_convert.virtual_module 模块的单元测试
"""

from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import (
    ModelFreeLinear,
    PassthroughModule,
    create_model_free_module,
    set_submodule_by_path,
)


class TestPassthroughModule:
    """测试 PassthroughModule 类"""

    def test_named_parameters_return_prefix_when_single_binding_matches_full_name(self):
        mod = PassthroughModule(
            full_name="model.experts.down_proj",
            tensor_bindings={
                "down_proj": TensorRef(
                    "down_proj",
                    "model.experts.down_proj",
                    "s0",
                    "bf16",
                    (2, 2),
                ),
            },
            source_ir=SourceIR(kind=IRKind.FLOAT),
        )
        mod.register_parameter("down_proj", nn.Parameter(torch.ones(2, 2)))
        params = dict(mod.named_parameters(prefix="model.experts.down_proj", recurse=False))
        assert "model.experts.down_proj" in params  # 校验单子量写出名为 prefix


class TestModelFreeLinear:
    """测试 ModelFreeLinear 类"""

    def test_lazy_init_load_tensors_when_direct_binding(self):
        mod = ModelFreeLinear(
            full_name="layers.0.q_proj",
            tensor_bindings={
                "weight": TensorRef("weight", "layers.0.q_proj.weight", "s0", "bf16", (2, 2)),
            },
            source_ir=SourceIR(kind=IRKind.FLOAT),
        )
        reader = MagicMock()
        reader.load_tensors.return_value = {
            "layers.0.q_proj.weight": torch.ones(2, 2),
        }
        mod.lazy_init(reader, device="cpu")
        assert mod.lazy_initialized
        assert mod.weight.shape == (2, 2)


class TestCreateModelFreeModule:
    """测试 create_model_free_module 工厂函数"""

    def test_create_model_free_module_return_passthrough_when_norm_kind(self):
        mod = create_model_free_module(
            "norm",
            {},
            SourceIR(kind=IRKind.FLOAT),
            IRKind.FLOAT,
            module_kind="norm",
        )
        assert isinstance(mod, PassthroughModule)


class TestSetSubmoduleByPath:
    """测试 set_submodule_by_path 函数"""

    def test_set_submodule_by_path_create_intermediate_nodes_when_missing(self):
        root = nn.Module()
        child = ModelFreeLinear("a.b.c", source_ir=SourceIR(kind=IRKind.FLOAT))
        set_submodule_by_path(root, "a.b.c", child)
        assert isinstance(root.get_submodule("a.b.c"), ModelFreeLinear)
