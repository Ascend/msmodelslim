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

msmodelslim.core.convert.tasks 模块的单元测试
"""

import pytest
import torch
from torch import nn
from torch.nn import Parameter

from msmodelslim.core.convert.tasks import IRResult, IRTask, PortableTensor
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef


class TestPortableTensor:
    """测试 PortableTensor 类"""

    def test_from_tensor_roundtrip_restore_bf16_when_given(self):
        src = torch.randn(2, 3, dtype=torch.bfloat16)
        packed = PortableTensor.from_tensor(src)
        restored = packed.to_tensor()
        assert restored.shape == src.shape  # 校验 shape 还原
        assert restored.dtype == torch.bfloat16  # 校验 dtype 还原
        assert torch.equal(restored, src)  # 校验数值一致


class TestIRTask:
    """测试 IRTask 类"""

    def test_create_empty_module_return_model_free_linear_when_bindings_given(self):
        task = IRTask(
            module_path="layers.0.q_proj",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.W8A8_MXFP8,
            tensor_bindings={
                "weight": TensorRef("weight", "layers.0.q_proj.weight", "s0", "bf16", (2, 2)),
            },
            inverse_weight_map={"s0": ["layers.0.q_proj.weight"]},
        )
        mod = task.create_empty_module()
        assert mod.full_name == "layers.0.q_proj"
        assert mod.target_ir == IRKind.W8A8_MXFP8


class TestIRResult:
    """测试 IRResult 类"""

    def test_materialize_to_module_copy_state_when_module_payload_exists(self):
        src = nn.Linear(2, 3, bias=False)
        result = IRResult(
            module_path="layers.0.q_proj",
            final_ir=IRKind.FLOAT,
            module=src,
        )
        target = nn.Linear(2, 3, bias=False)
        result.materialize_to_module(target)
        assert target.weight.shape == src.weight.shape

    def test_materialize_to_module_raise_error_when_no_payload(self):
        result = IRResult(module_path="x", final_ir=IRKind.FLOAT)
        with pytest.raises(ValueError, match="no payload"):
            result.materialize_to_module(nn.Linear(2, 2))

    def test_resolve_module_rebuild_float_linear_from_state_dict(self):
        result = IRResult(
            module_path="layers.0.q_proj",
            final_ir=IRKind.FLOAT,
            state_dict={"weight": Parameter(torch.randn(2, 4, dtype=torch.bfloat16))},
        )
        mod = result.resolve_module()
        assert isinstance(mod, nn.Linear)
        assert mod.weight.shape == (2, 4)
        assert mod.weight.dtype == torch.bfloat16

    def test_materialize_to_module_load_state_dict_when_state_dict_payload(self):
        target = nn.Linear(4, 2, bias=False)
        weight = Parameter(torch.randn(2, 4, dtype=torch.bfloat16))
        result = IRResult(
            module_path="layers.0.q_proj",
            final_ir=IRKind.FLOAT,
            state_dict={"weight": weight},
        )
        result.materialize_to_module(target)
        assert target.weight.shape == (2, 4)  # 校验 state_dict 路径写回

    def test_resolve_module_return_module_when_module_payload_exists(self):
        src = nn.Linear(2, 2, bias=False)
        result = IRResult(module_path="x", final_ir=IRKind.FLOAT, module=src)
        assert result.resolve_module() is src  # 校验已有 module 直返

    def test_resolve_module_raise_error_when_final_ir_unsupported(self):
        result = IRResult(
            module_path="x",
            final_ir=IRKind.INT4_PACKED,
            state_dict={"weight": Parameter(torch.randn(2, 2))},
        )
        with pytest.raises(ValueError, match="Cannot rebuild module"):
            result.resolve_module()

    def test_resolve_module_rebuild_mxfp8_float_linear_from_state_dict(self):
        result = IRResult(
            module_path="layers.0.q_proj",
            final_ir=IRKind.W8A8_MXFP8,
            state_dict={"weight": Parameter(torch.randn(2, 4, dtype=torch.bfloat16))},
        )
        mod = result.resolve_module()
        assert isinstance(mod, nn.Linear)
        assert mod.weight.shape == (2, 4)
        assert mod.weight.dtype == torch.bfloat16
