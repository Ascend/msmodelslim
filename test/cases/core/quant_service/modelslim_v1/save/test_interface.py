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

import torch
from torch import nn

from msmodelslim.core.quant_service.modelslim_v1.save.interface import (
    AscendV1GlobalModelDtypeInterface,
    AscendV1SaveInterface,
)


class _DtypeAdapter(AscendV1GlobalModelDtypeInterface):
    def get_global_model_torch_dtype(self) -> torch.dtype:
        return torch.bfloat16


class _SaveAdapter(AscendV1SaveInterface):
    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        model._saved_dir = save_directory

    def ascendv1_save_module_preprocess(self, prefix: str, module: nn.Module, model: nn.Module):
        return f"pre_{prefix}", module


class TestAscendV1GlobalModelDtypeInterface:
    """Tests for AscendV1GlobalModelDtypeInterface."""

    def test_get_global_model_torch_dtype_returns_bfloat16_when_stub(self):
        """场景：子类返回全局 dtype。预期：bfloat16。"""
        adapter = _DtypeAdapter()
        assert adapter.get_global_model_torch_dtype() is torch.bfloat16


class TestAscendV1SaveInterface:
    """Tests for AscendV1SaveInterface."""

    def test_ascendv1_save_postprocess_sets_dir_when_called(self):
        """场景：调用 ascendv1_save_postprocess。预期：模块记录目录。"""
        model = nn.Linear(2, 2)
        adapter = _SaveAdapter()
        adapter.ascendv1_save_postprocess(model, "/tmp/out")
        assert model._saved_dir == "/tmp/out"

    def test_ascendv1_save_module_preprocess_returns_prefixed_name_when_called(self):
        """场景：调用 ascendv1_save_module_preprocess。预期：前缀更新。"""
        model = nn.Linear(2, 2)
        module = nn.Linear(2, 2)
        adapter = _SaveAdapter()
        prefix, out_module = adapter.ascendv1_save_module_preprocess("layer", module, model)
        assert prefix == "pre_layer"
        assert out_module is module
