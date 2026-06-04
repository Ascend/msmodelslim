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
import torch

from msmodelslim.core.context.base import ValidatedDict
from msmodelslim.utils.exception import SchemaValidateError


class CustomInvalidType:
    """非白名单类型，用于拒绝测试。"""

    def __init__(self, name: str = "invalid") -> None:
        self.name = name


class TestValidatedDict:
    """Tests for ValidatedDict whitelist validation."""

    def test_setitem_accepts_whitelist_primitive_when_int_str_dict_list(self):
        """场景：写入白名单基本类型。预期：可正常读写。"""
        d = ValidatedDict()
        d["int"] = 1
        d["str"] = "ok"
        d["list"] = [1, 2]
        d["dict"] = {"k": "v"}
        assert d["int"] == 1
        assert d["str"] == "ok"

    def test_setitem_accepts_cpu_tensor_when_on_cpu(self):
        """场景：写入 CPU tensor。预期：可存储。"""
        d = ValidatedDict()
        t = torch.tensor([1.0, 2.0], device="cpu")
        d["tensor"] = t
        assert torch.allclose(d["tensor"], t)

    def test_setitem_raises_schema_validate_error_when_custom_type(self):
        """场景：写入非白名单自定义类型。预期：SchemaValidateError。"""
        d = ValidatedDict()
        with pytest.raises(SchemaValidateError, match="Unsupported value type"):
            d["bad"] = CustomInvalidType()

    def test_setitem_raises_schema_validate_error_when_cuda_tensor(self):
        """场景：写入非 CPU tensor（若 CUDA 可用则测 device 校验）。"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        d = ValidatedDict()
        with pytest.raises(SchemaValidateError, match="Only cpu tensors"):
            d["gpu"] = torch.tensor([1.0], device="cuda")
