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

import tempfile
from pathlib import Path

import pytest

from msmodelslim.app.auto_tuning.evaluation_service_infra import (
    EvaluateContext,
    EvaluateServiceInfra,
)
from msmodelslim.utils.exception import SchemaValidateError


class TestEvaluateContext:
    """Test suite for EvaluateContext — Pydantic 校验。"""

    # ---------- 正常情形 ----------

    def test_evaluate_context_creates_with_defaults_when_device_and_indices_omitted(self):
        """主路径：仅传必填字段时，device/indices 应使用默认值。"""
        with tempfile.TemporaryDirectory() as tmp:
            ctx = EvaluateContext(evaluate_id="e1", working_dir=Path(tmp))

        assert ctx.evaluate_id == "e1"
        assert ctx.device.value == "npu"  # default
        assert ctx.device_indices is None  # default

    def test_evaluate_context_uses_explicit_device_and_indices_when_provided(self):
        """主路径：显式传 device + device_indices 时应被使用。"""
        with tempfile.TemporaryDirectory() as tmp:
            ctx = EvaluateContext(
                evaluate_id="e1",
                device="cpu",
                device_indices=[0, 1],
                working_dir=Path(tmp),
            )

        assert ctx.device.value == "cpu"
        assert ctx.device_indices == [0, 1]

    # ---------- 边界情形 ----------

    def test_evaluate_context_accepts_empty_device_indices_list_when_caller_passes_empty(self):
        """边界：device_indices=[]（空列表）应被接受（区别于 None）。"""
        with tempfile.TemporaryDirectory() as tmp:
            ctx = EvaluateContext(
                evaluate_id="e1",
                device_indices=[],
                working_dir=Path(tmp),
            )

        assert not ctx.device_indices

    def test_evaluate_context_working_dir_accepts_string_path_when_caller_passes_string(self):
        """边界：working_dir 传字符串路径应自动转 Path。"""
        with tempfile.TemporaryDirectory() as tmp:
            ctx = EvaluateContext(evaluate_id="e1", working_dir=tmp)

        assert isinstance(ctx.working_dir, Path)

    # ---------- 异常情形 ----------

    def test_evaluate_context_raises_schema_validate_error_when_evaluate_id_missing(self):
        """异常：缺少 evaluate_id 应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            EvaluateContext(working_dir=Path("/tmp"))

    def test_evaluate_context_raises_schema_validate_error_when_working_dir_missing(self):
        """异常：缺少 working_dir 应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            EvaluateContext(evaluate_id="e1")


class TestEvaluateServiceInfra:
    """Test suite for EvaluateServiceInfra — ABC 接口契约。"""

    def test_evaluate_service_infra_raises_type_error_when_instantiated_directly(self):
        """异常：直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            EvaluateServiceInfra()  # pylint: disable=abstract-class-instantiated

    def test_subclass_with_full_impl_works_when_all_abstract_methods_implemented(self):
        """边界：完整实现的子类应可实例化。"""

        class _Concrete(EvaluateServiceInfra):
            def evaluate(self, context, evaluate_config, model_path):
                return "ok"

        # 完整实现后不应再抛 TypeError
        instance = _Concrete()
        assert isinstance(instance, EvaluateServiceInfra)
