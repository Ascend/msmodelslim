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

import json
import os
import shutil
import tempfile

import pytest
import torch

from .base import FakeLlamaModelAdapter, invoke_test, is_npu_available


def _check_optional_quarot_export(tmp_dir: str) -> None:
    """检查 quant_model_description.json 中 optional 字段及 optional 下 quarot 的 safetensors 存在性。"""
    quant_desc_path = os.path.join(tmp_dir, "quant_model_description.json")
    assert os.path.exists(quant_desc_path), "quant_model_description.json should exist"

    with open(quant_desc_path, "r", encoding="utf-8") as f:
        desc = json.load(f)

    assert "optional" in desc, "quant_model_description.json should contain 'optional' field"
    optional = desc["optional"]
    assert isinstance(optional, dict), "'optional' should be a dict"

    assert "quarot" in optional, "'optional' should contain 'quarot' scope"
    quarot_scope = optional["quarot"]
    assert isinstance(quarot_scope, dict), "'optional.quarot' should be a dict"
    assert "rotation_map" in quarot_scope, "'optional.quarot' should contain 'rotation_map'"
    rotation_map = quarot_scope["rotation_map"]
    assert "global_rotation" in rotation_map, "'optional.quarot.rotation_map' should contain 'global_rotation'"

    relative_safetensors_path = rotation_map["global_rotation"]
    safetensors_abs_path = os.path.join(tmp_dir, relative_safetensors_path)
    assert os.path.exists(safetensors_abs_path), (
        f"optional quarot safetensors should exist at {safetensors_abs_path} (relative: {relative_safetensors_path})"
    )
    assert safetensors_abs_path.endswith(".safetensors"), (
        f"optional.quarot.global_rotation should point to a .safetensors file, got {relative_safetensors_path}"
    )


def _check_optional_dir_not_exists(tmp_dir: str) -> None:
    """检查在不导出 optional 信息时不会创建 optional 目录。"""
    optional_dir = os.path.join(tmp_dir, "optional")
    assert not os.path.exists(optional_dir), "optional directory should not be created when no optional infos"


@pytest.mark.parametrize(
    "test_device, test_dtype",
    [
        pytest.param("cpu", torch.float32),
        pytest.param(
            "npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
        ),
        pytest.param(
            "npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
        ),
    ],
)
@pytest.mark.smoke
def test_quarot_only_process(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行QuaRot量化测试
        model_adapter = invoke_test("quarot_only.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 测试伪量化
        tokenizer = model_adapter.loaded_tokenizer
        input_text = "Hello world"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True)
        model_adapter.loaded_model(**input_ids)

        # 检查 optional 字段及 optional 下 quarot 的 safetensors 存在性
        _check_optional_quarot_export(tmp_dir)

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.parametrize(
    "test_device, test_dtype",
    [
        pytest.param("cpu", torch.float32),
        pytest.param(
            "npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
        ),
        pytest.param(
            "npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
        ),
    ],
)
@pytest.mark.smoke
def test_quarot_autoround_process(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行QuaRot+AutoRound量化测试
        model_adapter = invoke_test("quarot_autoround.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 测试伪量化
        tokenizer = model_adapter.loaded_tokenizer
        input_text = "Hello world"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True)
        model_adapter.loaded_model(**input_ids)

        # 检查 optional 字段及 optional 下 quarot 的 safetensors 存在性
        _check_optional_quarot_export(tmp_dir)

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.parametrize(
    "test_device, test_dtype",
    [
        pytest.param("cpu", torch.float32),
        pytest.param(
            "npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
        ),
        pytest.param(
            "npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
        ),
    ],
)
@pytest.mark.smoke
def test_quarot_without_optional_export_does_not_create_optional_dir(
    test_device: str,
    test_dtype: torch.dtype,
) -> None:
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行配置中 export_extra_info=False 的 QuaRot 量化测试
        model_adapter = invoke_test("quarot_without_optional_export.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 不导出 optional.quarot.global_rotation 时，不应创建 optional 目录
        _check_optional_dir_not_exists(tmp_dir)
    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.parametrize(
    "test_device, test_dtype",
    [
        pytest.param("cpu", torch.float32),
        pytest.param(
            "npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
        ),
        pytest.param(
            "npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")
        ),
    ],
)
@pytest.mark.smoke
def test_without_quarot_does_not_create_optional_dir(
    test_device: str,
    test_dtype: torch.dtype,
) -> None:
    tmp_dir = tempfile.mkdtemp()

    try:
        # 使用不包含 QuaRot 处理流程的配置，验证不会创建 optional 目录
        model_adapter = invoke_test("w8a8_static_per_channel.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        _check_optional_dir_not_exists(tmp_dir)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
