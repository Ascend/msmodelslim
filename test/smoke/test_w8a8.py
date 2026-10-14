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

import os
import shutil
import tempfile

import pytest
import torch

from msmodelslim.ir import (
    W8A8StaticFakeQuantLinear,
    W8A8DynamicPerChannelFakeQuantLinear,
    W8A8DynamicPerGroupFakeQuantLinear,
    W8A8PDMixFakeQuantLinear,
    W8A8MXDynamicPerBlockFakeQuantLinear,
)
from .base import FakeLlamaModelAdapter, is_npu_available, invoke_test
from .utils import (
    run_fake_quantization_test,
    check_w8a8_static_export,
    check_w8a8_dynamic_per_channel_export,
    check_w8a8_dynamic_per_group_export,
    check_w8a8_pd_mix_export,
    check_w8a8_mx_dynamic_per_block_export,
)


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
def test_w8a8_static_per_channel_quantization(test_device: str, test_dtype: torch.dtype):
    """测试W8A8 per_channel量化功能（act: per_tensor, weight: per_channel）"""

    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_channel量化测试
        model_adapter = invoke_test("w8a8_static_per_channel.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        print(model_adapter.loaded_model)

        # 使用公共函数进行伪量化测试
        module_checkers = {W8A8StaticFakeQuantLinear: check_w8a8_static_export}
        run_fake_quantization_test(
            model_adapter=model_adapter, tmp_dir=tmp_dir, expected_quant_types="W8A8", module_checkers=module_checkers
        )

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
def test_w8a8_mixed_quantization(test_device: str, test_dtype: torch.dtype):
    """测试W8A8混合量化功能（MOE模型）"""

    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行混合量化测试
        model_adapter = invoke_test("w8a8_per_channel_mix.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {
            W8A8StaticFakeQuantLinear: check_w8a8_static_export,
            W8A8DynamicPerChannelFakeQuantLinear: check_w8a8_dynamic_per_channel_export,
        }
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types=["W8A8", "W8A8_DYNAMIC"],
            module_checkers=module_checkers,
        )

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
def test_w8a8_dynamic_per_channel_quantization(test_device: str, test_dtype: torch.dtype):
    """测试W8A8 per_token量化功能（act: per_token, weight: per_channel）"""

    tmp_dir = tempfile.mkdtemp()

    try:
        model_adapter = invoke_test("w8a8_dynamic_per_channel.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W8A8DynamicPerChannelFakeQuantLinear: check_w8a8_dynamic_per_channel_export}
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W8A8_DYNAMIC",
            module_checkers=module_checkers,
        )

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
def test_w8a8_dynamic_per_group_quantization(test_device: str, test_dtype: torch.dtype):
    """测试W8A8 per_token量化功能（act: per_token, weight: per_channel）"""

    tmp_dir = tempfile.mkdtemp()

    try:
        model_adapter = invoke_test("w8a8_dynamic_per_group.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W8A8DynamicPerGroupFakeQuantLinear: check_w8a8_dynamic_per_group_export}
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W8A8_DYNAMIC",
            module_checkers=module_checkers,
            group_size=32,
        )

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
def test_w8a8_pd_mix_quantization(test_device: str, test_dtype: torch.dtype):
    """测试W8A8 PDMIX量化功能（act: pd_mix, weight: per_channel）"""

    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_channel量化测试
        model_adapter = invoke_test("w8a8_pd_mix.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        print(model_adapter.loaded_model)

        # 使用公共函数进行伪量化测试
        module_checkers = {W8A8PDMixFakeQuantLinear: check_w8a8_pd_mix_export}
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W8A8_MIX",
            module_checkers=module_checkers,
        )

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
def test_w8a8_mx_dynamic_per_block_quantization(test_device, test_dtype):
    """测试W8A8 per_token量化功能（act: per_token, weight: per_channel）"""
    torch.set_default_dtype(test_dtype)
    tmp_dir = tempfile.mkdtemp()

    try:
        model_adapter = invoke_test("w8a8_mx_dynamic_per_block.yaml", tmp_dir)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W8A8MXDynamicPerBlockFakeQuantLinear: check_w8a8_mx_dynamic_per_block_export}

        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W8A8_MXFP8",
            module_checkers=module_checkers,
        )

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
