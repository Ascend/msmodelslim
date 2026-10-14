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

from msmodelslim.ir import W4A8MXDynamicPerBlockFakeQuantLinear

from .base import FakeLlamaModelAdapter, invoke_test, is_npu_available
from .utils import check_w4a8_mx_dynamic_per_block_export
from .utils import run_fake_quantization_test


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
def test_w4a8_dynamic_per_channel_quantization(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_channel量化测试
        model_adapter = invoke_test("w4a8_dynamic_per_channel.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {}
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="FLOAT",
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
def test_w4a8_mx_dynamic_per_block_quantization(test_device, test_dtype):
    """测试W4A8 per_token量化功能（act: per_token, weight: per_channel）"""
    torch.set_default_dtype(test_dtype)
    tmp_dir = tempfile.mkdtemp()

    try:
        model_adapter = invoke_test("w4a8_mx_dynamic_per_block.yaml", tmp_dir)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W4A8MXDynamicPerBlockFakeQuantLinear: check_w4a8_mx_dynamic_per_block_export}

        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A8_MXFP",
            module_checkers=module_checkers,
        )

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
