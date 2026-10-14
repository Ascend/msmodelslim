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

from .base import invoke_test, is_npu_available


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
        model_adapter = invoke_test("kv_smooth.yaml", tmp_dir, device=test_device)  # noqa: F841
    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
