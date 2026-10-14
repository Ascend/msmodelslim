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
from collections import namedtuple

import pytest
import torch

from msmodelslim.ir import (
    W4A4DynamicPerGroupFakeQuantLinear,
    W4A4DynamicPerChannelFakeQuantLinear,
    W8A8DynamicPerChannelFakeQuantLinear,
    W4A4MXDynamicPerBlockFakeQuantLinear,
)
from .base import FakeLlamaModelAdapter, invoke_test, is_npu_available
from .utils import (
    run_fake_quantization_test,
    check_w4a4_dynamic_per_group_export,
    check_w4a4_dynamic_per_channel_export,
    check_w8a8_dynamic_per_channel_export,
    check_tensors_by_mapping,
    check_w4a4_mx_dynamic_per_block_export,
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
def test_w4a4_dynamic_per_group_quantization(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_group量化测试
        model_adapter = invoke_test("w4a4_dynamic_per_group.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W4A4DynamicPerGroupFakeQuantLinear: check_w4a4_dynamic_per_group_export}
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A4_DYNAMIC",
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
def test_w4a4_dynamic_per_channel_quantization(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_channel量化测试
        model_adapter = invoke_test("w4a4_dynamic_per_channel.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W4A4DynamicPerChannelFakeQuantLinear: check_w4a4_dynamic_per_channel_export}
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A4_DYNAMIC",
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
def test_w4a4_laos_pipeline(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行per_channel量化测试（w8a8-static-per-channel.yaml使用per_tensor+per_channel）
        model_adapter = invoke_test("w4a4_laos.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {
            W4A4DynamicPerGroupFakeQuantLinear: check_w4a4_dynamic_per_group_export,
            W8A8DynamicPerChannelFakeQuantLinear: check_w8a8_dynamic_per_channel_export,
        }
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A4_DYNAMIC",
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
def test_w4a4_laos_with_float_rollback_pipeline(test_device: str, test_dtype: torch.dtype):
    """
    测试W4A4 LAOS pipeline with float rollback功能
    这个测试专门验证新增的_convert_hookir_to_wrapper函数和WrapperIR处理逻辑
    """
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行w4a4_laos_with_float_rollback.yaml配置的测试
        model_adapter = invoke_test("w4a4_laos_with_float_rollback.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 验证模型包含预期的量化模块类型
        quantized_model = model_adapter.loaded_model

        # 检查是否包含W4A4和W8A8量化模块
        has_w4a4_per_group = False
        has_w4a4_per_channel = False
        has_w8a8_per_channel = False

        for name, module in quantized_model.named_modules():
            if isinstance(module, W4A4DynamicPerGroupFakeQuantLinear):
                has_w4a4_per_group = True
            elif isinstance(module, W4A4DynamicPerChannelFakeQuantLinear):
                has_w4a4_per_channel = True
            elif isinstance(module, W8A8DynamicPerChannelFakeQuantLinear):
                has_w8a8_per_channel = True

        # 验证至少有一种量化模块存在
        assert has_w4a4_per_group or has_w4a4_per_channel or has_w8a8_per_channel, (
            "Model should contain at least one quantized module"
        )

        # 使用公共函数进行伪量化测试，验证保存功能
        module_checkers = {
            W4A4DynamicPerGroupFakeQuantLinear: check_w4a4_dynamic_per_group_export,
            W4A4DynamicPerChannelFakeQuantLinear: check_w4a4_dynamic_per_channel_export,
            W8A8DynamicPerChannelFakeQuantLinear: check_w8a8_dynamic_per_channel_export,
        }
        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A4_DYNAMIC",
            module_checkers=module_checkers,
            group_size=32,
        )

        # 验证保存的文件结构
        quant_desc_file = os.path.join(tmp_dir, "quant_model_description.json")
        assert os.path.exists(quant_desc_file), "quant_model_description.json should exist"

        safetensors_files = [f for f in os.listdir(tmp_dir) if f.endswith('.safetensors')]
        assert len(safetensors_files) > 0, "Should have safetensors files saved"

        # 验证在线旋转矩阵的保存 - 使用简化的批量检查函数
        # 定义TensorInfo结构
        TensorInfo = namedtuple("TensorInfo", ["dtype", "shape"])

        # 定义应该存在的tensor映射
        assert_in_safetensors_map = {
            "model.layers.0.self_attn.o_proj.heads_rotation": TensorInfo(torch.float32, (2, 2)),
            "model.layers.1.self_attn.o_proj.heads_rotation": TensorInfo(torch.float32, (2, 2)),
            "model.layers.2.self_attn.o_proj.heads_rotation": TensorInfo(torch.float32, (2, 2)),
            "model.layers.1.mlp.down_proj.kronecker_rotation_m": TensorInfo(torch.float32, (16, 16)),
            "model.layers.1.mlp.down_proj.kronecker_rotation_n": TensorInfo(torch.float32, (16, 16)),
            "model.layers.0.self_attn.q_proj.weight": TensorInfo(torch.int8, None),
            "model.layers.0.self_attn.q_proj.weight_scale": TensorInfo(torch.float32, None),
        }

        # 定义不应该存在的tensor映射（这些层不应该有旋转矩阵）
        assert_not_in_safetensors_set = {
            "model.layers.0.mlp.down_proj.kronecker_rotation_m",
            "model.layers.2.mlp.down_proj.kronecker_rotation_m",
        }

        # 使用简化的批量检查函数（自动加载文件和打印调试信息）
        check_tensors_by_mapping(
            tmp_dir=tmp_dir, assert_in_map=assert_in_safetensors_map, assert_not_in_map=assert_not_in_safetensors_set
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
def test_w4a4_mx_dynamic_per_block_quantization(test_device, test_dtype):
    """测试W4A4 per_token量化功能（act: per_token, weight: per_channel）"""
    torch.set_default_dtype(test_dtype)
    tmp_dir = tempfile.mkdtemp()

    try:
        model_adapter = invoke_test("w4a4_mx_dynamic_per_block.yaml", tmp_dir)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 使用公共函数进行伪量化测试
        module_checkers = {W4A4MXDynamicPerBlockFakeQuantLinear: check_w4a4_mx_dynamic_per_block_export}

        run_fake_quantization_test(
            model_adapter=model_adapter,
            tmp_dir=tmp_dir,
            expected_quant_types="W4A4_MXFP4",
            module_checkers=module_checkers,
        )

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
