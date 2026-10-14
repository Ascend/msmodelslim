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
# pylint: disable=consider-iterating-dictionary,redefined-outer-name,reimported,unspecified-encoding

import json
import os
from typing import Union, List, Dict, Any, Callable, Optional

import torch
from safetensors.torch import safe_open

from msmodelslim.ir import (
    W8A8StaticFakeQuantLinear,
    W8A8DynamicPerChannelFakeQuantLinear,
    W8A8DynamicPerGroupFakeQuantLinear,
    W4A4DynamicPerGroupFakeQuantLinear,
    W4A4DynamicPerChannelFakeQuantLinear,
    W4A4MXDynamicPerBlockFakeQuantLinear,
    W8A8MXDynamicPerBlockFakeQuantLinear,
    W4A8MXDynamicPerBlockFakeQuantLinear,
)


def check_quant_model_description(tmp_dir: str, expected_quant_types: Union[str, List[str]]) -> None:
    """检查quant_model_description.json文件的基本内容"""
    quant_desc_file = os.path.join(tmp_dir, "quant_model_description.json")
    assert os.path.exists(quant_desc_file), "Config file should exist"

    with open(quant_desc_file, "r") as f:
        config_data = json.load(f)

    print(json.dumps(config_data, indent=4, ensure_ascii=False))

    assert isinstance(config_data, dict), "Config data should be a dictionary"
    assert "version" in config_data, "Config data should have version field"
    assert config_data["version"] == "1.0.0", "version should be 1.0.0"
    assert "model_quant_type" in config_data, "Config data should have model_quant_type field"

    if isinstance(expected_quant_types, str):
        assert config_data["model_quant_type"] == expected_quant_types, (
            f"model_quant_type should be {expected_quant_types}, got {config_data['model_quant_type']}"
        )
    else:
        assert config_data["model_quant_type"] in expected_quant_types, (
            f"model_quant_type should be one of {expected_quant_types}, got {config_data['model_quant_type']}"
        )


def check_w8a8_static_export(
    module: W8A8StaticFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor]
) -> None:
    """检查W8A8StaticFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    input_scale_key = f"{name}.input_scale"
    input_offset_key = f"{name}.input_offset"
    quant_bias_key = f"{name}.quant_bias"
    deq_scale_key = f"{name}.deq_scale"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]
    assert weight_tensor.dtype == torch.int8, f"Weight tensor {weight_key} should be int8, got {weight_tensor.dtype}"
    assert weight_tensor.shape == module.weight.shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"
    )

    # 验证输入缩放因子tensor必须存在
    assert input_scale_key in all_tensors, f"Input scale tensor {input_scale_key} must exist in safetensors file"
    input_scale_tensor = all_tensors[input_scale_key]
    assert input_scale_tensor.dtype == torch.float32, (
        f"Input scale tensor {input_scale_key} should be float32, got {input_scale_tensor.dtype}"
    )
    assert input_scale_tensor.shape == (1,), (
        f"Input scale tensor {input_scale_key} shape mismatch: expected {module.input_scale.shape}, "
        f"got {input_scale_tensor.shape}"
    )

    # 验证输入偏移量tensor必须存在
    assert input_offset_key in all_tensors, f"Input offset tensor {input_offset_key} must exist in safetensors file"
    input_offset_tensor = all_tensors[input_offset_key]
    assert input_offset_tensor.dtype == torch.float32, (
        f"Input offset tensor {input_offset_key} should be float32, got {input_offset_tensor.dtype}"
    )
    assert input_offset_tensor.shape == (1,), (
        f"Input offset tensor {input_offset_key} shape mismatch: expected {module.input_offset.shape}, "
        f"got {input_offset_tensor.shape}"
    )

    # 验证量化偏置tensor必须存在
    assert quant_bias_key in all_tensors, f"Quant bias tensor {quant_bias_key} must exist in safetensors file"
    quant_bias_tensor = all_tensors[quant_bias_key]
    assert quant_bias_tensor.dtype == torch.int32, (
        f"Quant bias tensor {quant_bias_key} should be int32, got {quant_bias_tensor.dtype}"
    )
    assert quant_bias_tensor.shape == (module.weight.shape[0],), (
        f"Quant bias tensor {quant_bias_key} shape mismatch: expected {module.quant_bias.shape}, "
        f"got {quant_bias_tensor.shape}"
    )

    # 验证deq缩放因子tensor必须存在（非 bf16 时保存层将 deq_scale 转为 int64，deqscale2int64）
    assert deq_scale_key in all_tensors, f"Deq scale tensor {deq_scale_key} must exist in safetensors file"
    deq_scale_tensor = all_tensors[deq_scale_key]
    assert deq_scale_tensor.dtype == torch.int64, (
        f"Deq scale tensor {deq_scale_key} should be int64, got {deq_scale_tensor.dtype}"
    )
    assert deq_scale_tensor.shape == (module.weight.shape[0],), (
        f"Deq scale tensor {deq_scale_key} shape mismatch: expected {module.deq_scale.shape}, "
        f"got {deq_scale_tensor.shape}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, got {bias_tensor.shape}"
        )


def check_w8a8_dynamic_per_channel_export(
    module: W8A8DynamicPerChannelFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor]
) -> None:
    """检查W8A8DynamicPerChannelFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    weight_scale_key = f"{name}.weight_scale"
    weight_offset_key = f"{name}.weight_offset"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]
    assert weight_tensor.dtype == torch.int8, f"Weight tensor {weight_key} should be int8, got {weight_tensor.dtype}"
    assert weight_tensor.shape == module.weight.shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"
    )

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.float32, (
        f"Weight scale tensor {weight_scale_key} should be float32, got {weight_scale_tensor.dtype}"
    )
    expected_scale_shape = (module.weight.shape[0], 1)
    assert weight_scale_tensor.shape == expected_scale_shape, (
        f"Weight scale tensor {weight_scale_key} shape mismatch: expected {expected_scale_shape}, "
        f"got {weight_scale_tensor.shape}"
    )

    # 验证权重偏移量tensor必须存在
    assert weight_offset_key in all_tensors, f"Weight offset tensor {weight_offset_key} must exist in safetensors file"
    weight_offset_tensor = all_tensors[weight_offset_key]
    assert weight_offset_tensor.dtype == torch.float32, (
        f"Weight offset tensor {weight_offset_key} should be float32, got {weight_offset_tensor.dtype}"
    )
    expected_offset_shape = (module.weight.shape[0], 1)
    assert weight_offset_tensor.shape == expected_offset_shape, (
        f"Weight offset tensor {weight_offset_key} shape mismatch: expected {expected_offset_shape}, "
        f"got {weight_offset_tensor.shape}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, got {bias_tensor.shape}"
        )


def check_w8a8_mx_dynamic_per_block_export(
    module: W8A8MXDynamicPerBlockFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor]
) -> None:
    """检查W8A8DynamicPerChannelFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    weight_scale_key = f"{name}.weight_scale"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]

    assert weight_tensor.dtype == torch.float8_e4m3fn, (
        f"Weight tensor {weight_key} should be float8_e4m3fn, got {weight_tensor.dtype}"
    )
    assert weight_tensor.shape == module.weight.shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"
    )
    weight_float32 = weight_tensor.to(dtype=torch.float32)
    max_val = weight_float32.max().item()
    min_val = weight_float32.min().item()
    assert max_val <= torch.tensor([+448.0], device=weight_tensor.device), (
        f"Weight tensor {weight_key} max value should be less than 448.0, got {max_val}"
    )
    assert min_val >= torch.tensor([-448.0], device=weight_tensor.device), (
        f"Weight tensor {weight_key} min value should be greater than -448.0, got {min_val}"
    )

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.uint8, (
        f"Weight scale tensor {weight_scale_key} should be uint8, got {weight_scale_tensor.dtype}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, \
            got {bias_tensor.shape}"
        )


def check_w4a8_mx_dynamic_per_block_export(
    module: W4A8MXDynamicPerBlockFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor]
) -> None:
    """检查W4A8DynamicPerChannelFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    weight_scale_key = f"{name}.weight_scale"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]

    assert weight_tensor.dtype == torch.uint8, f"Weight tensor {weight_key} should be uint8, got {weight_tensor.dtype}"
    expected_weight_shape = (module.weight.shape[0], module.weight.shape[1] // 2)
    assert weight_tensor.shape == expected_weight_shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {expected_weight_shape}, got {weight_tensor.shape}"
    )

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.uint8, (
        f"Weight scale tensor {weight_scale_key} should be uint8, got {weight_scale_tensor.dtype}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, \
            got {bias_tensor.shape}"
        )


def check_w4a4_mx_dynamic_per_block_export(
    module: W4A4MXDynamicPerBlockFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor]
) -> None:
    """检查W4A4DynamicPerChannelFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    weight_scale_key = f"{name}.weight_scale"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]

    assert weight_tensor.dtype == torch.uint8, f"Weight tensor {weight_key} should be uint8, got {weight_tensor.dtype}"
    expected_weight_shape = (module.weight.shape[0], module.weight.shape[1] // 2)
    assert weight_tensor.shape == expected_weight_shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {expected_weight_shape}, got {weight_tensor.shape}"
    )

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.uint8, (
        f"Weight scale tensor {weight_scale_key} should be uint8, got {weight_scale_tensor.dtype}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, \
            got {bias_tensor.shape}"
        )


def check_w8a8_pd_mix_export(
    module: W8A8StaticFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor]
) -> None:
    """检查W8A8StaticFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    input_scale_key = f"{name}.input_scale"
    input_offset_key = f"{name}.input_offset"
    quant_bias_key = f"{name}.quant_bias"
    deq_scale_key = f"{name}.deq_scale"
    weight_scale_key = f"{name}.weight_scale"
    weight_offset_key = f"{name}.weight_offset"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]
    assert weight_tensor.dtype == torch.int8, f"Weight tensor {weight_key} should be int8, got {weight_tensor.dtype}"
    assert weight_tensor.shape == module.weight.shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"
    )

    # 验证输入缩放因子tensor必须存在
    assert input_scale_key in all_tensors, f"Input scale tensor {input_scale_key} must exist in safetensors file"
    input_scale_tensor = all_tensors[input_scale_key]
    assert input_scale_tensor.dtype == torch.float32, (
        f"Input scale tensor {input_scale_key} should be float32, got {input_scale_tensor.dtype}"
    )
    assert input_scale_tensor.shape == (1,), (
        f"Input scale tensor {input_scale_key} shape mismatch: expected {module.input_scale.shape}, "
        f"got {input_scale_tensor.shape}"
    )

    # 验证输入偏移量tensor必须存在
    assert input_offset_key in all_tensors, f"Input offset tensor {input_offset_key} must exist in safetensors file"
    input_offset_tensor = all_tensors[input_offset_key]
    assert input_offset_tensor.dtype == torch.float32, (
        f"Input offset tensor {input_offset_key} should be float32, got {input_offset_tensor.dtype}"
    )
    assert input_offset_tensor.shape == (1,), (
        f"Input offset tensor {input_offset_key} shape mismatch: expected {module.input_offset.shape}, "
        f"got {input_offset_tensor.shape}"
    )

    # 验证量化偏置tensor必须存在
    assert quant_bias_key in all_tensors, f"Quant bias tensor {quant_bias_key} must exist in safetensors file"
    quant_bias_tensor = all_tensors[quant_bias_key]
    assert quant_bias_tensor.dtype == torch.int32, (
        f"Quant bias tensor {quant_bias_key} should be int32, got {quant_bias_tensor.dtype}"
    )
    assert quant_bias_tensor.shape == (module.weight.shape[0],), (
        f"Quant bias tensor {quant_bias_key} shape mismatch: expected {module.quant_bias.shape}, "
        f"got {quant_bias_tensor.shape}"
    )

    # 验证deq缩放因子tensor必须存在（非 bf16 时保存层将 deq_scale 转为 int64，deqscale2int64）
    assert deq_scale_key in all_tensors, f"Deq scale tensor {deq_scale_key} must exist in safetensors file"
    deq_scale_tensor = all_tensors[deq_scale_key]
    assert deq_scale_tensor.dtype == torch.int64, (
        f"Deq scale tensor {deq_scale_key} should be int64, got {deq_scale_tensor.dtype}"
    )
    assert deq_scale_tensor.shape == (module.weight.shape[0],), (
        f"Deq scale tensor {deq_scale_key} shape mismatch: expected {module.deq_scale.shape}, "
        f"got {deq_scale_tensor.shape}"
    )

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.float32, (
        f"Weight scale tensor {weight_scale_key} should be float32, got {weight_scale_tensor.dtype}"
    )
    expected_scale_shape = (module.weight.shape[0], 1)
    assert weight_scale_tensor.shape == expected_scale_shape, (
        f"Weight scale tensor {weight_scale_key} shape mismatch: expected {expected_scale_shape}, "
        f"got {weight_scale_tensor.shape}"
    )

    # 验证权重偏移量tensor必须存在
    assert weight_offset_key in all_tensors, f"Weight offset tensor {weight_offset_key} must exist in safetensors file"
    weight_offset_tensor = all_tensors[weight_offset_key]
    assert weight_offset_tensor.dtype == torch.float32, (
        f"Weight offset tensor {weight_offset_key} should be float32, got {weight_offset_tensor.dtype}"
    )
    expected_offset_shape = (module.weight.shape[0], 1)
    assert weight_offset_tensor.shape == expected_offset_shape, (
        f"Weight offset tensor {weight_offset_key} shape mismatch: expected {expected_offset_shape}, "
        f"got {weight_offset_tensor.shape}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, got {bias_tensor.shape}"
        )


def check_w8a8_dynamic_per_group_export(
    module: W8A8DynamicPerGroupFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor], group_size
) -> None:
    """检查W8A8DynamicPerChannelFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    weight_scale_key = f"{name}.weight_scale"
    weight_offset_key = f"{name}.weight_offset"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]
    assert weight_tensor.dtype == torch.int8, f"Weight tensor {weight_key} should be int8, got {weight_tensor.dtype}"
    assert weight_tensor.shape == module.weight.shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"
    )

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.float32, (
        f"Weight scale tensor {weight_scale_key} should be float32, got {weight_scale_tensor.dtype}"
    )
    expected_scale_shape = (module.weight.shape[0], module.weight.shape[1] // group_size)
    assert weight_scale_tensor.shape == expected_scale_shape, (
        f"Weight scale tensor {weight_scale_key} shape mismatch: expected {expected_scale_shape}, "
        f"got {weight_scale_tensor.shape}"
    )

    # 验证权重偏移量tensor必须存在
    assert weight_offset_key in all_tensors, f"Weight offset tensor {weight_offset_key} must exist in safetensors file"
    weight_offset_tensor = all_tensors[weight_offset_key]
    assert weight_offset_tensor.dtype == torch.float32, (
        f"Weight offset tensor {weight_offset_key} should be float32, got {weight_offset_tensor.dtype}"
    )
    expected_offset_shape = (module.weight.shape[0], module.weight.shape[1] // group_size)
    assert weight_offset_tensor.shape == expected_offset_shape, (
        f"Weight offset tensor {weight_offset_key} shape mismatch: expected {expected_offset_shape}, "
        f"got {weight_offset_tensor.shape}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, got {bias_tensor.shape}"
        )


def check_w4a4_dynamic_per_group_export(
    module: W4A4DynamicPerGroupFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor], group_size
) -> None:
    """检查W8A8DynamicPerChannelFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    weight_scale_key = f"{name}.weight_scale"
    weight_offset_key = f"{name}.weight_offset"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]
    assert weight_tensor.dtype == torch.int8, f"Weight tensor {weight_key} should be int8, got {weight_tensor.dtype}"
    assert weight_tensor.shape == module.weight.shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"
    )
    assert weight_tensor.max() <= torch.tensor([+7.0], device=weight_tensor.device), (
        f"Weight tensor {weight_key} max value should be less than 7.0, got {weight_tensor.max()}"
    )
    assert weight_tensor.min() >= torch.tensor([-8.0], device=weight_tensor.device), (
        f"Weight tensor {weight_key} min value should be greater than -8.0, got {weight_tensor.min()}"
    )

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.float32, (
        f"Weight scale tensor {weight_scale_key} should be float32, got {weight_scale_tensor.dtype}"
    )
    expected_scale_shape = (module.weight.shape[0], module.weight.shape[1] // group_size)
    assert weight_scale_tensor.shape == expected_scale_shape, (
        f"Weight scale tensor {weight_scale_key} shape mismatch: expected {expected_scale_shape}, "
        f"got {weight_scale_tensor.shape}"
    )

    # 验证权重偏移量tensor必须存在
    assert weight_offset_key in all_tensors, f"Weight offset tensor {weight_offset_key} must exist in safetensors file"
    weight_offset_tensor = all_tensors[weight_offset_key]
    assert weight_offset_tensor.dtype == torch.float32, (
        f"Weight offset tensor {weight_offset_key} should be float32, got {weight_offset_tensor.dtype}"
    )
    expected_offset_shape = (module.weight.shape[0], module.weight.shape[1] // group_size)
    assert weight_offset_tensor.shape == expected_offset_shape, (
        f"Weight offset tensor {weight_offset_key} shape mismatch: expected {expected_offset_shape}, "
        f"got {weight_offset_tensor.shape}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, got {bias_tensor.shape}"
        )


def check_w4a4_dynamic_per_channel_export(
    module: W4A4DynamicPerChannelFakeQuantLinear, name: str, all_tensors: Dict[str, torch.Tensor]
) -> None:
    """检查W8A8DynamicPerChannelFakeQuantLinear模块的导出内容"""
    # 检查权重相关tensor
    weight_key = f"{name}.weight"
    weight_scale_key = f"{name}.weight_scale"
    weight_offset_key = f"{name}.weight_offset"
    bias_key = f"{name}.bias"

    # 验证权重tensor必须存在
    assert weight_key in all_tensors, f"Weight tensor {weight_key} must exist in safetensors file"
    weight_tensor = all_tensors[weight_key]
    assert weight_tensor.dtype == torch.int8, f"Weight tensor {weight_key} should be int8, got {weight_tensor.dtype}"
    assert weight_tensor.shape == module.weight.shape, (
        f"Weight tensor {weight_key} shape mismatch: expected {module.weight.shape}, got {weight_tensor.shape}"
    )
    assert weight_tensor.max() <= torch.tensor([+7.0], device=weight_tensor.device), (
        f"Weight tensor {weight_key} max value should be less than 7.0, got {weight_tensor.max()}"
    )
    assert weight_tensor.min() >= torch.tensor([-8.0], device=weight_tensor.device), (
        f"Weight tensor {weight_key} min value should be greater than -8.0, got {weight_tensor.min()}"
    )

    # 验证权重缩放因子tensor必须存在
    assert weight_scale_key in all_tensors, f"Weight scale tensor {weight_scale_key} must exist in safetensors file"
    weight_scale_tensor = all_tensors[weight_scale_key]
    assert weight_scale_tensor.dtype == torch.float32, (
        f"Weight scale tensor {weight_scale_key} should be float32, got {weight_scale_tensor.dtype}"
    )
    expected_scale_shape = (module.weight.shape[0], 1)
    assert weight_scale_tensor.shape == expected_scale_shape, (
        f"Weight scale tensor {weight_scale_key} shape mismatch: expected {expected_scale_shape}, "
        f"got {weight_scale_tensor.shape}"
    )

    # 验证权重偏移量tensor必须存在
    assert weight_offset_key in all_tensors, f"Weight offset tensor {weight_offset_key} must exist in safetensors file"
    weight_offset_tensor = all_tensors[weight_offset_key]
    assert weight_offset_tensor.dtype == torch.float32, (
        f"Weight offset tensor {weight_offset_key} should be float32, got {weight_offset_tensor.dtype}"
    )
    expected_offset_shape = (module.weight.shape[0], 1)
    assert weight_offset_tensor.shape == expected_offset_shape, (
        f"Weight offset tensor {weight_offset_key} shape mismatch: expected {expected_offset_shape}, "
        f"got {weight_offset_tensor.shape}"
    )

    if module.bias is not None:
        # 验证偏置tensor必须存在
        assert bias_key in all_tensors, f"Bias tensor {bias_key} must exist in safetensors file"
        bias_tensor = all_tensors[bias_key]
        assert bias_tensor.dtype == torch.float32, f"Bias tensor {bias_key} should be float32, got {bias_tensor.dtype}"
        assert bias_tensor.shape == module.bias.shape, (
            f"Bias tensor {bias_key} shape mismatch: expected {module.bias.shape}, got {bias_tensor.shape}"
        )


def run_fake_quantization_test(
    model_adapter: Any,
    tmp_dir: str,
    expected_quant_types: Union[str, List[str]],
    module_checkers: Dict[type, Callable],
    input_text: str = "Hello world",
    group_size: Optional[int] = None,
) -> None:
    """
    运行伪量化测试的公共函数

    Args:
        model_adapter: 模型适配器对象
        tmp_dir: 临时目录路径
        expected_quant_types: 期望的量化类型
        module_checkers: 模块类型到检查函数的映射
        input_text: 输入文本，默认为 "Hello world"
        group_size: 组大小，用于per_group量化
    """
    # 测试 __repr__
    print(model_adapter.loaded_model)

    try:
        import accelerate

        device_map = accelerate.infer_auto_device_map(model_adapter.loaded_model)
        model_adapter.loaded_model = accelerate.dispatch_model(model_adapter.loaded_model, device_map=device_map)
    except ImportError:
        pass

    # 测试伪量化
    tokenizer = model_adapter.loaded_tokenizer
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).to(device=model_adapter.loaded_model.device)
    model_adapter.loaded_model(**input_ids)

    # 检查quant_model_description.json包含基本的正确的内容
    if expected_quant_types:
        check_quant_model_description(tmp_dir, expected_quant_types)

    # 检查safetensors中包含相应模块的导出内容，dtype和shape都符合预期
    quantized_model = model_adapter.loaded_model
    safetensors_files = [f for f in os.listdir(tmp_dir) if f.endswith('.safetensors')]
    assert len(safetensors_files) > 0, "No safetensors files found"

    # 将所有safetensors文件加载到一个字典中，避免频繁打开文件
    all_tensors = {}
    for safetensors_file in safetensors_files:
        file_path = os.path.join(tmp_dir, safetensors_file)
        with safe_open(file_path, framework="pt") as f:
            tensor_keys = list(f.keys())
            assert len(tensor_keys) > 0, f"No tensors found in {safetensors_file}"
            # 将所有tensor加载到字典中
            for key in tensor_keys:
                all_tensors[key] = f.get_tensor(key)

    # 验证每个量化模块的导出内容
    for name, module in quantized_model.named_modules():
        module_type = type(module)
        if module_type in module_checkers:
            check_func = module_checkers[module_type]
            if group_size is not None and 'group' in check_func.__name__:
                check_func(module, name, all_tensors, group_size)
            else:
                check_func(module, name, all_tensors)


def load_all_tensors_from_safetensors(tmp_dir: str) -> dict:
    """
    一次性加载所有safetensors文件中的tensor到字典中

    Args:
        tmp_dir: 临时目录路径，包含保存的safetensors文件

    Returns:
        dict: 包含所有tensor的字典，key为tensor名称，value为tensor数据
    """
    from safetensors.torch import safe_open

    all_tensors = {}
    safetensors_files = [f for f in os.listdir(tmp_dir) if f.endswith('.safetensors')]
    assert len(safetensors_files) > 0, "No safetensors files found in tmp_dir"

    for safetensors_file in safetensors_files:
        file_path = os.path.join(tmp_dir, safetensors_file)
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    return all_tensors


def check_tensor_in_dict(
    tensor_dict: dict,
    tensor_name: str,
    expected_shape: tuple = None,
    expected_dtype: torch.dtype = None,
    should_exist: bool = True,
) -> bool:
    """
    检查指定的tensor是否存在于tensor字典中，并验证其形状和数据类型

    Args:
        tensor_dict: 包含所有tensor的字典
        tensor_name: 要检查的tensor名称
        expected_shape: 期望的tensor形状，如果为None则不检查形状
        expected_dtype: 期望的tensor数据类型，如果为None则不检查数据类型
        should_exist: 是否期望tensor存在，True表示应该存在，False表示不应该存在

    Returns:
        bool: 检查是否通过

    Raises:
        AssertionError: 当检查失败时抛出异常
    """
    tensor_found = tensor_name in tensor_dict
    tensor_data = tensor_dict.get(tensor_name) if tensor_found else None

    if should_exist:
        assert tensor_found, f"Tensor '{tensor_name}' should exist in tensor dict but was not found"

        if expected_shape is not None:
            assert tensor_data.shape == expected_shape, (
                f"Tensor '{tensor_name}' shape mismatch: expected {expected_shape}, got {tensor_data.shape}"
            )

        if expected_dtype is not None:
            assert tensor_data.dtype == expected_dtype, (
                f"Tensor '{tensor_name}' dtype mismatch: expected {expected_dtype}, got {tensor_data.dtype}"
            )

        print(f"[PASS] Tensor '{tensor_name}' found with shape {tensor_data.shape} and dtype {tensor_data.dtype}")
        return True
    else:
        assert not tensor_found, f"Tensor '{tensor_name}' should not exist in tensor dict but was found"
        print(f"[PASS] Tensor '{tensor_name}' correctly not found in tensor dict")
        return True


def check_tensor_in_safetensors(
    tmp_dir: str,
    tensor_name: str,
    expected_shape: tuple = None,
    expected_dtype: torch.dtype = None,
    should_exist: bool = True,
) -> bool:
    """
    检查指定的tensor是否存在于保存的safetensors文件中，并验证其形状和数据类型
    （为了向后兼容保留此函数，但推荐使用load_all_tensors_from_safetensors + check_tensor_in_dict）

    Args:
        tmp_dir: 临时目录路径，包含保存的safetensors文件
        tensor_name: 要检查的tensor名称
        expected_shape: 期望的tensor形状，如果为None则不检查形状
        expected_dtype: 期望的tensor数据类型，如果为None则不检查数据类型
        should_exist: 是否期望tensor存在，True表示应该存在，False表示不应该存在

    Returns:
        bool: 检查是否通过

    Raises:
        AssertionError: 当检查失败时抛出异常
    """
    all_tensors = load_all_tensors_from_safetensors(tmp_dir)
    return check_tensor_in_dict(all_tensors, tensor_name, expected_shape, expected_dtype, should_exist)


def check_tensors_by_mapping(tmp_dir: str, assert_in_map: dict = None, assert_not_in_map: set = None) -> None:
    """
    基于映射字典批量检查tensor的存在性、形状和数据类型
    自动加载safetensors文件并打印调试信息

    Args:
        tmp_dir: 临时目录路径，包含保存的safetensors文件
        assert_in_map: 应该存在的tensor映射，格式为 {tensor_name: TensorInfo(dtype, shape)}
        assert_not_in_map: 不应该存在的tensor映射，格式为 {tensor_name: TensorInfo(dtype, shape)}

    Raises:
        AssertionError: 当检查失败时抛出异常
    """
    # 加载所有tensor
    all_tensors = load_all_tensors_from_safetensors(tmp_dir)

    # 打印调试信息
    print(f"Total saved tensors: {len(all_tensors)}")
    rotation_tensors = [name for name in all_tensors.keys() if 'rotation' in name.lower()]
    print(f"Rotation tensors found: {rotation_tensors}")

    # 打印旋转tensor的实际形状
    for tensor_name in rotation_tensors:
        if tensor_name in all_tensors:
            tensor_data = all_tensors[tensor_name]
            print(f"Tensor '{tensor_name}' shape: {tensor_data.shape}, dtype: {tensor_data.dtype}")

    # 检查应该存在的tensor
    if assert_in_map:
        print(f"Checking {len(assert_in_map)} tensors that should exist...")
        for tensor_name, tensor_info in assert_in_map.items():
            check_tensor_in_dict(
                tensor_dict=all_tensors,
                tensor_name=tensor_name,
                expected_shape=tensor_info.shape,
                expected_dtype=tensor_info.dtype,
                should_exist=True,
            )

    # 检查不应该存在的tensor
    if assert_not_in_map:
        print(f"Checking {len(assert_not_in_map)} tensors that should NOT exist...")
        for tensor_name in assert_not_in_map:
            check_tensor_in_dict(tensor_dict=all_tensors, tensor_name=tensor_name, should_exist=False)
