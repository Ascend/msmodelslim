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
# pylint: disable=logging-fstring-interpolation

import os
from enum import Enum, auto
from functools import lru_cache

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from example.common.security.path import json_safe_load, get_valid_read_path, MAX_READ_FILE_SIZE_32G
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import replace_device_align_hook_if_needed
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import get_offloaded_weights_loader_if_have
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.utils import judge_model_with_accelerate

WEIGHT_SCALE_INV = '.weight_scale_inv'
HF_HOOK = '_hf_hook'


class OpsType(Enum):
    FP8 = auto()
    BF16 = auto()
    AUTO = auto()

    @staticmethod
    def get_ops_type(is_bf16: bool, is_fp8: bool):
        if is_bf16 and is_fp8:
            raise ValueError('Using both label fp8 and label bf16.')

        ops_type = OpsType.AUTO
        if is_bf16:
            ops_type = OpsType.BF16
        if is_fp8:
            ops_type = OpsType.FP8
        return ops_type


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape(M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.
    """

    # Get the original dimensions of weight
    m, n = weight.shape

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:m, :n]

    # Perform element-wise multiplication
    weight *= scale_expanded

    # Convert the output to the default dtype
    weight = weight.to(torch.bfloat16)

    return weight


@lru_cache(maxsize=3)
def get_tensor_file(file_dir, file_name):
    file_path = os.path.join(file_dir, file_name)
    file_path = get_valid_read_path(file_path, 'safetensors', size_max=MAX_READ_FILE_SIZE_32G)
    return load_file(file_path, device='cpu')


def get_tensor(tensor_name, fp8_path, weight_map):
    file_name = weight_map[tensor_name]
    loaded_file = get_tensor_file(fp8_path, file_name)
    return loaded_file[tensor_name]


def get_module_by_name(model, submodule_key=None):
    if submodule_key is None:
        return submodule_key
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s, None)
    return cur_mod


def auto_convert_model_fp8_to_bf16(model, model_path, ops_type: OpsType = OpsType.AUTO):
    if ops_type is OpsType.BF16:
        return

    scale_list, weight_map = get_weight_map_and_scale_list_from_index(model, model_path)

    if ops_type is OpsType.FP8:
        if not scale_list:
            raise ValueError('Can not find any fp8 inv scale, please check whether model is of fp8.')
        convert_model_fp8_to_bf16(model, model_path, scale_list, weight_map)
        return

    # auto
    if not scale_list:
        return

    try:
        convert_model_fp8_to_bf16(model, model_path, scale_list, weight_map)
    except KeyError:
        msmodelslim_logger.warning('Safetensors files not match index.json, please check whether model is of bf16.')
        msmodelslim_logger.warning('Skip fp8 to bf16.')
    except Exception as e:
        msmodelslim_logger.error(f'Unexpected error occurred: {e}.')
        raise


def get_weight_map_and_scale_list_from_index(model, model_path):
    model_index_path = os.path.join(model_path, "model.safetensors.index.json")
    model_index = json_safe_load(model_index_path)
    weight_map = model_index['weight_map']
    convert_list = set(
        map(lambda x: x.replace(WEIGHT_SCALE_INV, ''), filter(lambda x: WEIGHT_SCALE_INV in x, weight_map.keys()))
    )
    scale_list = []
    for name, _ in model.named_modules():
        if name in convert_list:
            scale_list.append(name)
    return scale_list, weight_map


def convert_model_fp8_to_bf16(model, model_path, scale_list, weight_map):
    if not judge_model_with_accelerate(model):
        raise ValueError('Deepseek V3/R1 only support npu limited cpu unlimited quantization for now')

    replace_device_align_hook_if_needed(model)

    with torch.no_grad():
        for name in tqdm(scale_list, desc='fp8 to bf16'):
            module = get_module_by_name(model, name)
            scale = get_tensor(name + WEIGHT_SCALE_INV, model_path, weight_map)

            weight_loader = get_offloaded_weights_loader_if_have(module)
            if weight_loader and getattr(module, HF_HOOK).old_hook.offload:
                weight_name = name + '.weight'
                weight_loader[weight_name][:] = weight_dequant(weight_loader[weight_name], scale)
                continue

            device = getattr(module, HF_HOOK).old_hook.execution_device
            # only support npu
            if device != 'cpu':
                device = f'npu:{device}'
            scale = scale.to(device)
            module.weight[:] = weight_dequant(module.weight, scale)
