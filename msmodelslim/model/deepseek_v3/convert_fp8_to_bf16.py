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
# pylint: disable=disallowed-name,redefined-argument-from-local

import os
from functools import lru_cache
from typing import Dict

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import get_valid_read_path, MAX_READ_FILE_SIZE_32G, json_safe_load

WEIGHT_SCALE_INV = '.weight_scale_inv'
HF_HOOK = '_hf_hook'


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


@lru_cache(maxsize=1)
def get_inv_weight_map(model_path: str):
    model_index_path = os.path.join(model_path, "model.safetensors.index.json")
    model_index = json_safe_load(model_index_path)
    weight_map = model_index['weight_map']
    weight_map = {k.replace(WEIGHT_SCALE_INV, ''): v for k, v in weight_map.items() if WEIGHT_SCALE_INV in k}
    return weight_map


def get_inv_tensor(tensor_name, fp8_path, weight_map):
    file_name = weight_map[tensor_name]
    file_path = os.path.join(fp8_path, file_name)
    file_path = get_valid_read_path(file_path, 'safetensors', size_max=MAX_READ_FILE_SIZE_32G)
    with safe_open(file_path, framework='pt', device='cpu') as f:
        return f.get_tensor(tensor_name + WEIGHT_SCALE_INV)


def auto_convert_module_fp8_to_bf16(name: str, module: nn.Module, model_path: str):
    weight_map = get_inv_weight_map(model_path)

    if not weight_map:
        return

    try:
        sub_weight_map = {
            sub_name: weight_map[sub_name]
            for sub_name, _ in module.named_modules(prefix=name)
            if sub_name in weight_map
        }
        convert_module_fp8_to_bf16(name, module, model_path, weight_map=sub_weight_map)
    except KeyError:
        get_logger().warning('Safetensors files not match index.json, please check whether model is of bf16.')
        get_logger().warning('Skip fp8 to bf16.')


@torch.no_grad()
def convert_module_fp8_to_bf16(name: str, module: nn.Module, model_path: str, weight_map: Dict[str, str]):
    with tqdm(total=len(weight_map), desc='fp8 to bf16') as bar:
        for sub_name, module in module.named_modules(prefix=name):
            if sub_name not in weight_map:
                continue

            scale = get_inv_tensor(sub_name, model_path, weight_map)
            module.weight[:] = weight_dequant(module.weight, scale.to(module.weight.device))
            bar.update(1)
