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
# pylint: disable=possibly-used-before-assignment,redefined-outer-name,used-before-assignment

import os
import sys
import argparse
import json
from typing import List

import numpy as np
import torch


from safetensors.torch import load_file, save_file

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)

from example.common.security.path import get_valid_write_path, SafeWriteUmask, get_valid_read_path  # noqa: E402

TOOL_AWQ = 'awq'
TOOL_GPTQ = 'gptq'
QUANT_TYPE = ['W4A16', 'W8A16']
STORAGE_BITS = 32
ORDINAL_PACK_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]
AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def awq_pack(iweight: torch.Tensor, w_bit: int, direction: str = "column"):
    pack_num = STORAGE_BITS // w_bit
    shifts = torch.arange(0, STORAGE_BITS, w_bit, device=iweight.device)

    iweight = iweight.to(torch.int8)
    iweight = torch.bitwise_and(iweight, 0x0F)  # eventually correct overflow

    if direction == "column":
        iweight = iweight.view(-1, iweight.shape[1] // pack_num, pack_num)
        qmatrix = torch.bitwise_left_shift(iweight, shifts[None, None, :]).sum(dim=-1)

    elif direction == "row":
        iweight = iweight.view(iweight.shape[0] // pack_num, pack_num, -1)
        qmatrix = torch.bitwise_left_shift(iweight, shifts[None, :, None]).sum(dim=1)

    qmatrix = qmatrix.to(torch.int32)
    return qmatrix


def apply_order(
    iweight: torch.Tensor,
    w_bit: int,
    direction: str = "column",
    order: List[int] = None,
):
    pack_num = STORAGE_BITS // w_bit
    try:
        if direction == "column":
            iweight = iweight.view(-1, pack_num)[:, order].view(iweight.shape)
        elif direction == "row":
            iweight = iweight.view(pack_num, -1)[order, :].view(iweight.shape)
    except IndexError as ide:
        raise IndexError(
            f"Order index {order} out of range for pack_num {pack_num}. "
            f"Order indices must be < {pack_num} for w_bit={w_bit}"
        ) from ide
    return iweight


def gptq_qweight_pack(iweight: torch.Tensor, w_bit: int):
    i = 0
    row = 0
    iweight = iweight.numpy().astype(np.uint32)
    if len(iweight.shape) < 2:
        raise ValueError("Expected qweight to have at least 2 dimensions, but got shape: {}".format(iweight.shape))
    qweight = np.zeros((iweight.shape[0] // STORAGE_BITS * w_bit, iweight.shape[1]), dtype=np.uint32)
    while row < qweight.shape[0]:
        if w_bit in (4, 8):
            for j in range(i, i + (32 // w_bit)):
                qweight[row] |= iweight[j] << (w_bit * (j - i))
            i += 32 // w_bit
            row += 1
    qweight = qweight.astype(np.int32)
    qweight = torch.from_numpy(qweight)
    return qweight


def gptq_qzeros_pack(zeros: torch.Tensor, w_bit: int):
    i = 0
    col = 0
    if zeros.dtype == torch.bfloat16:
        zeros = zeros.to(torch.float16)

    # AutoGPTQ zeros -= 1, or it may breaks exllama kernels
    zeros -= 1
    zeros = zeros.numpy().astype(np.uint32)
    if len(zeros.shape) < 2:
        raise ValueError("Expected zeros to have at least 2 dimensions, but got shape: {}".format(zeros.shape))
    qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // STORAGE_BITS * w_bit), dtype=np.uint32)
    while col < qzeros.shape[1]:
        if w_bit in (4, 8):
            for j in range(i, i + (32 // w_bit)):
                qzeros[:, col] |= zeros[:, j] << (w_bit * (j - i))
            i += 32 // w_bit
            col += 1

    qzeros = qzeros.astype(np.int32)
    qzeros = torch.from_numpy(qzeros)
    return qzeros


def convert_ms_to_vllm(target_tool, w_bit, weight_dict, json_dict):
    vllm_weight_dict = {}
    for name, quant_type in json_dict.items():
        if name in weight_dict.keys():
            tensor = weight_dict[name]
            if quant_type in QUANT_TYPE:
                order = AWQ_PACK_ORDER
                direction = 'column'
                if name.endswith('.weight') and '.module.weight' not in name:
                    base_name = name.rsplit('.', 1)[0]
                    tmp_key = base_name + '.module.weight'
                    if tmp_key in weight_dict.keys():
                        continue
                    vllm_name = base_name + '.qweight'
                    tensor = tensor.t().contiguous()
                    tensor = torch.clamp(tensor, -(2 ** (w_bit - 1)), 2 ** (w_bit - 1) - 1)
                    if w_bit == 8:
                        tensor = tensor.to(torch.int32)
                    tensor.add_(2 ** (w_bit - 1))

                    if target_tool == TOOL_AWQ:
                        iweights = apply_order(tensor, w_bit, direction, order)
                        qweight = awq_pack(iweights, w_bit)
                    elif target_tool == TOOL_GPTQ:
                        qweight = gptq_qweight_pack(tensor, w_bit)
                    vllm_weight_dict[vllm_name] = qweight

                elif name.endswith('.weight_scale'):
                    vllm_name = name.rsplit('.', 1)[0] + '.scales'
                    tensor = tensor.t().contiguous()
                    vllm_weight_dict[vllm_name] = tensor

                elif name.endswith('.weight_offset'):
                    vllm_name = name.rsplit('.', 1)[0] + '.qzeros'
                    tensor = tensor.t().contiguous()
                    tensor = torch.clamp(tensor, -(2 ** (w_bit - 1)), 2 ** (w_bit - 1) - 1)
                    if w_bit == 8:
                        tensor = tensor.to(torch.int32)
                    tensor.add_(2 ** (w_bit - 1))

                    if target_tool == TOOL_AWQ:
                        izeros = apply_order(tensor, w_bit, direction, order)
                        qzeros = awq_pack(izeros, w_bit, direction)
                    elif target_tool == TOOL_GPTQ:
                        qzeros = gptq_qzeros_pack(tensor, w_bit)
                    vllm_weight_dict[vllm_name] = qzeros

                elif 'module.weight' in name and 'model.norm' not in name:
                    vllm_name = name.replace('module.weight', 'weight')
                    vllm_weight_dict[vllm_name] = tensor
                elif 'model.norm.module.bias' in name or 'model.norm.module.weight' in name:
                    pass
                else:
                    vllm_weight_dict[name] = tensor
            else:
                vllm_weight_dict[name] = tensor
    return vllm_weight_dict


def load_json_info(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        quant_info = json.load(f)

    return quant_info


def check_w_bit(value):
    ivalue = int(value)
    if ivalue not in (4, 8):
        raise argparse.ArgumentTypeError(f"Invalid w_bit value: {value}. Supported values are 4 and 8.")
    return ivalue


def check_target_tool(value):
    if value not in ("awq", "gptq"):
        raise argparse.ArgumentTypeError(f"Invalid target_tool value: {value}. Supported values are 'awq' and 'gptq'.")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Quantied safetensors file path")
    parser.add_argument("--json", type=str, default=None, help="Quantied description file path")
    parser.add_argument(
        "--save_path", type=str, default='res.safetensors', help="The path to save converted quant weights"
    )
    parser.add_argument("--w_bit", type=check_w_bit, default=4, help="Quantied weight bits")
    parser.add_argument(
        "--target_tool", type=check_target_tool, default="awq", help="target tool, value include awq and gptq"
    )
    args = parser.parse_args()

    save_path = args.save_path
    w_bit = args.w_bit
    quant_tool = args.target_tool

    model_path = get_valid_read_path(args.model, size_max=0)
    tensor_info = load_file(model_path)

    json_path = get_valid_read_path(args.json)
    json_info = load_json_info(json_path)

    vllm_weight = convert_ms_to_vllm(quant_tool, w_bit, weight_dict=tensor_info, json_dict=json_info)

    save_path = get_valid_write_path(save_path)
    with SafeWriteUmask(umask=0o377):
        save_file(vllm_weight, save_path)
