#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from typing import List, Any, Optional, Tuple, Dict
from enum import Enum
import math
import torch
from torch import nn
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.seed import seed_all
from .hadamard import random_hadamard_matrix


GLOBAL_DTYPE = torch.float32


def random_hadamard_matrix_block(in_channel, stride, eye_step, dtype=torch.float32, device=torch.device("cpu")):
    """生成对角线上放置Hadamard矩阵的in_channel维矩阵"""
    # 生成stride维的Hadamard矩阵
    h_stride = random_hadamard_matrix(stride, dtype, device)

    # 创建in_channel维的全零张量，并确保它在指定的设备上
    h_in_channel = torch.zeros((in_channel, in_channel), device=device, dtype=dtype)
    # 将stride维的Hadamard矩阵的副本放置在in_channel维矩阵的对角线上
    num_blocks = in_channel // stride
    for i in range(num_blocks):
        start_idx = i * stride
        end_idx = start_idx + stride
        if i in eye_step:
            h_in_channel[start_idx:end_idx, start_idx:end_idx] = torch.eye(stride).to(device=device, dtype=dtype)
        else:
            h_in_channel[start_idx:end_idx, start_idx:end_idx] = h_stride

    return h_in_channel


class QuaRotMode(Enum):
    HADAMARD = "hadamard"
    BLOCK_HADAMARD_SHIFTED = "block_hadamard_shifted"


def create_rot(
    mode: QuaRotMode,
    size: int,
    block_size: int = -1,
    rot_step: int = 1,
    eye_step: tuple = (-1,),
    seed: int = 1234,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    创建旋转矩阵。

    Args:
        mode: 旋转模式
        size: 旋转矩阵尺寸
        block_size: 块大小，默认为-1
        rot_step: 旋转步数，默认为1
        eye_step: eye步数，默认为(-1,)
        seed: 随机数种子，默认为1234

    Returns:
        旋转矩阵
    """
    # 设置随机数种子
    seed_all(seed)
    device = torch.device("cpu")
    shift = 16

    if mode == QuaRotMode.HADAMARD:
        if block_size == -1:
            transformation_dim = size
        else:
            transformation_dim = block_size
        rot = random_hadamard_matrix(transformation_dim, dtype, device)
        if block_size != -1:
            rot = rot.repeat(size // block_size, 1, 1)
            rot = torch.block_diag(*rot)
    elif mode == QuaRotMode.BLOCK_HADAMARD_SHIFTED:
        block_size = 32 if block_size == -1 else block_size
        rot = random_hadamard_matrix_block(size, block_size, eye_step, dtype, device)
        identity = torch.eye(size, dtype=torch.float32)
        p = torch.cat((identity[:, -shift:], identity[:, :-shift]), dim=1).to(rot.device)
        if rot_step == 1:
            rot = rot @ p @ rot
        elif rot_step == 2:
            rot = rot @ p @ rot @ p.T @ rot.T
        elif rot_step == 3:
            rot = rot @ p @ rot @ p.T @ rot.T @ p @ rot @ p.T @ rot.T
        else:
            raise UnsupportedError("rot_step must be 1, 2, or 3!", action="Please check the rot_step!")

    return rot


def _rotation_device(weight_device: torch.device) -> torch.device:
    """Where to run the rotation product.

    The Ascend images ship OMP_NUM_THREADS=1, so a hidden-size rotation on the host
    is a single-core float32 matmul that runs for minutes (256 cores sit idle);
    the accelerator finishes the same product in well under a second.  Only
    redirect tensors that are already on the host, and only when this process
    really has an NPU open.
    """
    if weight_device.type != "cpu":
        return weight_device
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available() or not npu.is_initialized():
        return weight_device
    return torch.device(f"npu:{npu.current_device()}")


def rotate_linear(linear: torch.nn.Linear, rot: Any, right_rotate: bool = True) -> None:
    if isinstance(rot, (list, tuple)):
        rot = torch.block_diag(*rot)
    dtype = linear.weight.data.dtype
    device = linear.weight.device
    compute_device = _rotation_device(device)

    weight_data = linear.weight.data.to(device=compute_device, dtype=GLOBAL_DTYPE)
    dim_size = weight_data.shape[1] if right_rotate else weight_data.shape[0]
    # support diag block rotate automatically
    if dim_size != rot.shape[0]:
        block_num = dim_size // rot.shape[0]
        if dim_size % rot.shape[0] != 0:
            raise UnsupportedError(
                "rotate matrix dim must be a divisor of weight dim!",
                action="Please check the linear weight dim and rotate matrix dim!",
            )
        rot = torch.block_diag(*[rot] * block_num)
    if right_rotate:
        linear.weight.data = torch.matmul(weight_data, rot.to(weight_data)).to(device=device, dtype=dtype)
    else:
        linear.weight.data = torch.matmul(rot.T.to(weight_data), weight_data).to(device=device, dtype=dtype)
        if linear.bias is not None:
            bias_data = linear.bias.data.to(device=compute_device, dtype=GLOBAL_DTYPE)
            linear.bias.data = torch.matmul(rot.T.to(weight_data), bias_data).to(device=device, dtype=dtype)


def rotate_weight(weight: torch.Tensor, rot: Any, right_rotate: bool = True) -> None:
    if isinstance(rot, (list, tuple)):
        rot = torch.block_diag(*rot)
    dtype = weight.data.dtype
    device = weight.device
    compute_device = _rotation_device(device)

    weight_data = weight.data.to(device=compute_device, dtype=GLOBAL_DTYPE)
    dim_size = weight_data.shape[1] if right_rotate else weight_data.shape[0]
    # support diag block rotate automatically
    if dim_size != rot.shape[0]:
        block_num = dim_size // rot.shape[0]
        if dim_size % rot.shape[0] != 0:
            raise UnsupportedError(
                "rotate matrix dim must be a divisor of weight dim!",
                action="Please check the weight dim and rotate matrix dim!",
            )
        rot = torch.block_diag(*[rot] * block_num)
    if right_rotate:
        weight.data = torch.matmul(weight_data, rot.to(weight_data)).to(device=device, dtype=dtype)
    else:
        weight.data = torch.matmul(rot.T.to(weight_data), weight_data).to(device=device, dtype=dtype)


def fuse_ln_linear(layernorms: List[torch.nn.Module], linear_layers: List[torch.nn.Linear]) -> None:
    ln_data = []
    ln_bias_data = []
    for ln in layernorms:
        if hasattr(ln, 'bias'):
            ln_bias_data.append(ln.bias.data)
        if ln.weight.dim() != 1:
            raise UnsupportedError(
                "layernorm type is not supported! quarot rotate only support RMSNorm!",
                action="Please check the model's layernorm type!",
            )
        ln_data.append(ln.weight.data)
    ln_weight = torch.concat(ln_data, dim=0) if len(ln_data) > 1 else ln_data[0]
    ln_weight = ln_weight.to(dtype=GLOBAL_DTYPE)
    ln_bias = torch.concat(ln_bias_data, dim=0).to(dtype=GLOBAL_DTYPE) if len(ln_bias_data) > 1 else None
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        current_weight = linear.weight.data.to(dtype=GLOBAL_DTYPE)
        if ln_weight.shape[0] != current_weight.shape[1]:
            raise UnsupportedError(
                "layernorm weight dim must be equal to linear weight input dim!",
                action="Please check the model's layernorm and linear input size!",
            )
        linear.weight.data = (current_weight * ln_weight.to(device=current_weight.device, dtype=GLOBAL_DTYPE)).to(
            linear_dtype
        )
        if ln_bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=GLOBAL_DTYPE))
            linear.bias.data = linear.bias.data + torch.matmul(current_weight, ln_bias.to(device=current_weight.device))
            linear.bias.data = linear.bias.data.to(linear_dtype)
    for ln in layernorms:
        ln.weight.data.fill_(1.0)
        if hasattr(ln, 'bias'):
            ln.bias.data.fill_(0.0)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    weight_tmp = linear.weight.data.double()
    linear.weight.data = weight_tmp - weight_tmp.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def is_power_of_two(n: int) -> bool:
    """检查一个数是否为2的幂"""
    return n > 0 and (n & (n - 1)) == 0


def get_decompose_dim(n: int) -> Tuple[int, int]:
    """获取分解维度"""
    sup_list = {1, 2} | {4 * i for i in range(1, 65)}
    max_sup = max(sup_list)

    min_a = int(math.sqrt(n))
    if min_a * min_a < n:
        min_a += 1
    for a in range(min_a, max_sup + 1):
        tmp = a * a - n
        if tmp < 0:
            continue
        b = int(math.sqrt(tmp))
        if b * b == tmp and (a - b) in sup_list and (a + b) in sup_list:
            return a - b, a + b

    raise UnsupportedError(f"Can not decompose {n}")


def online_rotate_o_proj_input(
    ov_pairs: Dict[nn.Module, nn.Module], rot_online: Optional[torch.Tensor], num_attn_heads: int
) -> None:
    """在线旋转矩阵融合到o_proj层权重"""
    for o_proj, _ in ov_pairs.items():
        dtype = o_proj.weight.dtype
        device = o_proj.weight.device
        identity = torch.eye(o_proj.weight.shape[1] // num_attn_heads, dtype=dtype, device=device)
        rot_online = rot_online.to(device)
        h_full = torch.kron(rot_online, identity)
        rotate_linear(o_proj, h_full.to(device))
        del h_full


def online_rotate_down_proj(up_down_pairs: Dict[nn.Module, nn.Module], rot1: torch.Tensor, rot2: torch.Tensor) -> None:
    """在线旋转矩阵融合到down_proj层权重"""
    for _, down_proj in up_down_pairs.items():
        dtype = down_proj.weight.dtype
        device = down_proj.weight.device

        init_shape = down_proj.weight.data.shape
        weight = down_proj.weight.data.view(-1, rot1.shape[0], rot2.shape[0])
        weight = torch.matmul(weight, rot2.to(device, dtype)).to(device=device, dtype=dtype)
        weight = torch.matmul(rot1.T.to(device, dtype), weight).reshape(init_shape).to(device=device, dtype=dtype)
        down_proj.weight.data.copy_(weight)

        del weight
