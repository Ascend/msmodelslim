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
from typing import Tuple, Optional, Dict

import torch

from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import get_valid_read_path

MAX_READ_FILE_SIZE_1M: int = 1 * 1024 * 1024  # 1MB, 1 * 1024 * 1024

HADAMARD_TXT_DATA_FILE_NAME: Dict[int, str] = {
    200: "had.200.pal.txt",
    172: "had.172.will.txt",
    160: "had.160.tpal.txt",
    156: "had.156.will.txt",
    140: "had.140.pal.txt",
    136: "had.136.twill.txt",
    108: "had.108.will.txt",
    76: "had.76.will.txt",
    60: "had.60.will.txt",
    52: "had.52.will.txt",
    36: "had.36.will.txt",
    28: "had.28.will.txt",
    40: "had.40.twill.txt",
    20: "had.20.will.txt",
    12: "had.12.txt",
}


def txt_safe_load(
    path: str,
    extensions: Tuple[str, ...] = ("txt",),
    size_max: int = MAX_READ_FILE_SIZE_1M,
    check_user_stat: bool = True,
) -> str:
    path = get_valid_read_path(path, extensions, size_max, check_user_stat)

    with open(path, 'r', encoding='utf-8') as txt_file:
        content = txt_file.read()

    return content


def walsh_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Generate a Walsh matrix of given size.
    """
    if size == 1:
        had: torch.Tensor = torch.tensor([[1.0]], dtype=dtype, device=device)
    else:
        had = walsh_matrix(size // 2, dtype=dtype, device=device)
        had = torch.cat([torch.cat([had, had], dim=1), torch.cat([had, -had], dim=1)], dim=0)
    binary_indices: torch.Tensor = (
        torch.arange(size, device=device).unsqueeze(1).bitwise_xor(torch.arange(size, device=device).unsqueeze(0))
    )
    sorted_indices: torch.Tensor = torch.argsort(torch.sum(binary_indices, dim=1))
    return had[sorted_indices]


def load_hadamard_matrix_from_txt(txt_file_name: str, txt_dir: Optional[str] = None) -> torch.FloatTensor:
    if txt_dir is None:
        txt_dir = os.path.join(os.path.dirname(__file__), "hadamard_txt")

    txt_path: str = os.path.join(txt_dir, txt_file_name)

    content: str = txt_safe_load(txt_path, check_user_stat=True)

    # 解析文件内容为矩阵数据
    lines = content.strip().split('\n')
    matrix_data = [[int(a.replace('+', '1').replace('-', '-1')) for a in line] for line in lines]
    return torch.FloatTensor(matrix_data)


def get_had_k(n: int, transpose: bool = False) -> Tuple[Optional[torch.FloatTensor], int]:
    had_k: Optional[torch.FloatTensor] = None
    k: Optional[int] = None
    # 优先尝试较大的 k
    for k_val in sorted(HADAMARD_TXT_DATA_FILE_NAME.keys(), reverse=True):
        if n % k_val == 0 and is_pow2(n // k_val):
            matrix_file: str = HADAMARD_TXT_DATA_FILE_NAME[k_val]
            try:
                mat: torch.FloatTensor = load_hadamard_matrix_from_txt(matrix_file)
            except Exception:
                get_logger().warning(f"Failed to load hadamard matrix from {matrix_file}")
                continue
            had_k = mat.T if transpose else mat
            k = k_val
            break

    if k is None:
        if not is_pow2(n):
            raise UnsupportedError(f"Can not construct hadamard matrix with size {n}")
        k = 1

    return had_k, k


def matmul_had_u(x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    n: int = x.shape[-1]
    had_k: Optional[torch.FloatTensor]
    k: int
    had_k, k = get_had_k(n, transpose)
    input_tensor: torch.Tensor = x.clone().view(-1, n, 1)
    output_tensor: torch.Tensor = input_tensor.clone()
    while input_tensor.shape[1] > k:
        input_tensor = input_tensor.view(input_tensor.shape[0], input_tensor.shape[1] // 2, 2, input_tensor.shape[2])
        output_tensor = output_tensor.view(input_tensor.shape)
        output_tensor[:, :, 0, :] = input_tensor[:, :, 0, :] + input_tensor[:, :, 1, :]
        output_tensor[:, :, 1, :] = input_tensor[:, :, 0, :] - input_tensor[:, :, 1, :]
        output_tensor = output_tensor.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        (input_tensor, output_tensor) = (output_tensor, input_tensor)
    del output_tensor

    if k > 1:
        input_tensor = had_k.view(1, k, k).to(input_tensor) @ input_tensor

    return input_tensor.view(x.shape) / torch.tensor(n).sqrt()


def matmul_had_u_t(x: torch.Tensor) -> torch.Tensor:
    return matmul_had_u(x, transpose=True)


def random_hadamard_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    with torch.device(device=device):
        rot: torch.Tensor = torch.randint(low=0, high=2, size=(size,)).to(dtype)
        rot = rot * 2 - 1
        rot = torch.diag(rot)
        return matmul_had_u(rot).to(device)


def is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)
