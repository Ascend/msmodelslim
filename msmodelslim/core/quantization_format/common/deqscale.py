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

import numpy as np
import torch

from msmodelslim.utils.exception import SchemaValidateError


def deqscale2int64(scale: torch.Tensor) -> torch.Tensor:
    """
    Interpret float32 deq_scale as int32 bit pattern and store as int64.
    The inference side can load as INT64 then cast back to float32 for faster weight loading.
    """
    scale = scale.cpu().numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    return torch.tensor(scale)


def deqscale2int64_by_dtype(scale: torch.Tensor, is_bf16: bool) -> torch.Tensor:
    """
    Convert deq_scale to INT64 based on dtype: keep as-is for bf16, otherwise convert to INT64.
    """
    if is_bf16:
        return scale
    return deqscale2int64(scale)


def int64_deqscale_to_float32(deq_scale: torch.Tensor) -> torch.Tensor:
    """Inverse of ``deqscale2int64``: reinterpret int64 bit-pattern as float32.

    The save side stores float32 deq_scale as int64 (via ``deqscale2int64``) for faster
    weight loading. This function reverses that: reads the int64 tensor, reinterprets
    the lower 32 bits as float32, and returns the recovered scale.
    """
    if deq_scale.dtype in (torch.float16, torch.float32, torch.bfloat16):
        return deq_scale.to(torch.float32)
    if deq_scale.dtype != torch.int64:
        raise SchemaValidateError(
            f"Unexpected deq_scale dtype: {deq_scale.dtype}",
            action="AscendV1 W8A8 deq_scale should be int64 or float32.",
        )
    arr = deq_scale.detach().cpu().numpy().astype(np.int32)
    recovered = np.frombuffer(arr.tobytes(), dtype=np.float32).copy()
    return torch.from_numpy(recovered).reshape(deq_scale.shape)


__all__ = ["deqscale2int64", "deqscale2int64_by_dtype", "int64_deqscale_to_float32"]
