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

import numpy as np
import torch


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
