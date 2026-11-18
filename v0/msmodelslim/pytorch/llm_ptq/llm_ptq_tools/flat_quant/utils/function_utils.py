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

import math
from collections import OrderedDict
import torch

import numpy as np
from scipy.linalg import qr

npu_available = False
try:
    import torch_npu
except ImportError:
    pass
else:
    npu_available = True


def get_init_scale(w_smax, x_smax, alpha=0.5):
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def get_random_orthg(size):
    h = np.random.randn(size, size)
    q, r = qr(h)
    q_modified = q @ np.diag(np.sign(np.diag(r)))
    return torch.from_numpy(q_modified)


def get_init_weight(dim, ):
    return get_random_orthg(dim)


def get_inverse(matrix):
    dtype = matrix.dtype
    if not npu_available:
        return matrix.double().inverse().to(dtype)
    else:
        device = matrix.device
        return matrix.cpu().double().inverse().to(device=device, dtype=dtype)


def get_n_set_parameters_byname(model, required_names):
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return iter(params)


def set_require_grad_all(model, requires_grad):
    for _, param in model.named_parameters():
        param.requires_grad = requires_grad
    return
