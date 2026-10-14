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
import random

import numpy as np
import torch
from packaging import version

from msmodelslim.utils.logging import get_logger

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False


def seed_all(seed=1234, mode=False):
    try:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cuda_version = torch.version.cuda
        if cuda_version is not None and version.parse(cuda_version) >= version.parse("10.2"):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['HCCL_DETERMINISTIC'] = str(mode)
        torch.use_deterministic_algorithms(False)
        if is_gpu:
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enable = False
            torch.backends.cudnn.benchmark = False
        else:
            torch_npu.npu.manual_seed_all(seed)
            torch_npu.npu.manual_seed(seed)
    except Exception as e:
        get_logger().error(f"There is an unexpected error while determinating randomness. {e}")
