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

from typing import Optional

import torch
from pydantic import BaseModel
from torch import distributed as dist, nn

from msmodelslim.utils.distributed import sync_base_operation
from msmodelslim.utils.exception import SpecError


class RecallWindowObserverConfig(BaseModel):
    ratio: float = 1.0
    dim: int = -1
    keepdim: bool = False


class RecallWindowObserver(nn.Module):
    def __init__(self, config: RecallWindowObserverConfig):
        super().__init__()
        self.config = config
        self._min_values = None
        self._max_values = None

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
        sample_min, sample_max = recall_window(x, self.config.ratio, self.config.dim, self.config.keepdim)
        if self._min_values is None:
            self._min_values = sample_min
        else:
            index = sample_min < self._min_values
            self._min_values[index] = sample_min[index]

        if self._max_values is None:
            self._max_values = sample_max
        else:
            index = sample_max > self._max_values
            self._max_values[index] = sample_max[index]

        if sync and dist.is_initialized():
            sync_base_operation(self._min_values, op="min", group=group)
            sync_base_operation(self._max_values, op="max", group=group)

    def reset(self):
        self._min_values = None
        self._max_values = None

    def get_min(self) -> torch.Tensor:
        if self._min_values is None:
            raise SpecError(
                "Trying to get min but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated.",
                action="Please check your model and quant config.",
            )
        return self._min_values

    def get_max(self) -> torch.Tensor:
        if self._max_values is None:
            raise SpecError(
                "Trying to get max but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated.",
                action="Please check your model and quant config.",
            )
        return self._max_values


def recall_window(tensor: torch.Tensor, ratio: float, dim=-1, keepdim=False):
    if dim < 0:
        dim += tensor.dim()

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    total_elements = tensor.size(dim)
    target_elements = int(ratio * total_elements)

    left_endpoints = []
    right_endpoints = []

    for head in sorted_tensor:
        min_window_length = float('inf')
        best_window = (0, 0)

        for i in range(total_elements - target_elements + 1):
            window_start = head[i].item()
            window_end = head[i + target_elements - 1].item()
            window_length = window_end - window_start
            if window_length < min_window_length:
                min_window_length = window_length
                best_window = (window_start, window_end)
        left_endpoints.append(best_window[0])
        right_endpoints.append(best_window[1])

    left_endpoints = torch.tensor(left_endpoints, device=tensor.device, dtype=tensor.dtype)
    right_endpoints = torch.tensor(right_endpoints, device=tensor.device, dtype=tensor.dtype)

    if keepdim:
        left_endpoints = left_endpoints.unsqueeze(dim)
        right_endpoints = right_endpoints.unsqueeze(dim)

    return left_endpoints, right_endpoints
