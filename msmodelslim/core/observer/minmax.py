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

from typing import Optional, Tuple, Union, List, Literal

import torch
from pydantic import BaseModel
from torch import distributed as dist, nn

from msmodelslim.utils.distributed import sync_base_operation
from msmodelslim.utils.exception import SpecError


class MinMaxObserverConfig(BaseModel):
    dim: Union[int, List[int]] = []
    keepdim: bool = False
    aggregation_type: Literal["max", "mean"] = "max"


class MsMinMaxObserver(nn.Module):
    def __init__(self, config: MinMaxObserverConfig):
        super().__init__()
        self.config = config
        if config.aggregation_type == "max":
            self._impl = _MaxMinMaxObserver(config)
        else:
            self._impl = _MeanMinMaxObserver(config)

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
        self._impl.update(x, sync, group)

    def reset(self):
        self._impl.reset()

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._impl.get_min_max()


class _MaxMinMaxObserver(nn.Module):
    def __init__(self, config: MinMaxObserverConfig):
        super().__init__()
        self.config = config
        self.min_val = None
        self.max_val = None

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
        current_min = torch.amin(x, self.config.dim, self.config.keepdim)
        current_max = torch.amax(x, self.config.dim, self.config.keepdim)
        if self.min_val is None:
            self.min_val = current_min
        else:
            self.min_val = torch.min(self.min_val, current_min)

        if self.max_val is None:
            self.max_val = current_max
        else:
            self.max_val = torch.max(self.max_val, current_max)

        if sync and dist.is_initialized():
            sync_base_operation(self.min_val, op='min')
            sync_base_operation(self.max_val, op='max')

    def reset(self):
        self.min_val = None
        self.max_val = None

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_val is None or self.max_val is None:
            raise SpecError(
                "Trying to get stats but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated.",
                action="Please check your model and quant config.",
            )
        return self.min_val, self.max_val


class _MeanMinMaxObserver(nn.Module):
    def __init__(self, config: MinMaxObserverConfig):
        super().__init__()
        self.config = config
        self.min_sum = None
        self.max_sum = None
        self.batch_count = 0
        self.mean_min = None
        self.mean_max = None
        self._orig_dtype = None

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
        current_min = torch.amin(x, self.config.dim, self.config.keepdim)
        current_max = torch.amax(x, self.config.dim, self.config.keepdim)
        if self._orig_dtype is None:
            self._orig_dtype = current_min.dtype
        self.batch_count += 1
        if self.min_sum is None:
            self.min_sum = current_min.to(torch.float64)
            self.max_sum = current_max.to(torch.float64)
        else:
            self.min_sum.add_(current_min.to(torch.float64))
            self.max_sum.add_(current_max.to(torch.float64))
        self.mean_min = (self.min_sum / self.batch_count).to(self._orig_dtype)
        self.mean_max = (self.max_sum / self.batch_count).to(self._orig_dtype)

        if sync and dist.is_initialized():
            if self.min_sum is not None and self.max_sum is not None:
                temp_min_sum = self.min_sum.clone()
                temp_max_sum = self.max_sum.clone()
                sync_base_operation(temp_min_sum, op='sum')
                sync_base_operation(temp_max_sum, op='sum')
                batch_count_tensor = torch.tensor([self.batch_count], device=x.device, dtype=torch.int64)
                sync_base_operation(batch_count_tensor, op='sum')
                global_batch_count = batch_count_tensor.item()
                if global_batch_count > 0:
                    self.mean_min = (temp_min_sum / global_batch_count).to(self._orig_dtype)
                    self.mean_max = (temp_max_sum / global_batch_count).to(self._orig_dtype)

    def reset(self):
        self.min_sum = None
        self.max_sum = None
        self.batch_count = 0
        self.mean_min = None
        self.mean_max = None
        self._orig_dtype = None

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mean_min is None or self.mean_max is None:
            raise SpecError(
                "Trying to get stats but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated.",
                action="Please check your model and quant config.",
            )
        return self.mean_min, self.mean_max


class MinMaxBlockObserverConfig(BaseModel):
    method: str = 'max'
    clip: float = 1.0


class MsMinMaxBlockObserver(nn.Module):
    def __init__(self, config: MinMaxBlockObserverConfig):
        super().__init__()
        self.config = config
        self.min_val = None
        self.max_val = None

    def update(
        self,
        x: torch.Tensor,
        sync: bool = True,
        group: Optional[dist.ProcessGroup] = None,
        shared_exp_axes=None,  # 用于指定需要共享指数（或缩放因子）的维度
    ):
        if self.config.method == "max":
            if shared_exp_axes is None:
                # 若未指定共享维度，则直接计算输入张量x的全局绝对值最大值
                self.max_val = torch.max(torch.abs(x))
            else:
                # 若指定了共享维度，需沿这些维度聚合计算最大值（用于生成共享的指数/缩放因子）
                self.max_val = x
                for axis in shared_exp_axes:
                    # 需沿当前维度取绝对值的最大值，keepdim=True保持维度结构
                    # 该操作会将指定维度上的所有元素聚合为一个最大值，实现跨该维度的统计量共享
                    self.max_val, _ = torch.max(torch.abs(self.max_val), dim=axis, keepdim=True)
                    # 乘以配置的裁剪系数，对最大值进行限制（避免异常值影响）
                    self.max_val = self.max_val * self.config.clip
        elif self.config.method == 'none':
            self.max_val = torch.abs(x)  # 若方法为'none'，则直接记录每个元素的绝对值作为max_val
        self.min_val = self.max_val.clone()  # min_val暂用max_val的副本

        if sync and dist.is_initialized():
            sync_base_operation(self.min_val, op='min')
            sync_base_operation(self.max_val, op='max')

    def reset(self):
        self.min_val = None
        self.max_val = None

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_val is None or self.max_val is None:
            raise SpecError(
                "Trying to get stats but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated.",
                action="Please check your model and quant config.",
            )
        return self.min_val, self.max_val
