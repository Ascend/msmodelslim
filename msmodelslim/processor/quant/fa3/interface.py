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

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import nn

from msmodelslim.utils.exception import UnsupportedError, SchemaValidateError


class FA3QuantAdapterInterface(ABC):
    @abstractmethod
    def inject_fa3_placeholders(
        self,
        root_name: str,
        root_module: nn.Module,
        should_inject: Callable[[str], bool],
    ) -> None:
        """
        在模型指定位置安装占位模块，用于 FA3 per-head 激活量化统计。

        要求：
        - root_module.named_modules() 可定位 Attention 模块内的插入点；
        - 对每个需要的插入点 set_submodule(name, FA3QuantPlaceHolder())；
        - 是否注入由回调 should_inject(name: str) 判定（name 为目标模块全名）；
        - 适配器需自行保证注入成功，无需返回注入列表。
        """
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support inject_fa3_placeholders",
            action="Please implement this method before using FA3 activation quantization",
        )


class FA3QuantPlaceHolder(nn.Module):
    """占位模块：供适配器在模型中插入，后续由 Processor 替换为监听器/IR。"""

    def __init__(self, ratio: float = 1.0):
        super().__init__()
        if not isinstance(ratio, float) or ratio < 0.0 or ratio > 1.0:
            raise SchemaValidateError(
                f"Invalid FA3 quant ratio: {ratio}, it should be a float between 0.0 and 1.0",
                action="Please check your model adapter.",
            )
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_ratio(self) -> float:
        return self.ratio
