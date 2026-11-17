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

Smooth quantization types and configurations.

Data classes for smooth quantization algorithms including:
- OASQ: Outlier-Aware Smoothing Quantization algorithm
- IterSmooth: Iterative Smooth algorithm types
- FlexSmoothQuant: Flex Smooth Quantization algorithm types
- FlexAWQSSZ: Flex AWQ SSZ algorithm types
"""

from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from msmodelslim.core.quantizer.linear import LinearQConfig


@dataclass
class SmoothQuantContext:
    """SmoothQuant runtime context"""

    version: int
    a_smooth_scale: torch.Tensor
    shift: torch.Tensor


@dataclass
class SmoothQuantConfig:
    """SmoothQuant algorithm configuration"""

    version: int = 1
    alpha: float = 0.5
    shift: bool = False


@dataclass
class IterSmoothContext:
    version: int
    a_smooth_scale: torch.Tensor
    shift: torch.Tensor


@dataclass
class OASQContext:
    version: int
    a_smooth_scale: torch.Tensor
    shift: Optional[torch.Tensor] = None


@dataclass
class IterSmoothConfig:
    """

    iter_smooth算法的配置项。
    允许后续扩展配置项，但仅可新增新字段，且不得修改已有字段，
    version用于指定配置版本号，每次修改后，版本号需要加1。

    """

    version: int = 1
    alpha: float = 0.9
    shift: bool = False
    scale_min: float = 1e-5


@dataclass
class OASQConfig:
    max_iters: Optional[int] = None
    shift: bool = False
    version: int = 1


@dataclass
class FlexSmoothQuantContext:
    version: int
    a_smooth_scale: torch.Tensor
    tensors: Optional[List[torch.Tensor]] = None


@dataclass
class FlexSmoothQuantConfig:
    """

    flex_smooth_quant算法的配置项。
    允许后续扩展配置项，但仅可新增新字段，且不得修改已有字段，
    version用于指定配置版本号，每次修改后，版本号需要加1。

    """

    version: int = 1
    alpha: Optional[float] = None
    beta: Optional[float] = None
    extra_config: Optional[Dict[str, Any]] = None


@dataclass
class FlexAWQSSZContext:
    version: int
    tensors: List[torch.Tensor]


@dataclass
class FlexAWQSSZConfig:
    """

    flex_awq_ssz算法的配置项。

    """

    version: int = 1
    alpha: Optional[float] = None
    beta: Optional[float] = None
    qconfig: Optional['LinearQConfig'] = None


SmoothContext = Union[
    SmoothQuantContext,
    IterSmoothContext,
    OASQContext,
    FlexSmoothQuantContext,
    FlexAWQSSZContext,
]
