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

from torch import nn

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from msmodelslim.core.quantization_schema import RMSNormBias


class Subgraph:
    pass


@dataclass
class NonFusionSubgraph(Subgraph):
    """
    该子图代表了非融合操作符。
    """

    linears: List[nn.Linear]
    linear_names: Optional[List[str]] = None


@dataclass
class NormLinearSubgraph(Subgraph):
    """

    该子图可以用以下计算公式表示：

    x = norm(x)
    y = torch.cat([linear(x) for linear in linears], dim=-1)

    其中，norm是归一化层，linears是线性层列表。

    """

    norm: Union["RMSNormBias"]
    linears: List[nn.Linear]
    linear_names: Optional[List[str]] = None


@dataclass
class LinearLinearSubgraph(Subgraph):
    """
    该子图可以用以下计算公式表示：

    y = linear2(linear1(x))

    其中，linear1和linear2是线性层，linear1的输出是linear2的输入。
    """

    linear1: nn.Linear
    linear2: nn.Linear
    linear1_name: Optional[str] = None
    linear2_name: Optional[str] = None


@dataclass
class OVSubgraph(Subgraph):
    """
    该子图代表了Attention中的O和V的子图。

    对于MHA，通常有
        num_attention_heads = key_value_heads

    对于MQA，通常有
        key_value_heads = 1

    对于GQA，通常有
        num_attention_heads > key_value_heads
        num_attention_heads % key_value_heads == 0

    """

    o_proj: nn.Linear
    v_proj: nn.Linear
    num_attention_heads: int
    key_value_heads: int
    extra_config: Optional[Dict[str, Any]] = None
    o_proj_name: Optional[str] = None
    v_proj_name: Optional[str] = None


@dataclass
class UpDownSubgraph(Subgraph):
    """
    该子图代表了MLP中的UpDown子图，该子图通常可以用以下计算公式表示：

    y = down_proj(ReLU(gate_proj(x)) * up_proj(x))

    其中，up_proj和down_proj是线性层，gate_proj是sigmoid激活函数，ReLU是激活函数。
    """

    up_proj: nn.Linear
    down_proj: nn.Linear
    gate_proj: nn.Linear
    up_proj_name: Optional[str] = None
    down_proj_name: Optional[str] = None
    gate_proj_name: Optional[str] = None
