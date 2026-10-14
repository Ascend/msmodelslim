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
from enum import Enum
from typing import List

from pydantic import BaseModel

from msmodelslim.utils.exception import UnsupportedError


class KVSmoothFusedType(Enum):
    StateViaRopeToNorm = 'state-rope-norm'
    StateViaRopeToLinear = 'state-rope-linear'


class KVSmoothFusedUnit(BaseModel):
    attention_name: str  # specific whole name of attention layer, e.g. "model.layers.0.self_attn"
    layer_idx: int  # specific layer index, e.g. 0
    fused_from_query_states_name: str  # specific base name of module fused from query_states, e.g. "q_proj"
    fused_from_key_states_name: str  # specific base name of module fused from key_states, e.g. "k_proj"
    fused_type: KVSmoothFusedType  # fused type, support state-rope-norm and state-rope-linear for now


class KVSmoothFusedInterface(ABC):
    @abstractmethod
    def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
        raise UnsupportedError(
            f'{self.__class__.__name__} does not support get_kvcache_smooth_fused_subgraph',
            action='Please implement this method before using kvcache smooth',
        )

    @abstractmethod
    def get_head_dim(self) -> int:
        raise UnsupportedError(
            f'{self.__class__.__name__} does not support get_head_dim',
            action='Please implement this method before using kvcache smooth',
        )

    @abstractmethod
    def get_num_key_value_groups(self) -> int:
        raise UnsupportedError(
            f'{self.__class__.__name__} does not support get_num_key_value_groups',
            action='Please implement this method before using kvcache smooth',
        )

    @abstractmethod
    def get_num_key_value_heads(self) -> int:
        raise UnsupportedError(
            f'{self.__class__.__name__} does not support get_num_key_value_heads',
            action='Please implement this method before using kvcache smooth',
        )
