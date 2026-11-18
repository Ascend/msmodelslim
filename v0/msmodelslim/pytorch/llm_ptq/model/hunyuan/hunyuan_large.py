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


from typing import List, Any, Dict, Optional, Type, TYPE_CHECKING

import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.model.base import ModelAdapter, ModelAdapterRegistry

if TYPE_CHECKING:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig


@ModelAdapterRegistry.register("hunyuan")
class HunyuanLargeAdapter(ModelAdapter):
    def get_norm_linear_subgraph(self,
                                 cfg: 'AntiOutlierConfig',
                                 dummy_input: Optional[torch.Tensor] = None,
                                 norm_class: Optional[List[Type[nn.Module]]] = None):
        norm_linear = {}
        layer_num = self.model.config.num_hidden_layers

        # 校验layer_num是否过大或过小
        if layer_num < 1 or layer_num > 999:
            raise ValueError(f"The number of hidden layers({layer_num}) is invalid. It must be between 1 and 999.")

        for layer in range(layer_num):
            input_layernorm = 'model.layers.' + str(layer) + '.input_layernorm'
            if layer % 2 != 0:
                q_proj = 'model.layers.' + str(layer) + '.self_attn.q_proj'
                norm_linear[input_layernorm] = [q_proj]
            else:
                q_proj = 'model.layers.' + str(layer) + '.self_attn.q_proj'
                k_proj = 'model.layers.' + str(layer) + '.self_attn.k_proj'
                v_proj = 'model.layers.' + str(layer) + '.self_attn.v_proj'
                norm_linear[input_layernorm] = [q_proj, k_proj, v_proj]

        return norm_linear

    def modify_smooth_args(self,
                           cfg: 'AntiOutlierConfig',
                           norm_name: str,
                           linear_names: str,
                           args: List[Any],
                           kwargs: Dict[str, Any]):
        # 针对该模型进行m4量化时，需要对特定层开启偏移
        if cfg.anti_method == 'm4':
            is_shift = False
            if 'norm' in norm_name:
                is_shift = True

            kwargs['is_shift'] = is_shift
            kwargs['alpha'] = cfg.alpha
        return args, kwargs
