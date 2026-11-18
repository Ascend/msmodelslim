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


from typing import List, Optional, Type, TYPE_CHECKING

import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.model.base import ModelAdapter, ModelAdapterRegistry

if TYPE_CHECKING:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig


@ModelAdapterRegistry.register("glm4v")
class GLM41VAdapter(ModelAdapter):
    def get_norm_linear_subgraph(self,
                                 cfg: 'AntiOutlierConfig',
                                 dummy_input: Optional[torch.Tensor] = None,
                                 norm_class: Optional[List[Type[nn.Module]]] = None):
        norm_linear = {}
        text_num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if not isinstance(text_num_hidden_layers, int):
            raise TypeError("num_hidden_layers in text_config must be an integer.")
        if text_num_hidden_layers < 1 or text_num_hidden_layers > 999:
            raise ValueError("num_hidden_layers in text_config must be in the range 1 to 999.")

        for layer in range(text_num_hidden_layers):
            input_layernorm = 'model.language_model.layers.' + str(layer) + '.input_layernorm'
            q_proj = 'model.language_model.layers.' + str(layer) + '.self_attn.q_proj'
            k_proj = 'model.language_model.layers.' + str(layer) + '.self_attn.k_proj'
            v_proj = 'model.language_model.layers.' + str(layer) + '.self_attn.v_proj'
            norm_linear[input_layernorm] = [q_proj, k_proj, v_proj]

            post_layernorm = 'model.language_model.layers.' + str(layer) + '.post_attention_layernorm'
            gate_up_proj = 'model.language_model.layers.' + str(layer) + '.mlp.gate_up_proj'

            norm_linear[post_layernorm] = [gate_up_proj]

        return norm_linear
