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


@ModelAdapterRegistry.register("hunyuan_video")
class HunyuanVideoAdapter(ModelAdapter):
    def get_norm_linear_subgraph(self,
                                 cfg: 'AntiOutlierConfig',
                                 dummy_input: Optional[torch.Tensor] = None,
                                 norm_class: Optional[List[Type[nn.Module]]] = None):
        norm_linear = {}
        double_layer_num = self.model.config.mm_double_blocks_depth
        if not isinstance(double_layer_num, int):
            raise TypeError("mm_double_blocks_depth must be an integer.")
        if double_layer_num < 1 or double_layer_num > 999:
            raise ValueError("mm_double_blocks_depth must be in the range 1 to 999.")

        for layer in range(double_layer_num):
            input_layernorm = str(layer) + 'img_qkv_anti'
            img_qkv_proj = 'double_blocks.' + str(layer) + '.img_attn_qkv'
            norm_linear[input_layernorm] = [img_qkv_proj]

            input_layernorm = str(layer) + 'txt_qkv_anti'
            txt_qkv_proj = 'double_blocks.' + str(layer) + '.txt_attn_qkv'
            norm_linear[input_layernorm] = [txt_qkv_proj]

        return norm_linear

    def modify_smooth_args(self,
                           cfg: 'AntiOutlierConfig',
                           norm_name: str,
                           linear_names: str,
                           args: List[Any],
                           kwargs: Dict[str, Any]):
        # 针对该模型进行m4量化时，需要对特定层开启偏移
        if cfg.anti_method == 'm4':
            kwargs['is_shift'] = False
            kwargs['alpha'] = cfg.alpha
        return args, kwargs
