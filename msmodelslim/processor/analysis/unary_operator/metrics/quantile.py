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

from typing import Any, Callable, Dict

import torch
from torch import nn

from msmodelslim.processor.analysis.methods_base import AnalysisTargetMatcher
from .base import UnaryAnalysisMethod


class QuantileAnalysisMethod(UnaryAnalysisMethod, AnalysisTargetMatcher):
    """Quantile 分析方法。"""

    def __init__(self, sample_step: int = 100):
        self.sample_step = sample_step

    @property
    def name(self) -> str:
        return "quantile"

    @property
    def supports_distributed(self) -> bool:
        return True

    @staticmethod
    def get_quantile_score(act: torch.Tensor, device: torch.device):
        """Helper method to compute quantile score from activation tensor"""
        act_max = torch.max(torch.abs(act))
        act = act.to(device)
        sorted_act = torch.sort(act)[0]
        sorted_act = sorted_act.to("cpu")
        number = len(sorted_act)
        number_1_4 = number // 4
        number_3_4 = number_1_4 * 3
        range_param = 2 * act_max / 254 / (sorted_act[number_3_4] - sorted_act[number_1_4] + 1e-10)
        return range_param.item()

    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """Compute quantile score for the layer"""
        tensor_data = torch.cat(layer_data['tensor']).view(-1).float()
        device = layer_data['device']
        score = QuantileAnalysisMethod.get_quantile_score(tensor_data, device)
        return score

    def get_hook(self) -> Callable:
        """Get hook function for collecting activation data."""

        def activation_hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]

            # Flatten and sort the input tensor
            flattened = input_tensor.reshape(-1)
            sorted_tensor = torch.sort(flattened)[0]

            # Sample the tensor
            if sorted_tensor.numel() < self.sample_step:
                sampled = sorted_tensor
            else:
                sampled = sorted_tensor[self.sample_step // 2 :: self.sample_step].view(-1, 1)

            # Store data
            if layer_name not in stats_dict:
                stats_dict[layer_name] = {'tensor': [sampled.to('cpu')], 'device': input_tensor.device}
            else:
                stats_dict[layer_name]['tensor'].append(sampled.to('cpu'))

        return activation_hook

    def _matches(self, module: nn.Module) -> bool:
        return isinstance(
            module,
            (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear, nn.Conv2d),
        )
