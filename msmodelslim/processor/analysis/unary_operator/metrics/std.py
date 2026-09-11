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

from typing import Any, Callable, Dict, List

import torch
from torch import nn

from msmodelslim.processor.analysis.methods_base import AnalysisTargetMatcher
from .base import UnaryAnalysisMethod


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


class StdAnalysisMethod(UnaryAnalysisMethod, AnalysisTargetMatcher):
    """Std 分析方法。"""

    @property
    def name(self) -> str:
        return "std"

    @property
    def supports_distributed(self) -> bool:
        return True

    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """Compute std-based score for the layer"""
        t_max = _as_float(layer_data['t_max'])
        t_min = _as_float(layer_data['t_min'])
        abs_max = max(abs(t_max), abs(t_min))

        # 防止除零：如果标准差为0，返回abs_max或0
        std_value = _as_float(layer_data['std'])
        if std_value == 0:
            return abs_max if abs_max > 0 else 0.0

        return abs_max / std_value

    def pack_stats_for_distributed_merge(self, layer_data: Dict[str, Any]) -> Dict[str, Any]:
        # Match within-rank aggregation: max(t_max), min(t_min), max(std).
        return {
            "t_max": _as_float(layer_data["t_max"]),
            "t_min": _as_float(layer_data["t_min"]),
            "std": _as_float(layer_data["std"]),
        }

    def merge_distributed_stats(self, packed_stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "t_max": max(s["t_max"] for s in packed_stats_list),
            "t_min": min(s["t_min"] for s in packed_stats_list),
            "std": max(s["std"] for s in packed_stats_list),
        }

    def get_hook(self) -> Callable:
        def activation_hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]

            tensor_float = input_tensor.float()
            hidden_dim = tensor_float.shape[-1]
            reshaped = tensor_float.reshape(-1, hidden_dim).detach()

            tensor_max = torch.max(reshaped, dim=0)[0]
            tensor_min = torch.min(reshaped, dim=0)[0]

            if layer_name not in stats_dict:
                stats_dict[layer_name] = {}

            # Store shift (center point)
            stats_dict[layer_name]['shift'] = (tensor_max + tensor_min) / 2

            # Update global max/min
            global_max = torch.max(reshaped)
            global_min = torch.min(reshaped)

            if 't_max' in stats_dict[layer_name]:
                stats_dict[layer_name]['t_max'] = torch.max(stats_dict[layer_name]['t_max'], global_max)
                stats_dict[layer_name]['t_min'] = torch.min(stats_dict[layer_name]['t_min'], global_min)
            else:
                stats_dict[layer_name]['t_max'] = global_max
                stats_dict[layer_name]['t_min'] = global_min

            # Update standard deviation
            tensor_std = torch.std(reshaped - stats_dict[layer_name]['shift'])
            if 'std' in stats_dict[layer_name]:
                stats_dict[layer_name]['std'] = torch.max(stats_dict[layer_name]['std'], tensor_std)
            else:
                stats_dict[layer_name]['std'] = tensor_std

        return activation_hook

    def _matches(self, module: nn.Module) -> bool:
        return isinstance(
            module,
            (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear, nn.Conv2d),
        )
