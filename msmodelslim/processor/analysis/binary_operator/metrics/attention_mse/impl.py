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

from typing import Dict, Any, Callable

import torch
from torch import nn

from msmodelslim.processor.analysis.methods_base import AnalysisTargetMatcher
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import UnexpectedError, UnsupportedError
from .interface import AttentionMSEAnalysisInterface
from ..base import BinaryAnalysisMethod

logger = get_logger()


class AttentionMSEAnalysisMethod(BinaryAnalysisMethod, AnalysisTargetMatcher):
    """AttentionMSE 分析方法，需要模型适配器实现 AttentionMSEAnalysisInterface"""

    def __init__(self, adapter: object):
        if not isinstance(adapter, AttentionMSEAnalysisInterface):
            raise UnsupportedError(
                f'{adapter.__class__.__name__} does not implement AttentionMSEAnalysisInterface',
                action=f'Please ensure {adapter.__class__.__name__} inherits from AttentionMSEAnalysisInterface '
                f'and implements the methods defined by the interface',
            )
        self.adapter = adapter
        self.attention_module_cls = adapter.get_attention_module_cls()

    @property
    def name(self) -> str:
        return "mse"

    @property
    def supports_distributed(self) -> bool:
        return True

    def compute_score(self, layer_data_before: Dict[str, Any], layer_data_after: Dict[str, Any]) -> float:
        # 使用 phase 区分：attn_output["float"] 与 attn_output["quant"] 为等长列表，对应元素同形状 2D tensor，逐对 MSE 取平均
        list_a = layer_data_before.get("attn_output") or []
        list_b = layer_data_after.get("attn_output") or []
        if not list_a or not list_b:
            # 报错未采集到完整的激活状态
            raise UnexpectedError("Missing activation data for MSE analysis.")
        losses = [torch.nn.functional.mse_loss(a, b) for a, b in zip(list_a, list_b)]
        return torch.stack(losses).mean().item()

    def get_hook(self) -> Callable:
        attention_output_extractor = self.adapter.get_attention_output_extractor()

        def activation_hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            attention_output = attention_output_extractor(output_tensor)

            attention_output_float = attention_output.float()
            hidden_dim = attention_output_float.shape[-1]
            reshaped = attention_output_float.reshape(-1, hidden_dim).detach()

            if layer_name not in stats_dict:
                stats_dict[layer_name] = {"attn_output": []}

            stats_dict[layer_name]["attn_output"].append(reshaped.to("cpu"))

        return activation_hook

    def _matches(self, module: nn.Module) -> bool:
        return module.__class__.__name__ == self.attention_module_cls
