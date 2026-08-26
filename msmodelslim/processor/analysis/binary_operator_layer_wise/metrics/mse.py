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

from typing import Any, List, Optional

import torch

from msmodelslim.processor.analysis.binary_operator_layer_wise.metrics.base import LayerWiseAnalysisMethod


class MSELayerWiseAnalysisMethod(LayerWiseAnalysisMethod):
    """mse_layer_wise 指标实现：对比同一 block 的 float/quant 输出 MSE。"""

    def __init__(self, adapter: object = None):
        _ = adapter

    @property
    def name(self) -> str:
        return "mse_layer_wise"

    @property
    def supports_distributed(self) -> bool:
        return True

    @staticmethod
    def _to_tensor(item: Any) -> Optional[torch.Tensor]:
        t = item
        while isinstance(t, (list, tuple)):
            if not t:
                return None
            t = t[0]
        return t if isinstance(t, torch.Tensor) else None

    def compute_score(
        self,
        ref_outputs: List[Any],
        cand_outputs: List[Any],
    ) -> float:
        losses: List[torch.Tensor] = []
        for ref_out, cand_out in zip(ref_outputs, cand_outputs):
            ref_t = self._to_tensor(ref_out)
            cand_t = self._to_tensor(cand_out)
            if ref_t is None or cand_t is None:
                continue
            losses.append(
                torch.nn.functional.mse_loss(
                    ref_t.detach().float().cpu(),
                    cand_t.detach().float().cpu(),
                )
            )

        return torch.stack(losses).mean().item() if losses else 0.0
