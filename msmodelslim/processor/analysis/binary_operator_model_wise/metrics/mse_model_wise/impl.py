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

from msmodelslim.processor.analysis.binary_operator_model_wise.metrics.base import (
    ModelWiseAnalysisMethod,
)
from msmodelslim.utils.exception import UnsupportedError

from .block_data import resolve_mse_model_wise_block_data


class MSEModelWiseAnalysisMethod(ModelWiseAnalysisMethod):
    """模型输出 MSE：双路径 batch 与输出合并由 Processor 完成。"""

    def __init__(self, adapter: object = None):
        self._block_data = resolve_mse_model_wise_block_data(adapter)

    @property
    def name(self) -> str:
        return "mse_model_wise"

    @property
    def supports_distributed(self) -> bool:
        return True

    def _to_tensor(self, item: Any) -> Optional[torch.Tensor]:
        try:
            return self._block_data.extract_hidden_states(item)
        except UnsupportedError:
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
        """根据两组输出计算分数"""
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
