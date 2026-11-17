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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

from torch import nn
import torch

from msmodelslim.core.quantization_schema.qal.qbase import QStorage, QDType
from msmodelslim.core.quantizer.base import QConfig, AutoWeightQuantizer
from msmodelslim.utils.logging import get_logger

if TYPE_CHECKING:
    from msmodelslim.core.quantization_algorithm.anti_outlier.awq.common import AWQContext


class AWQSearcher(ABC):
    """Abstract base class for AWQ alpha searchers."""

    @abstractmethod
    def search(
        self,
        linears2scale: List[nn.Linear],
        context: AWQContext,
    ) -> Optional[torch.Tensor]:
        """Search for the best scales given the linears to scale and the context.

        Args:
            linears2scale: List of nn.Linear modules whose weights will be scaled and quantized during the search.
            context: AWQContext containing necessary information and statistics for the search.

        Returns:
            A tensor of best scales for each output channel, or None if search fails.
        """
        pass


class AWQBestScalesSearcher(AWQSearcher):
    def __init__(
        self,
        weight_qconfig: QConfig,
        n_grid: int = 20,
    ):
        self.n_grid = n_grid
        self.weight_qconfig = weight_qconfig

    @torch.no_grad()
    def search(
        self,
        linears2scale: List[nn.Linear],
        context: AWQContext,
    ) -> Optional[torch.Tensor]:
        if not linears2scale:
            raise ValueError("linears2scale must not be empty for alpha search")
        act_mean = context.act_mean
        module2inspect = context.inspect_module
        module2inspect_args = context.inspect_module_args

        if len(module2inspect_args) == 0:
            raise ValueError("block_kwargs_cache is empty, need at least one batch")

        device = next(module2inspect.parameters()).device
        act_mean = act_mean.to(device=device)

        org_sd = {k: v.detach().clone() for k, v in module2inspect.state_dict().items()}
        module2inspect.eval()
        golden_outputs = self._run_samples(module2inspect, module2inspect_args)

        best_scales = None
        best_loss = float("inf")

        for g_idx in range(self.n_grid):
            ratio = g_idx * 1 / self.n_grid
            scales = self._compute_scales(act_mean, ratio)
            self._apply_candidate_scales(linears2scale, scales)
            candidate_outputs = self._run_samples(module2inspect, module2inspect_args)

            loss = self._compute_loss(candidate_outputs, golden_outputs)
            if loss < best_loss:
                best_loss = loss
                best_scales = scales.clone()

            module2inspect.load_state_dict(org_sd, strict=True)

        get_logger().debug(
            "AWQ scales search: best_scales=%s, best_loss=%.8f, n_grid=%d",
            best_scales,
            best_loss,
            self.n_grid,
        )

        return best_scales

    @torch.no_grad()
    def _run_samples(self, block: nn.Module, block_kwargs_cache: list) -> List[torch.Tensor]:
        from msmodelslim.core.quantization_algorithm.anti_outlier.awq.common import onload

        device = next(block.parameters()).device
        outputs = [block(**onload(kwargs, device)) for kwargs in block_kwargs_cache]
        return [output[0] if isinstance(output, tuple) else output for output in outputs]

    def _compute_scales(self, act_mean: torch.Tensor, ratio: float) -> torch.Tensor:
        scales = act_mean.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        scales[torch.isinf(scales)] = 1
        scales[torch.isnan(scales)] = 1
        return scales

    def _apply_candidate_scales(self, linears: List[nn.Linear], scales: torch.Tensor) -> None:
        for fc in linears:
            s = scales.to(device=fc.weight.device, dtype=fc.weight.dtype)
            fc.weight.mul_(s.view(1, -1))
            quantizer = AutoWeightQuantizer.from_config(self.weight_qconfig)
            quantizer.init_weight(weight=QStorage(dtype=QDType.FLOAT, value=fc.weight.data))
            fc.weight.data = quantizer(fc.weight.data) / s.view(1, -1)

    def _compute_loss(self, candidate_outputs: List[torch.Tensor], golden_outputs: List[torch.Tensor]) -> float:
        total_loss = 0.0
        n_elements = 0
        for cand_out, gold_out in zip(candidate_outputs, golden_outputs):
            total_loss += torch.nn.functional.mse_loss(cand_out, gold_out, reduction='sum').item()
            n_elements += cand_out.numel()
        return total_loss / n_elements if n_elements > 0 else float('inf')
