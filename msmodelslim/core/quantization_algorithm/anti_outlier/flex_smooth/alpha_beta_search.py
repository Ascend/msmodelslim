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
# pylint: disable=arguments-renamed

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch import nn

from msmodelslim.core.quantizer.linear import LinearQuantizer, LinearQConfig
from msmodelslim.utils.logging import get_logger


@torch.no_grad()
def quant_int8sym(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    xmax = torch.abs(x).max(dim=dim, keepdim=True)[0]
    interval = xmax / 127
    quanted = x / interval
    quanted = torch.round(quanted).clip(min=-127, max=127)
    recovered = quanted * interval
    return recovered


@torch.no_grad()
def quant_int8asym(x: torch.Tensor) -> torch.Tensor:
    x_max = torch.max(x)
    x_min = torch.min(x)
    eps = torch.tensor([torch.finfo(torch.float32).eps]).type_as(x_min)
    scale = (x_max - x_min) / 255
    scale = torch.max(scale, eps)
    zero_point = -1 * x_min / scale
    zero_point = zero_point.round() - 128
    qx = (x / scale + zero_point).round()
    recovered = (qx - zero_point) * scale
    return recovered


class BaseAlphaBetaSearcher(ABC):
    @abstractmethod
    def search_alpha(self, act: torch.Tensor, weights, **kwargs) -> Tuple[float, float]:
        pass

    @abstractmethod
    def search_beta(self, act: torch.Tensor, weights, alpha: float, **kwargs) -> Tuple[float, float]:
        pass

    @abstractmethod
    def evaluate_alpha_beta(self, act: torch.Tensor, weights, alpha: float, beta: float, **kwargs) -> float:
        pass


class FlexSmoothAlphaBetaSearcher(BaseAlphaBetaSearcher):
    def __init__(self, act_sym: bool = True, search_step: float = 0.05):
        self.act_sym = act_sym
        self.search_step = search_step

    @torch.no_grad()
    def evaluate_alpha_beta(
        self, act: torch.Tensor, weights: torch.Tensor, alpha: float, beta: float, **kwargs
    ) -> float:
        fp_golden = torch.matmul(act, weights.T)
        normal = torch.mean(fp_golden**2) ** 0.5
        scale = torch.max(torch.abs(act), dim=0, keepdims=True)[0] ** alpha * torch.max(
            torch.abs(weights), dim=0, keepdims=True
        )[0] ** (-beta)

        scaled_act = act / scale
        scaled_weights = weights * scale

        quant_weight = quant_int8sym(scaled_weights)
        if self.act_sym:
            quant_act = quant_int8sym(scaled_act)
        else:
            quant_act = quant_int8asym(scaled_act)

        quant_result = torch.matmul(quant_act, quant_weight.T)
        normal_mse = (torch.mean((torch.abs(quant_result - fp_golden) ** 2)) ** 0.5) / normal

        return normal_mse.item()

    @torch.no_grad()
    def search_alpha(self, act: torch.Tensor, weights: torch.Tensor, **kwargs) -> Tuple[float, float]:
        best_alpha = 0.0
        best_mse = float('inf')

        for alpha in np.round(np.arange(0.0, 1.0 + self.search_step, self.search_step), decimals=2):
            beta = 1.0 - alpha
            mse = self.evaluate_alpha_beta(act, weights, alpha, beta)

            if mse <= best_mse:
                best_mse = mse
                best_alpha = alpha

        get_logger().debug("Found optimal alpha: %.2f, MSE: %.6f", best_alpha, best_mse)
        return best_alpha, best_mse

    @torch.no_grad()
    def search_beta(self, act: torch.Tensor, weights: torch.Tensor, alpha: float, **kwargs) -> Tuple[float, float]:
        best_beta = 0.0
        best_mse = kwargs.get('normal_mse_best', float('inf'))

        for beta in np.round(np.arange(0.0, 1.0 + self.search_step, self.search_step), decimals=2):
            mse = self.evaluate_alpha_beta(act, weights, alpha, beta)

            if mse <= best_mse:
                best_mse = mse
                best_beta = beta

        get_logger().debug("Found optimal beta: %.2f (alpha=%.2f), final MSE: %.6f", best_beta, alpha, best_mse)
        return best_beta, best_mse

    @torch.no_grad()
    def search_alpha_beta(self, act: torch.Tensor, weights: torch.Tensor, **kwargs) -> Tuple[float, float, float]:
        # Phase 1: Search for optimal alpha (beta = 1 - alpha)
        best_alpha, mse_alpha = self.search_alpha(act, weights)

        # Phase 2: Search for optimal beta based on optimal alpha
        best_beta, final_mse = self.search_beta(act, weights, best_alpha, normal_mse_best=mse_alpha)

        return best_alpha, best_beta, final_mse


class FlexAWQSSZAlphaBetaSearcher(BaseAlphaBetaSearcher):
    def __init__(self, qconfig: LinearQConfig, search_step: float = 0.05):
        self.qconfig = qconfig
        self.search_step = search_step

    @torch.no_grad()
    def evaluate_alpha_beta(
        self, act: torch.Tensor, linear: nn.Linear, alpha: float, beta: float = 0.0, **kwargs
    ) -> float:
        """Evaluate the effectiveness of given alpha and beta parameters

        Uses actual quantizer to evaluate parameter effectiveness
        """
        # Save original weight
        original_weight = linear.weight.data.clone()

        try:
            fp_golden = torch.matmul(act, linear.weight.T)
            normal = torch.mean(fp_golden**2) ** 0.5
            scale = torch.max(torch.abs(act), dim=0, keepdims=True)[0] ** alpha
            scaled_act = act / scale
            scaled_w_scale = linear.weight * scale
            linear.weight.data = scaled_w_scale
            quantizer = LinearQuantizer(config=self.qconfig)
            quantizer.setup(linear)
            get_logger().debug(
                "  - Created quantizer with input_quantizer: %s", type(quantizer.input_quantizer).__name__
            )
            get_logger().debug(
                "  - Created quantizer with weight_quantizer: %s", type(quantizer.weight_quantizer).__name__
            )
            # 如果量化器是data free场景，则不进行前向
            if not quantizer.is_data_free():
                _ = quantizer.forward(scaled_act)
            ir_module = quantizer.deploy()

            get_logger().debug("  - Deployed IR module: %s", type(ir_module).__name__)
            with torch.no_grad():
                ir_result = ir_module(scaled_act)
            normal_mse = (torch.mean((torch.abs(ir_result - fp_golden) ** 2)) ** 0.5) / normal

            get_logger().debug("  - Normal MSE: %.6f", normal_mse)
            return normal_mse.item()
        finally:
            # Restore original weight
            linear.weight.data = original_weight

    @torch.no_grad()
    def search_alpha(self, act: torch.Tensor, linear: nn.Linear, **kwargs) -> Tuple[float, float]:
        best_alpha = 0.0
        best_mse = kwargs.get('normal_mse_best', float('inf'))

        for alpha in np.round(np.arange(0.0, 1.0 + self.search_step, self.search_step), decimals=2):
            mse = self.evaluate_alpha_beta(act, linear, alpha, beta=0.0)

            if best_mse is None or mse <= best_mse:
                best_mse = mse
                best_alpha = alpha

        get_logger().debug("Found optimal alpha: %.2f, MSE: %.6f", best_alpha, best_mse)
        return best_alpha, best_mse

    @torch.no_grad()
    def search_beta(self, act: torch.Tensor, linear: nn.Linear, alpha: float, **kwargs) -> Tuple[float, float]:
        # In FlexAWQSSZ, beta is fixed at 0
        best_beta = 0.0
        mse = self.evaluate_alpha_beta(act, linear, alpha, beta=0.0)
        return best_beta, mse
