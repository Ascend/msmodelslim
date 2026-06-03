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

import torch

import msmodelslim.ir as qir
from msmodelslim.ir.qal import QABCRegistry, QDType, QStorage, QParam, QScheme
from msmodelslim.ir.api import quantize, dequantize
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks
from ..base import AutoWeightQuantizer, QConfig


@QABCRegistry.multi_register(dispatch_key=[(qir.mxfp4_per_block_sym, "fouroversix")], abc_type=AutoWeightQuantizer)
@logger_setter()
class WeightFouroverSixQuantizer(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        self.axes = config.ext.get('axes', -1)
        self.block_size = config.dtype.mx_finfo.block_size
        self.w_q_param = QParam(
            scheme=QScheme(
                dtype=self.config.dtype,
                scope=self.config.scope,
                symmetric=self.config.symmetric,
            ),
            ext={"scale": None, "axes": self.axes, "block_size": self.block_size},
        )
        self.w_q_storage: QStorage = None

    def forward(self, x=None) -> torch.Tensor:
        quantize_weight = self.w_q_storage.value
        axes = self.axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [x + quantize_weight.ndim if x < 0 else x for x in axes]
        quantize_weight, axes_, orig_shape, padded_shape = reshape_to_blocks(quantize_weight, axes, self.block_size)
        dequant_value = dequantize(QStorage(self.config.dtype, quantize_weight), self.w_q_param).value
        dequant_value = undo_reshape_to_blocks(dequant_value, padded_shape, orig_shape, axes)
        return dequant_value

    def init_weight(self, weight: QStorage, bias: torch.Tensor = None) -> None:
        logger = get_logger()
        logger.info("Starting fouroversix algorithm")

        weight_tensor = weight.value.detach()
        axes = self.axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [x + weight_tensor.ndim if x < 0 else x for x in axes]

        for axis in axes:
            if weight_tensor.shape[axis] % self.block_size != 0:
                raise ValueError(
                    f"Weight dim {axis} (size={weight_tensor.shape[axis]}) "
                    f"must be divisible by block_size={self.block_size}"
                )

        weight_tensor, axes, orig_shape, padded_shape = reshape_to_blocks(weight_tensor, axes, self.block_size)

        max_per_block = weight_tensor.abs().max(axis=-1).values

        # 方案A: Scale to 6
        q_param_a, scale_E_a, quantized_weights_a, dequantized_weights_a = self.__quantize_with_scale(
            max_per_block, weight_tensor, 6.0
        )

        # 方案B: Scale to 4
        q_param_b, scale_E_b, quantized_weights_b, dequantized_weights_b = self.__quantize_with_scale(
            max_per_block, weight_tensor, 4.0
        )

        # 逐块计算MSE, 选择MSE较小的方案
        mse_a = torch.mean((weight_tensor - dequantized_weights_a) ** 2, dim=-1)
        mse_b = torch.mean((weight_tensor - dequantized_weights_b) ** 2, dim=-1)
        mask = mse_a <= mse_b

        # 计算选择缩放到4的比例
        total_blocks = mask.numel()
        num_selected_B = torch.sum(~mask).item()
        ratio_B = num_selected_B / (total_blocks * 1.0)
        percentage_B = ratio_B * 100
        logger.info("Percentage of blocks selected for Scale-to-4: %s%%", percentage_B)

        mask = mask.unsqueeze(-1)
        selected_scale = torch.where(mask, scale_E_a, scale_E_b)
        mask = mask.expand_as(weight_tensor)
        reshape_quantized_weights = torch.where(mask, quantized_weights_a, quantized_weights_b)
        quantized_weights = undo_reshape_to_blocks(reshape_quantized_weights, padded_shape, orig_shape, axes)

        # 更新最终的 QParam 和 QStorage
        self.w_q_param.ext.update({'scale': selected_scale})
        self.w_q_storage = QStorage(self.config.dtype, value=quantized_weights)

    def get_q_storage(self) -> QStorage:
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        return self.w_q_param

    def __nearest_neighbor_rounding_to_e8m0(self, scale: torch.Tensor):
        mask_invalid = scale <= 0.0
        scale = torch.where(mask_invalid, torch.ones_like(scale), scale)

        log2_scale = torch.log2(scale)

        int_part = torch.floor(log2_scale)
        fractional_part = log2_scale - int_part

        round_up = int_part + 1
        round_down = int_part
        ties = torch.where(int_part % 2 == 0, round_up, round_down)

        eps = 1e-6
        is_gt = fractional_part > 0.5 + eps
        is_lt = fractional_part < 0.5 - eps
        is_tie = torch.abs(fractional_part - 0.5) <= eps

        scale_e = torch.where(is_gt, round_up, torch.where(is_lt, round_down, torch.where(is_tie, ties, round_down)))

        return scale_e

    def __quantize_with_scale(self, max_per_block: torch.Tensor, weight_tensor: torch.Tensor, ceil: float):
        scale = max_per_block / ceil
        scale_E = self.__nearest_neighbor_rounding_to_e8m0(scale)
        scale_E = scale_E.unsqueeze(-1)
        q_param = QParam(
            scheme=QScheme(
                dtype=self.config.dtype,
                scope=self.config.scope,
                symmetric=self.config.symmetric,
            ),
            ext={"scale": scale_E, "axes": self.axes, "block_size": self.block_size},
        )
        quantized_weights = quantize(QStorage(QDType.FLOAT, weight_tensor), q_param).value
        dequantized_weights = dequantize(QStorage(self.config.dtype, quantized_weights), q_param).value
        return q_param, scale_E, quantized_weights, dequantized_weights
