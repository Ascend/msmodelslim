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

from typing import Optional

import torch
from pydantic import validate_call

import msmodelslim.core.quantization_schema as qir
from msmodelslim.core.quantization_schema.api import quantize, dequantize
from msmodelslim.core.quantization_schema.qal import QABCRegistry, QDType, QStorage, QParam, QScope, QScheme
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.core.quantization_schema.utils import reshape_to_blocks, undo_reshape_to_blocks
from msmodelslim.utils.exception import SpecError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter
from ..base import AutoWeightQuantizer, QConfig

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.mxfp8_per_block_sym, "mse_round"),
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter()
class MXWeightPerBlockMseRound(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None

        self.axes = config.ext.get("axes", -1)
        if not isinstance(self.axes, (int, list)):
            raise SchemaValidateError(f"Invalid value for 'axes': {self.axes}. Expected int or list[int].")
        self.block_size = config.dtype.mx_finfo.block_size
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None

    def _build_qparam(self, shared_exp: torch.Tensor) -> QParam:
        mx_finfo = QDType(self.config.dtype).mx_finfo
        shared_exp = shared_exp.clone()
        keep_mask = (shared_exp > -FP32_EXPONENT_BIAS) if mx_finfo.flush_fp32_subnorms else None

        scale_emax = 2 ** (mx_finfo.scale_bits - 1) - 1
        shared_exp[shared_exp > scale_emax] = float("NaN")
        shared_exp[shared_exp < -scale_emax] = -scale_emax

        return QParam(
            scheme=QScheme(
                dtype=QDType(self.config.dtype),
                scope=QScope(self.config.scope),
                symmetric=self.config.symmetric,
            ),
            ext={
                "scale": shared_exp,
                "offset": torch.zeros_like(shared_exp),
                "keep_mask": keep_mask,
            },
        )

    @staticmethod
    def _select_shared_exp_by_mse(
        mse_up: torch.Tensor, mse_down: torch.Tensor, shared_exp_up: torch.Tensor, shared_exp_down: torch.Tensor
    ) -> torch.Tensor:
        valid_up = torch.isfinite(mse_up)
        valid_down = torch.isfinite(mse_down)
        select_up = valid_up & ((mse_up < mse_down) | ~valid_down)
        return torch.where(select_up, shared_exp_up, shared_exp_down)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.w_q_storage is None or self.w_q_param is None:
            raise SpecError("No quantized weight was set", action="please call init_weight first")
        dequant_value = dequantize(self.w_q_storage, self.w_q_param).value
        return dequant_value

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        self.weight = weight
        self.bias = bias

        minmax_config = MinMaxBlockObserverConfig(axes=self.axes)
        minmax_block_observer = MsMinMaxBlockObserver(minmax_config)
        weight_value = weight.value.detach()
        axes = self.axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [x + weight_value.ndim if x < 0 else x for x in axes]

        weight_value, axes_, orig_shape, padded_shape = reshape_to_blocks(weight_value, axes, self.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.block_size > 0 else axes_

        minmax_block_observer.update(weight_value, sync=self.sync, shared_exp_axes=shared_exp_axes)
        _, max_val = minmax_block_observer.get_min_max()

        mx_finfo = QDType(self.config.dtype).mx_finfo
        log_arg = max_val + FP32_MIN_NORMAL * (max_val == 0).to(max_val.dtype)
        log2v = torch.log2(log_arg)
        shared_exp_up = torch.ceil(log2v) - mx_finfo.emax
        shared_exp_down = torch.floor(log2v) - mx_finfo.emax

        q_param_up = self._build_qparam(shared_exp_up)
        q_param_down = self._build_qparam(shared_exp_down)

        float_storage = QStorage(QDType.FLOAT, weight_value)
        dequant_up = dequantize(quantize(float_storage, q_param_up), q_param_up).value
        dequant_down = dequantize(quantize(float_storage, q_param_down), q_param_down).value

        mse_up = (weight_value - dequant_up).pow(2).mean(dim=-1, keepdim=True)
        mse_down = (weight_value - dequant_down).pow(2).mean(dim=-1, keepdim=True)
        shared_exp = self._select_shared_exp_by_mse(
            mse_up, mse_down, q_param_up.ext["scale"], q_param_down.ext["scale"]
        )

        self.w_q_param = self._build_qparam(shared_exp)
        self.w_q_param.ext["axes"] = self.axes
        self.w_q_storage = quantize(float_storage, self.w_q_param)
        self.w_q_storage.value = undo_reshape_to_blocks(self.w_q_storage.value, padded_shape, orig_shape, axes)

    def get_q_storage(self) -> QStorage:
        if self.w_q_storage is None:
            raise SpecError("No quantized weight was set", action="please call init_weight first")
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if self.w_q_param is None:
            raise SpecError("No q_param was set", action="please call init_weight first")
        return self.w_q_param
