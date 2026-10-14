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

from typing import Annotated, List, Optional, Union

import torch
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

import msmodelslim.ir as qir
from msmodelslim.ir.api import quantize, dequantize
from msmodelslim.ir.qal import QABCRegistry, QStorage, QParam, QScheme, QScope, QDType
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.utils.exception import SpecError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks
from ..base import AutoWeightQuantizer, QConfig


# ceil_x: MXFP4 per-block weight-only 量化方法。
# ceil_x: MXFP4 per-block weight-only 量化方法。
#
# 算法原理：
# 对于每个 block（block_size=32），计算 shared exponent（E8M0 格式）：
#
# s = ceil(log2(max(|x|) / c + 9.6e-7)) - emax
#
# 其中 c 是可配置的 ceil_x_value（默认 7.25，取值范围 [6.0, 12.0]），
# emax = 2^(ebits-1) = 2（MXFP4 的 ebits=2）。
#
# 量化/反量化过程：
# x_q = sign(x) * round(|x| / 2^s * 2^man_bits) / 2^man_bits * 2^private_exp
# x_dq = x_q * 2^s
#
# 其中 private_exp = clip(floor(log2(|x| / 2^s)), min_exp, emax)，
# man_bits = mbits - 2 = 1（MXFP4 的 mbits=3）。
#
# 设计特点：
# - ceil_x_value 作为除数，控制 shared exponent 的"收紧"程度：
# c 越大 → shared_exp 越小 → 量化步长越小 → 精度越高，但需防范溢出。
# 默认值 7.25 在 W4A4 MXFP4 对称量化中经验最优。
# - 支持 enable_search 模式：在 [search_min, search_max] 范围内
# 按 search_step 步长搜索使 MSE 最小的 ceil_x_value。
# """


class CeilXExtConfig(BaseModel):
    """Structured config for ceil_x quantization method, parsed from QConfig.ext."""

    model_config = {"extra": "forbid"}

    axes: Union[int, List[int]] = -1
    ceil_x_value: Annotated[float, Field(ge=6.0, le=12.0)] = 7.25
    enable_search: bool = False
    search_min: Annotated[float, Field(ge=6.0, le=12.0)] = 6.0
    search_max: Annotated[float, Field(ge=6.0, le=12.0)] = 12.0
    search_step: Annotated[float, Field(gt=0)] = 0.25

    @model_validator(mode="after")
    def validate_search_range(self) -> Self:
        if self.enable_search:
            if self.search_max <= self.search_min:
                raise ValueError(f"search_max ({self.search_max}) must be greater than search_min ({self.search_min})")
        return self


def ceil_x_qparam(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    ceil_x_value: float,
    q_dtype: QDType,
    q_scope: QScope,
    symmetric: bool,
    axes: object,
) -> QParam:
    """Compute MXFP4 per-block quantization param with ceil_x applied to max_val."""
    mx_finfo = q_dtype.mx_finfo
    shared_exp = torch.ceil(torch.log2((max_val / ceil_x_value).clamp(min=0) + 9.6e-7))
    scale_emax = 2 ** (mx_finfo.scale_bits - 1) - 1
    shared_exp = torch.clip(shared_exp, -scale_emax - mx_finfo.emax, scale_emax - mx_finfo.emax)
    return QParam(
        scheme=QScheme(dtype=q_dtype, scope=q_scope, symmetric=symmetric),
        ext={"scale": shared_exp},
    )


def ceil_x_search_best(
    weight_value: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    q_dtype: QDType,
    q_scope: QScope,
    symmetric: bool,
    axes: object,
    search_min: float = 6.0,
    search_max: float = 12.0,
    search_step: float = 0.25,
) -> float:
    """Search for the best ceil_x_value that minimizes MSE in the given range."""
    num_steps = int(round((search_max - search_min) / search_step)) + 1
    candidates = [search_min + i * search_step for i in range(num_steps)]

    best_mse = float("inf")
    best_value = search_min

    for value in candidates:
        q_param = ceil_x_qparam(
            min_val=min_val,
            max_val=max_val,
            ceil_x_value=value,
            q_dtype=q_dtype,
            q_scope=q_scope,
            symmetric=symmetric,
            axes=axes,
        )
        q_storage = quantize(QStorage(QDType.FLOAT, weight_value), q_param)
        recon = dequantize(q_storage, q_param).value
        mse = ((weight_value - recon) ** 2).mean().item()

        if mse < best_mse:
            best_mse = mse
            best_value = value

    get_logger().debug(
        "ceil_x search: best value=%.2f (MSE=%.8f), range=[%.1f, %.1f] step=%.2f",
        best_value,
        best_mse,
        search_min,
        search_max,
        search_step,
    )
    return best_value


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.mxfp4_per_block_sym, "ceil_x"),
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter()
class MXWeightPerBlockCeilX(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()

        if config.dtype != QDType.MXFP4:
            raise SpecError(
                f"ceil_x only supports MXFP4, got {config.dtype}",
                action="Please use dtype: mxfp4 for ceil_x method",
            )

        self.ceil_cfg = CeilXExtConfig.model_validate(config.ext)

        if not isinstance(self.ceil_cfg.axes, (int, list)):
            raise SchemaValidateError(f"Invalid value for 'axes': {self.ceil_cfg.axes}. Expected int or list[int].")
        self.block_size = config.dtype.mx_finfo.block_size

        # observer for computing min/max per block
        minmax_config = MinMaxBlockObserverConfig(axes=self.ceil_cfg.axes)
        self.minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

        self.config = config

        self.weight: Optional[QStorage] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None
        self.w_q_storage_orig: Optional[QStorage] = None
        self.axes: list = []
        self._orig_shape = None
        self._padded_shape = None
        self.is_quantized = False
        self._ceil_x_value: Optional[float] = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.is_quantized:
            if self.weight is None:
                raise SpecError("No weight was set", action="please call init_weight first")
            self._quantize()
            del self.weight
            self.weight = None
            self.is_quantized = True

        dequant_value = dequantize(self.w_q_storage, self.w_q_param).value
        dequant_value = undo_reshape_to_blocks(dequant_value, self._padded_shape, self._orig_shape, self.axes)
        return dequant_value

    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        # 量化延迟到 forward() 首次调用时执行（惰性量化）。
        # 基于 DTS 的 Data-Free Weight-Only 量化通过 invoke forward 触发
        # 量化参数计算；将量化统一放在 _quantize() 中可支持多卡加速场景下
        # 由 forward 统一触发量化入口，避免多卡间逻辑分支重复。
        self.weight = weight

    def _quantize(self) -> None:
        weight_value = self.weight.value.detach()
        axes = self.ceil_cfg.axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [x + weight_value.ndim if x < 0 else x for x in axes]

        weight_value, axes_, orig_shape, padded_shape = reshape_to_blocks(weight_value, axes, self.block_size)
        self._orig_shape = orig_shape
        self._padded_shape = padded_shape
        self.axes = axes

        shared_exp_axes = [x + 1 for x in axes_] if self.block_size > 0 else axes_
        self.minmax_block_observer.update(weight_value, sync=False, shared_exp_axes=shared_exp_axes)
        min_val, max_val = self.minmax_block_observer.get_min_max()

        # ceil_x search: find best value by MSE if enabled
        # 搜索结果存入独立的 _ceil_x_value，不修改配置对象以保留用户原始配置
        if self.ceil_cfg.enable_search:
            self._ceil_x_value = ceil_x_search_best(
                weight_value,
                min_val,
                max_val,
                q_dtype=QDType(self.config.dtype),
                q_scope=QScope(self.config.scope),
                symmetric=self.config.symmetric,
                axes=self.ceil_cfg.axes,
                search_min=self.ceil_cfg.search_min,
                search_max=self.ceil_cfg.search_max,
                search_step=self.ceil_cfg.search_step,
            )

        self.w_q_param = ceil_x_qparam(
            min_val=min_val,
            max_val=max_val,
            ceil_x_value=self._ceil_x_value if self._ceil_x_value is not None else self.ceil_cfg.ceil_x_value,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
            axes=self.ceil_cfg.axes,
        )
        self.w_q_param.ext["axes"] = self.ceil_cfg.axes
        self.w_q_storage = quantize(QStorage(QDType.FLOAT, weight_value), self.w_q_param)
        # keep w_q_storage in block shape for forward(); store original shape separately
        w_q_storage_orig_value = undo_reshape_to_blocks(
            self.w_q_storage.value.clone(), self._padded_shape, self._orig_shape, axes
        )
        self.w_q_storage_orig = QStorage(self.w_q_storage.dtype, w_q_storage_orig_value)

    def get_q_storage(self) -> QStorage:
        if self.w_q_storage is None:
            _ = self.forward(None)
        return self.w_q_storage_orig

    def get_q_param(self) -> QParam:
        if self.w_q_param is None:
            _ = self.forward(None)
        return self.w_q_param

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True
