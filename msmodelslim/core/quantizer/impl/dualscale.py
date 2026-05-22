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

from typing import Optional
import torch
from pydantic import validate_call

import msmodelslim.ir as qir
from msmodelslim.ir.qal import QABCRegistry, QDType, QStorage, QParam, QScope, QScheme
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks
from ..base import AutoActQuantizer, AutoWeightQuantizer, QConfig

MXFP4_MAX_NORMAL = 6.0


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.mxfp4_dual_scale_sym, "dualscale"),
    ],
    abc_type=AutoWeightQuantizer
)
@logger_setter()
class MXWeightDualScaleMinmax(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()

        self.config = config
        self.axes = config.ext.get('axes', -1)
        self.inner_quant_config = QConfig(
            scope=QScope.PER_BLOCK,
            dtype=self.config.dtype,
            symmetric=self.config.symmetric,
            method="minmax",
            ext={"axes": self.axes}
        )
        self.inner_quantizer = AutoWeightQuantizer.from_config(self.inner_quant_config)

        if not isinstance(self.axes, (int, list)):
            raise SchemaValidateError(
                f"Invalid value for 'axes': {self.axes}. Expected int or list[int]."
            )

        self.dual_block_size = self.config.ext['dual_block_size']

        self.w_q_param = QParam(
            scheme=QScheme(
                dtype=config.dtype,
                scope=config.scope,
                symmetric=config.symmetric,
            ),
            ext={
                "dual_block_size": self.dual_block_size,
                },
        )
        self.w_q_storage: Optional[QStorage] = None

        minmax_config = MinMaxBlockObserverConfig(axes=self.axes)
        self.minmax_block_observer = MsMinMaxBlockObserver(minmax_config)
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        inner_dequant_value = self.inner_quantizer(x)
        dual_scale = self.get_q_param().ext.get('dual_scale', None)
        axes_ = self.w_q_param.ext['axes']

        if dual_scale is None:
            raise UnexpectedError(
                f"The parameter 'dual_scale' cannot be None.",
                action="Please ensure that 'dual_scale' is correctly initialized or passed."
            )

        inner_dequant_value, axes_, orig_shape, padded_shape = reshape_to_blocks(inner_dequant_value, axes_, self.dual_block_size)
        dual_scale_dequant_value = inner_dequant_value.to(torch.float32) * dual_scale
        dual_scale_dequant_value = undo_reshape_to_blocks(dual_scale_dequant_value, padded_shape, orig_shape, axes_)
        return dual_scale_dequant_value

    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        """ 计算逻辑相比普通mxfp， 只是多了一个指定维度，32位的共享scale， mxfp4 之前 A/scale """
        weight_value = weight.value.detach()

        target_axes = self.axes
        target_axes = [target_axes] if isinstance(target_axes, int) else target_axes
        self.target_axes = [x + weight_value.ndim if x < 0 else x for x in target_axes]
        weight_value, axes_, orig_shape, padded_shape = reshape_to_blocks(weight_value, self.target_axes, self.dual_block_size)
        dual_scale_axes = [x + 1 for x in axes_] if self.dual_block_size > 0 else axes_
        self.minmax_block_observer.update(weight_value.to(torch.float32), sync=self.sync, shared_exp_axes=dual_scale_axes)
        dual_block_min_val, dual_block_max_val = self.minmax_block_observer.get_min_max()
        dual_scale = dual_block_max_val / MXFP4_MAX_NORMAL
        dual_scale_weight_value = weight_value.to(torch.float32) / dual_scale
        dual_scale_weight_value = undo_reshape_to_blocks(dual_scale_weight_value, padded_shape, orig_shape, self.target_axes)

        self.inner_quantizer.init_weight(QStorage(QDType.FLOAT, dual_scale_weight_value), bias)
        self.inner_w_q_param = self.inner_quantizer.get_q_param()
        self.inner_w_q_storage = self.inner_quantizer.get_q_storage()
        self.w_q_param.ext.update(self.inner_w_q_param.ext)
        self.w_q_param.ext.update({'dual_scale': dual_scale})
        self.w_q_storage = self.inner_w_q_storage

    def get_q_storage(self) -> QStorage:
        if not self.w_q_storage:
            raise UnexpectedError(
                f"The parameter 'self.w_q_storage' cannot be None.",
                action="Please execute 'self.init_weight' first to initialize the quantization parameters."
            )
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if not self.w_q_param:
            raise UnexpectedError(
                f"The parameter 'self.w_q_param' cannot be None.",
                action="Please execute 'self.init_weight' first to initialize the quantization parameters."
            )
        return self.w_q_param

@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.mxfp4_dual_scale_sym, "dualscale"),
    ],
    abc_type=AutoActQuantizer
)
@logger_setter()
class MXActDualScaleMinmax(AutoActQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        self.axes = config.ext.get('axes', -1)
        if not isinstance(self.axes, (int, list)):
            raise SchemaValidateError(
                f"Invalid value for 'axes': {self.axes}. Expected int or list[int]."
            )
        self.inner_quant_config = QConfig(
            scope=QScope.PER_BLOCK,
            dtype=self.config.dtype,
            symmetric=self.config.symmetric,
            method="minmax",
            ext={"axes": self.axes}
        )
        self.inner_quantizer = AutoActQuantizer.from_config(self.inner_quant_config)

        self.dual_block_size = self.config.ext['dual_block_size']
        
        self.q_param = QParam(
            scheme=QScheme(
                dtype=config.dtype,
                scope=config.scope,
                symmetric=config.symmetric,
            ),
            ext={
                "dual_block_size": self.dual_block_size,
                "axes": self.axes,
            },
        )
        
        minmax_config = MinMaxBlockObserverConfig(axes=self.axes)
        self.minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            return QParam(scheme=self.config.to_scheme())
        return self.q_param

    def is_data_free(self) -> bool:
        """
        mxfp8、mxfp4的dual_scale量化为data free场景
        """
        return True