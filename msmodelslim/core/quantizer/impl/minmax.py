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
from pydantic import validate_call, BaseModel, Field
from typing import Annotated, List, Union

import msmodelslim.core.quantization_schema as qir
from msmodelslim.core.quantization_schema.api import fake_quantize, quantize, dequantize, calculate_qparam
from msmodelslim.core.quantization_schema.qal import QABCRegistry, QDType, QStorage, QParam, QScope, QScheme
from msmodelslim.core.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.exception import SpecError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.core.quantization_schema.utils import (
    reshape_to_blocks,
    undo_reshape_to_blocks,
    pad_and_reshape,
    undo_pad_and_reshape,
)
from ..base import AutoActQuantizer, AutoWeightQuantizer, QConfig


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_tensor_sym, "minmax"),
        (qir.int8_per_tensor_asym, "minmax"),
        (qir.fp8_e4m3_per_tensor_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer,
)
@logger_setter()
class ActPerTensorMinmax(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        aggregation_type = config.ext.get("aggregation_type", "max")
        minmax_config = MinMaxObserverConfig(dim=[], keepdim=False, aggregation_type=aggregation_type)
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.minmax_observer.update(x, self.sync)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            raise SpecError("No q_param was set", action="Please call forward first")
        return self.q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_token_asym, "minmax"),
        (qir.int8_per_token_sym, "minmax"),
        (qir.fp8_e4m3_per_token_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer,
)
@logger_setter()
class ActPerTokenMinmax(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig(dim=[], keepdim=False, aggregation_type="max")
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        self.minmax_observer.reset()
        self.minmax_observer.update(x_reshaped, self.sync)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.reshape(x_shape)

    def is_data_free(self) -> bool:
        """
        激活值按照per_token量化时返回True
        """
        return True

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            return QParam(scheme=self.config.to_scheme())
        return self.q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_sym, "minmax"),
        (qir.int8_per_channel_asym, "minmax"),
        (qir.fp8_e4m3_per_channel_sym, "minmax"),
        (qir.mxfp8_per_channel_sym, "minmax"),
        (qir.fp4_e2m1_per_channel_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer,
)
class ActPerChannelMinmax(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        aggregation_type = config.ext.get("aggregation_type", "max")
        minmax_config = MinMaxObserverConfig(dim=[0], keepdim=False, aggregation_type=aggregation_type)
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        self.minmax_observer.reset()
        self.minmax_observer.update(x_reshaped, self.sync)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.reshape(x_shape)

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            raise RuntimeError("No q_param was set, please call forward first")
        return self.q_param


@QABCRegistry.multi_register(dispatch_key=[(qir.int8_pd_mix_asym, "minmax")], abc_type=AutoActQuantizer)
@logger_setter()
class ActPDMixMinmax(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        aggregation_type = config.ext.get("aggregation_type", "max")
        prefilling_config = MinMaxObserverConfig(dim=[], keepdim=False, aggregation_type="max")
        decoding_config = MinMaxObserverConfig(dim=[], keepdim=False, aggregation_type=aggregation_type)
        self.prefilling_observer = MsMinMaxObserver(prefilling_config)
        self.decoding_observer = MsMinMaxObserver(decoding_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calculate decoding static param
        self.decoding_observer.update(x, self.sync)
        min_val, max_val = self.decoding_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope.PER_TENSOR,
            symmetric=False,
        )

        # calibration as prefilling as dynamic
        x_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        self.prefilling_observer.reset()
        self.prefilling_observer.update(x_reshaped, self.sync)
        min_val, max_val = self.prefilling_observer.get_min_max()
        tmp_q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope.PER_TOKEN,
            symmetric=True,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), tmp_q_param).value.reshape(x_shape)

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            raise SpecError("No q_param was set", action="Please call forward first")
        return QParam(scheme=qir.int8_pd_mix_asym, ext=self.q_param.ext)


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_sym, "minmax"),
        (qir.int4_per_channel_sym, "minmax"),
        (qir.fp8_e4m3_per_channel_sym, "minmax"),
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter()
class WeightPerChannelMinmax(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        minmax_config = MinMaxObserverConfig(dim=[0], keepdim=False, aggregation_type="max")
        self.config = config
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.weight is None:
            raise SpecError("No weight was set", action="please call init_weight first")
        self.minmax_observer.update(self.weight.T.value)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.w_q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        self.w_q_storage = quantize(self.weight.T, self.w_q_param).T
        return dequantize(self.w_q_storage.T, self.w_q_param).T.value

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        self.weight = weight
        self.bias = bias

    def get_q_storage(self) -> QStorage:
        if self.w_q_storage is None:
            _ = self.forward(None)
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if self.w_q_param is None:
            _ = self.forward(None)
        return self.w_q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.mxfp8_per_block_sym, "minmax"),
        (qir.mxfp4_per_block_sym, "minmax"),
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter()
class MXWeightPerBlockMinmax(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()

        self.config = config

        self.axes = config.ext.get('axes', -1)
        if not isinstance(self.axes, (int, list)):
            raise SchemaValidateError(f"Invalid value for 'axes': {self.axes}. Expected int or list[int].")
        self.block_size = config.dtype.mx_finfo.block_size

        # Lazy quantization state: init_weight stores weight, forward triggers _quantize.
        # Unified entry point ensures consistent behavior across single-card and DTS multi-card.
        self.weight: Optional[QStorage] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None
        self.w_q_storage_orig: Optional[QStorage] = None
        self._orig_shape = None
        self._padded_shape = None
        self._axes = None
        self.is_quantized = False

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.is_quantized:
            if self.weight is None:
                raise SpecError("No weight was set", action="please call init_weight first")
            self._quantize()
            self.is_quantized = True

        dequant_value = dequantize(self.w_q_storage, self.w_q_param).value
        dequant_value = undo_reshape_to_blocks(dequant_value, self._padded_shape, self._orig_shape)
        return dequant_value

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        # 量化延迟到 forward() 首次调用时执行（惰性量化），
        # 统一入口确保 DTS 场景下单卡多卡行为一致。
        self.weight = weight

    def _quantize(self) -> None:
        weight_value = self.weight.value.detach()
        axes = self.axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [x + weight_value.ndim if x < 0 else x for x in axes]

        minmax_config = MinMaxBlockObserverConfig(axes=self.axes)
        minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

        weight_value, axes_, orig_shape, padded_shape = reshape_to_blocks(weight_value, axes, self.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.block_size > 0 else axes_

        minmax_block_observer.update(weight_value, sync=False, shared_exp_axes=shared_exp_axes)

        min_val, max_val = minmax_block_observer.get_min_max()

        self._axes = axes
        self._orig_shape = orig_shape
        self._padded_shape = padded_shape
        self.w_q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
            axes=self.config.ext.get('axes'),
        )
        self.w_q_param.ext['axes'] = self.axes
        self.w_q_storage = quantize(QStorage(QDType.FLOAT, weight_value), self.w_q_param)
        # keep w_q_storage in block shape for forward(); store original shape separately
        w_q_storage_orig_value = undo_reshape_to_blocks(self.w_q_storage.value.clone(), padded_shape, orig_shape)
        self.w_q_storage_orig = QStorage(self.w_q_storage.dtype, w_q_storage_orig_value)
        del self.weight
        self.weight = None

    def get_q_storage(self) -> QStorage:
        if self.w_q_storage_orig is None:
            _ = self.forward(None)
        return self.w_q_storage_orig

    def get_q_param(self) -> QParam:
        if self.w_q_param is None:
            _ = self.forward(None)
        return self.w_q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.mxfp8_per_block_sym, "minmax"),
        (qir.mxfp4_per_block_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer,
)
@logger_setter()
class MXActPerBlockMinmax(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        self.axes = config.ext.get('axes', -1)
        if not isinstance(self.axes, (int, list)):
            raise SchemaValidateError(f"Invalid value for 'axes': {self.axes}. Expected int or list[int].")
        self.q_param = QParam(
            scheme=QScheme(
                dtype=config.dtype,
                scope=config.scope,
                symmetric=config.symmetric,
            ),
            ext={'axes': self.axes},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            return QParam(scheme=self.config.to_scheme())
        return self.q_param

    def is_data_free(self) -> bool:
        """
        mxfp8、mxfp4的per_block量化为data free场景
        """
        return True


class Hif4ExtConfig(BaseModel):
    """Structured config for hif4 quantization method, parsed from QConfig.ext."""

    model_config = {"extra": "forbid"}

    axes: Union[int, List[int]] = -1
    scale_ratio: Annotated[float, Field(ge=0.5, le=1.5)] = 1
    clip2: Annotated[float, Field(ge=3.5, le=4.5)] = 4
    clip3: Annotated[float, Field(ge=1.75, le=2.25)] = 2


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.hif4_per_block_sym, "minmax"),
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter()
class HiFxWeightPerBlockMinmax(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()

        self.config = config
        self.hif4_cfg = Hif4ExtConfig(**config.ext)
        if not isinstance(self.hif4_cfg.axes, (int, list)):
            raise SchemaValidateError(f"Invalid value for 'axes': {self.hif4_cfg.axes}. Expected int or list[int].")
        self.block_size = config.dtype.hifx_finfo.block_size
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None
        self.orig_shape = None
        self.orig_dtype = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.w_q_storage is None:
            raise SpecError("No weight was set", action="please call init_weight first")
        weight_q, axes, padc, dim0 = pad_and_reshape(
            self.w_q_storage.value, self.orig_shape, self.hif4_cfg.axes, self.config.dtype.hifx_finfo
        )
        weight_q_dq = dequantize(QStorage(QDType.FLOAT, weight_q), self.w_q_param)
        kwargs = {"padc": padc, "padded_shape": weight_q_dq.value.shape, "orig_shape": self.orig_shape, "dim0": dim0}
        weight_q_dq.value = undo_pad_and_reshape(weight_q_dq.value.to(self.orig_dtype), axes, **kwargs)
        return weight_q_dq.value

    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        axes = self.hif4_cfg.axes
        minmax_config = MinMaxBlockObserverConfig(axes=axes)
        minmax_block_observer = MsMinMaxBlockObserver(minmax_config)
        weight_value = weight.value.detach()
        self.orig_shape, self.orig_dtype = weight_value.shape, weight_value.dtype

        weight_value = weight_value.to(torch.float32)
        weight_value, axes, padc, dim0 = pad_and_reshape(
            weight_value, self.orig_shape, axes, self.config.dtype.hifx_finfo
        )
        weight_level = torch.abs(weight_value)
        # Calculate and save the values of three-level quantization.
        max_levels = {}
        quant_level = 3  # The hifx uses three-level quantization, and three levels of scales are required.
        for ax in range(axes, axes - 3, -1):
            minmax_block_observer.update(weight_level, sync=self.sync, shared_exp_axes=[ax])
            _, max_lv = minmax_block_observer.get_min_max()
            # The key value indicates the level of scale. The ax parameter is not used to avoid confusion.
            max_levels[quant_level] = max_lv
            quant_level -= 1
            weight_level = max_lv
        self.w_q_param = calculate_qparam(
            min_val=None,
            max_val=None,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
            max_levels=max_levels,
            axes=axes,
            scale_ratio=self.hif4_cfg.scale_ratio,
            clip2=self.hif4_cfg.clip2,
            clip3=self.hif4_cfg.clip3,
        )
        self.w_q_storage = quantize(QStorage(QDType.FLOAT, weight_value), self.w_q_param)
        kwargs = {"padc": padc, "padded_shape": weight_value.shape, "orig_shape": self.orig_shape, "dim0": dim0}
        self.w_q_storage.value = undo_pad_and_reshape(self.w_q_storage.value.to(self.orig_dtype), axes, **kwargs)

    def get_q_storage(self) -> QStorage:
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        return self.w_q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.hif4_per_block_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer,
)
@logger_setter()
class HiFxActPerBlockMinmax(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        self.hif4_cfg = Hif4ExtConfig(**config.ext)
        if not isinstance(self.hif4_cfg.axes, (int, list)):
            raise SchemaValidateError(f"Invalid value for 'axes': {self.hif4_cfg.axes}. Expected int or list[int].")
        self.q_param = QParam(
            scheme=QScheme(
                dtype=config.dtype,
                scope=config.scope,
                symmetric=config.symmetric,
            ),
            ext={
                'axes': self.hif4_cfg.axes,
                'scale_ratio': self.hif4_cfg.scale_ratio,
                'clip2': self.hif4_cfg.clip2,
                'clip3': self.hif4_cfg.clip3,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            return QParam(scheme=self.config.to_scheme())
        return self.q_param

    def is_data_free(self) -> bool:
        """
        hif4的per_block量化为data free场景
        """
        return True
