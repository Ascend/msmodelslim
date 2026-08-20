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

from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from pydantic import validate_call
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.ir.qal import QDType, QStorage
from msmodelslim.utils.logging import logger_setter
from .base import AutoActQuantizer, AutoWeightQuantizer, QConfig, QScope


class LinearQConfig(BaseModel):
    """线性层（Linear）的量化配置，含激活与权重两路量化。

    `LinearQConfig` 是 `linear_quant` 处理器 `qconfig` 字段的配置类型，
    分别描述激活（`act`）与权重（`weight`）的量化方式。
    """

    act: QConfig = Field(
        default=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        description="激活值的量化配置。默认 `float`（不量化激活），仅对权重做量化。",
    )
    weight: QConfig = Field(description="权重的量化配置，必选。")


@logger_setter()
class LinearQuantizer(nn.Module):
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, config: LinearQConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.sync = False  # 默认不启用同步操作
        self.input_quantizer = AutoActQuantizer.from_config(config.act)
        self.weight_quantizer = AutoWeightQuantizer.from_config(config.weight)
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[nn.Parameter] = None
        self.q_weight: Optional[QStorage] = None

    def enable_sync(self):
        """启用同步操作，并递归启用所有子量化器的同步"""
        self.sync = True
        if hasattr(self.input_quantizer, 'enable_sync'):
            self.input_quantizer.enable_sync()
        if hasattr(self.weight_quantizer, 'enable_sync'):
            self.weight_quantizer.enable_sync()

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def setup(self, linear: nn.Linear):
        self.bias = linear.bias
        self.weight_quantizer.init_weight(QStorage(QDType.FLOAT, value=linear.weight.detach()), self.bias)

        for hook_id, hook in linear._forward_pre_hooks.items():
            with_kwargs = hook_id in linear._forward_pre_hooks_with_kwargs
            self.register_forward_pre_hook(hook, with_kwargs=with_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with QStorage.set_value_float_type(x.dtype):
            x = self.input_quantizer(x)
            weight = self.weight_quantizer(x)
        return F.linear(x, weight, self.bias)

    def deploy(self):
        fake_quantizer = qir.AutoFakeQuantLinear.create(
            self.input_quantizer.get_q_param(),
            self.weight_quantizer.get_q_param(),
            self.weight_quantizer.get_q_storage(),
            self.bias,
        )

        for hook in self._forward_pre_hooks.values():
            if isinstance(hook, qir.HookIR):
                fake_quantizer = hook.wrapper_module(fake_quantizer)

        return fake_quantizer

    def support_distributed(self) -> bool:
        """
        判断是否支持分布式
        通过检查 input_quantizer 和 weight_quantizer 是否都支持分布式来判断

        Returns:
            bool: 是否支持分布式
        """
        input_support = self.input_quantizer.support_distributed()
        weight_support = self.weight_quantizer.support_distributed()
        return input_support and weight_support

    def is_data_free(self) -> bool:
        """
        判断是否是data free场景
        通过检查 input_quantizer 和 weight_quantizer 是否都支持data free来判断

        Returns:
            bool: 是否是data free场景
        """
        return self.input_quantizer.is_data_free() and self.weight_quantizer.is_data_free()

    def validate_config(self):
        self.input_quantizer.validate_ext_config()
        self.weight_quantizer.validate_ext_config()
