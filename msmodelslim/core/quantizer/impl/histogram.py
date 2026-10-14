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

import msmodelslim.ir as qir
from msmodelslim.ir.api import fake_quantize, calculate_qparam
from msmodelslim.ir.qal import QABCRegistry, QDType, QStorage, QParam, QScope
from msmodelslim.core.observer.histogram import HistogramObserver, HistogramObserverConfig
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.exception import SpecError
from ..base import AutoActQuantizer, QConfig


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_tensor_sym, "histogram"),  # 注册为int8对称per-tensor量化器
        (qir.int8_per_tensor_asym, "histogram"),  # 注册为int8非对称per-tensor量化器
    ],
    abc_type=AutoActQuantizer,
)
@logger_setter()
class ActPerTensorHistogram(AutoActQuantizer):
    """
    Per-Tensor 直方图激活值量化器

    该类实现了基于直方图统计的激活值量化器，继承自AutoActQuantizer。
    通过分析输入张量的分布直方图，自动搜索最佳的截断值，作为min/max，计算相应的量化参数。

    主要成员变量：
    - config: 量化配置对象，包含dtype、scope、method、symmetric等配置信息
    - histogram_observer: 直方图观察器，用于统计输入分布并搜索最优的截断值
    - q_param: 量化参数（包含scale、offset等），在forward调用后自动设置

    使用方式：
    1. 创建量化器实例，传入量化配置
    2. 调用forward方法进行前向推理，自动更新量化参数
    3. 通过get_q_param获取计算得到的量化参数
    """

    def __init__(self, config: QConfig):
        """
        初始化直方图量化器

        Args:
            config (QConfig): 量化配置对象，包含量化类型、范围、对称性等参数

        Raises:
            SchemaValidateError: 当配置验证失败时抛出
        """
        super().__init__()
        self.config = config
        histogram_config = HistogramObserverConfig(symmetric=config.symmetric)
        # 初始化直方图观察器，用于统计输入张量的分布
        self.histogram_observer = HistogramObserver(histogram_config)
        # 量化参数，初始为None，在forward调用后会被设置
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，执行量化计算

        该方法会：
        1. 更新直方图观察器，统计输入张量的分布
        2. 获取最佳的截断值
        3. 计算量化参数
        4. 执行伪量化操作并返回结果

        Args:
            x (torch.Tensor): 输入的浮点张量，需要进行量化的激活值

        Returns:
            torch.Tensor: 量化后的张量，保持与输入相同的形状和数据类型
        """

        # 更新直方图观察器，统计输入张量的分布信息
        self.histogram_observer.update(x)

        # 获取基于直方图统计的最佳截断值
        # clip_min: 最小截断值，clip_max: 最大截断值
        clip_min, clip_max = self.histogram_observer.get_clip_bounds()

        # 根据截断值计算量化参数
        self.q_param = calculate_qparam(
            min_val=clip_min,  # 最小截断值
            max_val=clip_max,  # 最大截断值
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )

        # 执行伪量化操作
        # 将浮点输入包装为QStorage，然后使用计算得到的量化参数进行伪量化
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.clamp(
            clip_min, clip_max
        )  # 防止溢出

    def get_q_param(self) -> QParam:
        """
        获取计算得到的量化参数

        该方法返回在forward调用中计算得到的量化参数。
        如果还没有调用forward方法，则会抛出运行时错误。

        Returns:
            QParam: 量化参数对象，包含scale、offset等信息

        Raises:
            SpecError: 如果还没有调用forward方法，q_param为None时抛出
        """
        if self.q_param is None:
            raise SpecError("No q_param was set", action="Please call forward first")
        return self.q_param
