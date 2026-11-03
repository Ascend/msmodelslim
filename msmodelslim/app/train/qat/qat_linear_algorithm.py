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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

from torch import nn

from msmodelslim.core.convert.protocol import IIRTransformProcessor
from msmodelslim.utils.exception import UnsupportedError

__all__ = [
    "QATLinearAlgorithm",
]


class QATLinearAlgorithm(ABC):
    """
    QAT 线性层算法协议（processor 侧实现，如 ParetoQ）。

    职责边界：
    - 负责开训前将 ``nn.Linear`` 替换为伪量化壳，并初始化可学习量化参数；
    - 通过 ``get_convert_processors`` 声明 QAT→deploy 的 IR 边（由 Task 查询并注册）；
    - 不负责训练循环、checkpoint 选优或 convert 路由注册本身。

    约定：
    - 由 ``AlgorithmTrainHook.pre_train`` 按 ``replace_linears`` → ``init_weight_clip_val``
      顺序调用；``qparam_names`` 回写到训练任务供优化器分组使用；
    - ``QATTask.register_convert`` 调用 ``get_convert_processors`` 并挂到 ``IRRouter``。
    """

    @abstractmethod
    def replace_linears(self, model: nn.Module) -> List[nn.Module]:
        """
        将匹配的 ``nn.Linear`` 原地替换为算法伪量化壳。

        参数:
            model: 已加载的训练模型

        返回:
            本次替换得到的伪量化壳列表（供 ``init_weight_clip_val`` 使用）
        """
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support replace_linears",
            action="Please implement replace_linears on the QAT linear algorithm plugin.",
        )

    @abstractmethod
    def init_weight_clip_val(self, shells: Sequence[nn.Module]) -> None:
        """
        初始化伪量化壳上的可学习量化参数（如 weight_clip_val / α）。

        参数:
            shells: ``replace_linears`` 返回的壳模块列表
        """
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support init_weight_clip_val",
            action="Please implement init_weight_clip_val on the QAT linear algorithm plugin.",
        )

    @abstractmethod
    def qparam_names(self) -> Tuple[str, ...]:
        """
        返回训练期需单独分组的可学习量化参数名。

        返回:
            参数名元组（如 ``(\"weight_clip_val\",)``）
        """
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support qparam_names",
            action="Please implement qparam_names on the QAT linear algorithm plugin.",
        )

    @classmethod
    @abstractmethod
    def get_convert_processors(cls) -> List[IIRTransformProcessor]:
        """
        返回该算法提供的 convert ``IIRTransformProcessor`` 边。

        返回:
            可注册到 ``IRRouter`` 的 processor 列表（QAT FakeQuant IR → 部署 IR）
        """
        raise UnsupportedError(
            f"{cls.__name__} does not support get_convert_processors",
            action="Please implement get_convert_processors on the QAT linear algorithm plugin.",
        )
