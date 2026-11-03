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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional

from msmodelslim.app.train.qat.qat_linear_algorithm import QATLinearAlgorithm
from msmodelslim.core.train.protocol import ITrainHook, TrainHookContext
from msmodelslim.core.train.task import TrainAlgorithmConfig
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.plugin import TypedFactory

__all__ = [
    "AlgorithmTrainHook",
]


class AlgorithmTrainHook(ITrainHook):
    """
    显式实现 ``ITrainHook``：编排 ``QATLinearAlgorithm`` 能力。

    ``pre_train`` 顺序调用算法的 ``replace_linears`` → ``init_weight_clip_val``，
    并回写 ``qparam_names``；算法类本身不暴露 ``pre_train``。
    """

    def __init__(self, config: TrainAlgorithmConfig, factory: TypedFactory):
        self.config = config
        self._factory = factory
        self._impl: Optional[QATLinearAlgorithm] = None

    def _ensure(self, model=None) -> QATLinearAlgorithm:
        if self._impl is None:
            impl = self._factory.create(self.config, model=model)
            if not isinstance(impl, QATLinearAlgorithm):
                raise SchemaValidateError(
                    f"train algorithm plugin must implement QATLinearAlgorithm, got {type(impl).__name__}",
                    action="Please register a QATLinearAlgorithm subclass under Plugin:train_algorithm.",
                )
            self._impl = impl
        elif model is not None and getattr(self._impl, "model", None) is None:
            setattr(self._impl, "model", model)
        return self._impl

    def pre_train(self, ctx: TrainHookContext) -> None:
        model = ctx.model
        if model is None:
            raise SchemaValidateError(
                "AlgorithmTrainHook.pre_train requires ctx.model",
                action="Please load the model before invoking pre_train hooks.",
            )
        impl = self._ensure(model)
        # 1. 替换匹配的 nn.Linear
        shells = impl.replace_linears(model)
        # 2. 初始化 weight_clip_val
        impl.init_weight_clip_val(shells)

        names = list(impl.qparam_names())
        ctx.extra["qparam_names"] = names
        if ctx.core_task is not None:
            ctx.core_task.qparam_names = names
