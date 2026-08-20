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

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, SerializeAsAny

from msmodelslim.core.tune_strategy import StrategyConfig
from .evaluation_service_infra import EvaluateServiceConfig


class TuningPlanConfig(BaseModel):
    """自动调优计划配置：顶层只含 strategy（调优策略）与 evaluation（评估服务）两个字段。

    对应自动调优 YAML 的根节点结构，`strategy` 与 `evaluation` 均按各自的 `type`
    字段分派到具体的策略/评估服务配置。
    """

    strategy: SerializeAsAny[StrategyConfig] = Field(
        description="调优策略配置，按 `type` 字段分派，如 `standing_high`、`standing_high_with_experience`、`binary_fallback`"
    )
    evaluation: SerializeAsAny[EvaluateServiceConfig] = Field(
        description="评估服务配置，按 `type` 字段分派，如 `service_oriented`"
    )


class TuningPlanManagerInfra(ABC):
    @abstractmethod
    def get_plan_by_id(self, plan_id: str) -> TuningPlanConfig: ...
