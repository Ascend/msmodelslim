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
from decimal import Decimal
from typing import Generator, List, Optional, Annotated

from pydantic import BaseModel, Field, AfterValidator

from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.plugin import TypedConfig
import msmodelslim.utils.validation.pydantic as pydtc

TUNING_STRATEGY_PLUGIN_PATH = "msmodelslim.tuning_strategy.plugins"


class EvaluateAccuracy(BaseModel):
    dataset: str
    accuracy: Decimal


class AccuracyExpectation(BaseModel):
    """精度期望：要求模型在指定数据集上达到的目标精度（含容差）。"""

    dataset: Annotated[str, AfterValidator(pydtc.validate_str_length())] = Field(description="数据集名称")
    target: Annotated[Decimal, AfterValidator(pydtc.greater_than_zero)] = Field(description="目标精度，必须 > 0")
    tolerance: Annotated[Decimal, AfterValidator(pydtc.in_range(min_val=0))] = Field(
        description="相对目标精度可容忍的偏差，必须 >= 0"
    )


class EvaluateResult(BaseModel):
    accuracies: List[EvaluateAccuracy] = Field(default_factory=list)
    expectations: List[AccuracyExpectation] = Field(default_factory=list)
    is_satisfied: bool


@TypedConfig.plugin_entry(entry_point_group=TUNING_STRATEGY_PLUGIN_PATH)
class StrategyConfig(TypedConfig):
    """调优策略配置基类：按 `type` 字段分派到具体策略（如 standing_high）。

    具体策略配置继承本类并固定各自的 `type` 取值；框架按 `type` 加载对应插件。
    """

    type: TypedConfig.TypeField


class ITuningStrategy(ABC):
    @abstractmethod
    def generate_practice(
        self,
        model: IModel,
        device: DeviceType = DeviceType.NPU,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]: ...


class ITuningStrategyFactory(ABC):
    @abstractmethod
    def create_strategy(self, strategy_config: StrategyConfig) -> ITuningStrategy: ...
