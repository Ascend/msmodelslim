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

from abc import abstractmethod
from typing import Generator, Optional

from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import ITuningStrategy
from msmodelslim.core.tune_strategy.interface import StrategyConfig, EvaluateResult
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from .dataset_loader_infra import DatasetLoaderInfra


class BaseTuningStrategy(ITuningStrategy):
    def __init__(self, config: StrategyConfig, dataset_loader: DatasetLoaderInfra):
        self.config = config
        self.dataset_loader = dataset_loader

    @abstractmethod
    def generate_practice(
        self,
        model: IModel,
        device: DeviceType = DeviceType.NPU,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]: ...
