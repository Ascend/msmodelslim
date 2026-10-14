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

from msmodelslim.core.tune_strategy.interface import ITuningStrategyFactory, ITuningStrategy, StrategyConfig
from msmodelslim.utils.plugin.typed_factory import TypedFactory
from .dataset_loader_infra import DatasetLoaderInfra


class PluginTuningStrategyFactory(ITuningStrategyFactory):
    def __init__(self, dataset_loader: DatasetLoaderInfra):
        """
        初始化调优策略工厂

        使用 TypedFactory 来管理策略类的动态加载和实例化；entry_point_group 从 StrategyConfig 的 plugin_entry 读取。
        """
        self._factory = TypedFactory[ITuningStrategy](config_base_class=StrategyConfig)
        self.dataset_loader = dataset_loader

    def create_strategy(self, strategy_config: StrategyConfig) -> ITuningStrategy:
        return self._factory.create(strategy_config, dataset_loader=self.dataset_loader)
