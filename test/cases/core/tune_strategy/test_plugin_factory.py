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

from unittest.mock import MagicMock

from msmodelslim.core.tune_strategy.plugin_factory import PluginTuningStrategyFactory


class TestPluginTuningStrategyFactory:
    """Tests for PluginTuningStrategyFactory."""

    def test_create_strategy_delegates_to_typed_factory_when_called(self):
        """场景：create_strategy。预期：内部 TypedFactory.create 被调用并传入 dataset_loader。"""
        dataset_loader = MagicMock()
        factory = PluginTuningStrategyFactory(dataset_loader=dataset_loader)
        expected_strategy = MagicMock()
        factory._factory = MagicMock()
        factory._factory.create.return_value = expected_strategy
        strategy_config = MagicMock()

        result = factory.create_strategy(strategy_config)

        factory._factory.create.assert_called_once_with(strategy_config, dataset_loader=dataset_loader)
        assert result is expected_strategy
