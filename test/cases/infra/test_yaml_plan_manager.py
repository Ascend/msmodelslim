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

msmodelslim.infra.yaml_plan_manager 模块的单元测试
"""

from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import Field

from msmodelslim.app.auto_tuning import TuningPlanConfig
from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateServiceConfig
from msmodelslim.core.tune_strategy import StrategyConfig
from msmodelslim.infra.yaml_plan_manager import YamlTuningPlanManager
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.plugin.plugin_utils import register_plugin


class _PlanStrategyConfig(StrategyConfig):
    type: Literal["b1-plan-strategy"] = "b1-plan-strategy"
    rounds: int = Field(gt=0)


class _PlanEvaluationConfig(EvaluateServiceConfig):
    type: Literal["b1-plan-evaluation"] = "b1-plan-evaluation"
    endpoint: str


class _PlanComponent:
    pass


def _get_strategy_plugin():
    return _PlanStrategyConfig, _PlanComponent


def _get_evaluation_plugin():
    return _PlanEvaluationConfig, _PlanComponent


@pytest.fixture(autouse=True)
def _register_plan_plugins():
    register_plugin(_get_strategy_plugin)
    register_plugin(_get_evaluation_plugin)


class TestYamlTuningPlanManager:
    """测试 YAML 调优计划读取、权限检查与插件分派。"""

    @patch("msmodelslim.infra.yaml_plan_manager.yaml_safe_load")
    @patch("msmodelslim.infra.yaml_plan_manager.check_read_permission")
    def test_get_plan_by_id_loads_and_validates_plugin_configs(self, mock_permission, mock_yaml_load):
        """正常路径：读取 YAML 后按 type 分派为具体策略与评估配置。"""
        mock_yaml_load.return_value = {
            "strategy": {"type": "b1-plan-strategy", "rounds": 4},
            "evaluation": {"type": "b1-plan-evaluation", "endpoint": "http://localhost:8080"},
        }

        plan = YamlTuningPlanManager().get_plan_by_id("plan-a")

        assert isinstance(plan, TuningPlanConfig)
        assert isinstance(plan.strategy, _PlanStrategyConfig)
        assert plan.strategy.rounds == 4
        assert isinstance(plan.evaluation, _PlanEvaluationConfig)
        assert plan.evaluation.endpoint == "http://localhost:8080"
        mock_permission.assert_called_once_with("plan-a")
        mock_yaml_load.assert_called_once_with("plan-a")

    @patch("msmodelslim.infra.yaml_plan_manager.yaml_safe_load")
    @patch("msmodelslim.infra.yaml_plan_manager.check_read_permission")
    def test_get_plan_by_id_raises_when_strategy_config_invalid(self, mock_permission, mock_yaml_load):
        """异常：插件配置未通过字段约束时拒绝返回计划。"""
        mock_yaml_load.return_value = {
            "strategy": {"type": "b1-plan-strategy", "rounds": 0},
            "evaluation": {"type": "b1-plan-evaluation", "endpoint": "http://localhost:8080"},
        }

        with pytest.raises(SchemaValidateError, match="strategy.rounds"):
            YamlTuningPlanManager().get_plan_by_id("plan-a")
