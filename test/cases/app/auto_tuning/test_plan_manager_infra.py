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

import pytest

from msmodelslim.app.auto_tuning.plan_manager_infra import TuningPlanManagerInfra


class _ConcretePlanManager(TuningPlanManagerInfra):
    def __init__(self, plan):
        self._plan = plan
        self.queried = None

    def get_plan_by_id(self, plan_id: str):
        self.queried = plan_id
        return self._plan


def _fake_plan():
    """构造一个 Mock 模拟 TuningPlanConfig 实例（不实例化真实 Pydantic 避免插件依赖）。"""
    return MagicMock(spec=["strategy", "evaluation"])


class TestTuningPlanManagerInfra:
    """Test suite for TuningPlanManagerInfra — ABC 接口契约。"""

    # ---------- 正常情形 ----------

    def test_get_plan_by_id_returns_plan_when_subclass_implements(self):
        """主路径：get_plan_by_id 应返回子类提供的 plan。"""
        plan = _fake_plan()
        manager = _ConcretePlanManager(plan)

        result = manager.get_plan_by_id("plan-001")

        assert result is plan
        assert manager.queried == "plan-001"

    def test_get_plan_by_id_returns_plan_object_with_strategy_and_eval_attrs(self):
        """主路径：返回的 plan 应暴露 strategy/evaluation 属性。"""
        plan = _fake_plan()
        plan.strategy = "s1"
        plan.evaluation = "e1"
        manager = _ConcretePlanManager(plan)

        result = manager.get_plan_by_id("plan-002")

        assert result.strategy == "s1"
        assert result.evaluation == "e1"

    # ---------- 边界情形 ----------

    def test_get_plan_by_id_queries_with_empty_string_when_caller_passes_empty_id(self):
        """边界：空字符串 plan_id 应原样透传给子类。"""
        manager = _ConcretePlanManager(_fake_plan())
        manager.get_plan_by_id("")

        assert manager.queried == ""

    def test_get_plan_by_id_queries_with_unicode_id_when_caller_passes_unicode(self):
        """边界：unicode 字符串 plan_id 应原样透传。"""
        manager = _ConcretePlanManager(_fake_plan())
        manager.get_plan_by_id("计划-001")

        assert manager.queried == "计划-001"

    # ---------- 异常情形 ----------

    def test_tuning_plan_manager_infra_raises_type_error_when_instantiated_directly(self):
        """异常：直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            TuningPlanManagerInfra()  # pylint: disable=abstract-class-instantiated

    def test_subclass_raises_type_error_when_get_plan_by_id_not_implemented(self):
        """异常：子类未实现 get_plan_by_id 应抛 TypeError。"""

        class _Incomplete(TuningPlanManagerInfra):
            pass

        with pytest.raises(TypeError):
            _Incomplete()  # pylint: disable=abstract-class-instantiated
