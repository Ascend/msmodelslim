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

msmodelslim.infra.evaluation.precheck.prechecker 模块的单元测试
"""

# msmodelslim.infra.evaluation.precheck.prechecker 模块的单元测试
# 主要测试模型输出预检功能。单测不依赖插件加载，通过 mock 直接验证工厂与规则类业务逻辑。
from decimal import Decimal
from typing import Literal
from unittest.mock import patch, MagicMock

import pytest

from msmodelslim.core.tune_strategy import EvaluateAccuracy
from msmodelslim.infra.evaluation.precheck.base import BasePrecheckConfig
from msmodelslim.infra.evaluation.precheck.expected_answer_rule import (
    ExpectedAnswerRule,
    ExpectedAnswerPrecheckConfig,
)
from msmodelslim.infra.evaluation.precheck.garbled_text_rule import (
    GarbledTextRule,
    GarbledTextPrecheckConfig,
)
from msmodelslim.infra.evaluation.precheck.prechecker import (
    PrecheckerFactory,
    ModelOutputPrechecker,
    model_output_precheck,
)
from msmodelslim.infra.evaluation.precheck.prechecker import PRECHECK_PLUGIN_PATH


class _InvalidTypePrecheckConfig(BasePrecheckConfig):
    """单测用：不存在的插件类型，用于测试 create 失败时被跳过"""

    type: Literal["invalid_type"] = "invalid_type"


def _fake_load_plugin_function(entry_point_group: str, plugin_type: str):
    """不依赖 entry_points，直接返回预检查规则类（用于单测 mock）。"""
    if entry_point_group != PRECHECK_PLUGIN_PATH:
        from msmodelslim.utils.exception import UnsupportedError

        raise UnsupportedError(f"No plugin found for type {plugin_type!r} in group {entry_point_group!r}.")
    if plugin_type == "expected_answer":
        return (ExpectedAnswerPrecheckConfig, ExpectedAnswerRule)
    if plugin_type == "garbled_text":
        return (GarbledTextPrecheckConfig, GarbledTextRule)
    from msmodelslim.utils.exception import UnsupportedError

    raise UnsupportedError(f"No plugin found for type {plugin_type!r} in group {entry_point_group!r}.")


class TestPrecheckerFactory:
    """测试 PrecheckerFactory 类（mock 插件加载，只测工厂与规则业务逻辑）"""

    @patch('msmodelslim.infra.evaluation.precheck.prechecker.get_entry_points')
    def test_get_supported_types_return_list_when_call(self, mock_get_entry_points):
        """测试 get_supported_types 返回列表，is_supported_type 按类型判断正确（不依赖真实 entry_points）"""

        def _entry_point(plugin_name):
            ep = MagicMock()
            ep.name = plugin_name
            return ep

        mock_get_entry_points.return_value = [
            _entry_point("expected_answer"),
            _entry_point("garbled_text"),
        ]
        types = PrecheckerFactory.get_supported_types()
        assert isinstance(types, list)
        assert "expected_answer" in types
        assert "garbled_text" in types
        assert PrecheckerFactory.is_supported_type("expected_answer") is True
        assert PrecheckerFactory.is_supported_type("garbled_text") is True
        assert PrecheckerFactory.is_supported_type("unknown_type") is False

    @patch('msmodelslim.utils.plugin.plugin_utils.load_plugin_function', side_effect=_fake_load_plugin_function)
    def test_create_return_expected_answer_rule_when_type_expected_answer(self, mock_load_plugin):
        """测试 create 在 type=expected_answer 时返回 ExpectedAnswerRule 并校验业务字段"""
        factory = PrecheckerFactory()
        config = ExpectedAnswerPrecheckConfig(test_cases=[{"What is 2+2?": "4"}])
        rule = factory.create(config)
        assert isinstance(rule, ExpectedAnswerRule)
        assert len(rule.config.test_cases) == 1
        assert rule.config.test_cases[0].message == "What is 2+2?"
        assert rule.config.test_cases[0].expected_answer == "4"

    @patch('msmodelslim.utils.plugin.plugin_utils.load_plugin_function', side_effect=_fake_load_plugin_function)
    def test_create_return_garbled_text_rule_when_type_garbled_text(self, mock_load_plugin):
        """测试 create 在 type=garbled_text 时返回 GarbledTextRule 并校验业务字段"""
        factory = PrecheckerFactory()
        config = GarbledTextPrecheckConfig(test_cases=[{"message": "How are you?", "items": ["empty_text"]}])
        rule = factory.create(config)
        assert isinstance(rule, GarbledTextRule)
        assert len(rule.config.test_cases) == 1


class TestModelOutputPrechecker:
    """测试 ModelOutputPrechecker 类（mock 插件加载，只测管理器与规则业务逻辑）"""

    @pytest.mark.parametrize(
        "configs,expected_count",
        [
            (None, 0),
            ([ExpectedAnswerPrecheckConfig(test_cases=[{"test": "answer"}])], 1),
            ([_InvalidTypePrecheckConfig()], 0),
        ],
    )
    @patch('msmodelslim.utils.plugin.plugin_utils.load_plugin_function', side_effect=_fake_load_plugin_function)
    def test_init_set_precheckers_count_when_config_none_valid_invalid(self, mock_load_plugin, configs, expected_count):
        """测试空配置、有效配置、无效配置初始化时 precheckers 数量"""
        prechecker = ModelOutputPrechecker(configs=configs)
        assert len(prechecker.precheckers) == expected_count

    def test_check_return_none_when_no_precheckers(self):
        """测试没有预检查器时返回 None"""
        prechecker = ModelOutputPrechecker(configs=None)
        result = prechecker.check(
            host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"]
        )
        assert result is None

    @patch('msmodelslim.infra.evaluation.precheck.prechecker.BasePrecheckRule.test_chat_via_api')
    @patch('msmodelslim.utils.plugin.plugin_utils.load_plugin_function', side_effect=_fake_load_plugin_function)
    def test_check_return_none_when_all_passed(self, mock_load_plugin, mock_test_chat):
        """测试所有预检查通过时返回 None"""
        mock_test_chat.return_value = "The answer is 4"
        configs = [ExpectedAnswerPrecheckConfig(test_cases=[{"What is 2+2?": "4"}])]
        prechecker = ModelOutputPrechecker(configs=configs)
        result = prechecker.check(
            host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"]
        )
        assert result is None

    @patch('msmodelslim.infra.evaluation.precheck.prechecker.BasePrecheckRule.test_chat_via_api')
    @patch('msmodelslim.utils.plugin.plugin_utils.load_plugin_function', side_effect=_fake_load_plugin_function)
    def test_check_return_failed_results_when_one_failed(self, mock_load_plugin, mock_test_chat):
        """测试一个预检查失败时返回失败结果列表"""
        mock_test_chat.return_value = "I don't know"
        configs = [ExpectedAnswerPrecheckConfig(test_cases=[{"What is 2+2?": "4"}])]
        prechecker = ModelOutputPrechecker(configs=configs)
        result = prechecker.check(
            host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"]
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].dataset == "test_dataset"
        assert result[0].accuracy == Decimal('0')

    @patch('msmodelslim.infra.evaluation.precheck.prechecker.BasePrecheckRule.test_chat_via_api')
    @patch('msmodelslim.utils.plugin.plugin_utils.load_plugin_function', side_effect=_fake_load_plugin_function)
    def test_check_return_failed_results_when_first_fails(self, mock_load_plugin, mock_test_chat):
        """测试多个预检查器中第一个失败时不执行后续预检查器"""
        mock_test_chat.return_value = "I don't know"
        configs = [
            ExpectedAnswerPrecheckConfig(test_cases=[{"What is 2+2?": "4"}]),
            GarbledTextPrecheckConfig(test_cases=[{"message": "How are you?"}]),
        ]
        prechecker = ModelOutputPrechecker(configs=configs)
        result = prechecker.check(
            host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"]
        )
        assert isinstance(result, list)
        assert len(result) == 1


class TestModelOutputPrecheckDecorator:
    """测试model_output_precheck装饰器"""

    @pytest.mark.parametrize(
        "eval_config_attr,precheck_value,server_info",
        [
            (None, None, {}),
            ("eval_config", None, {}),
            ("eval_config", "not a list", {}),
            ("eval_config", [], {}),
            ("eval_config", [{"type": "expected_answer"}], {"host": None, "port": None, "served_model_name": None}),
        ],
    )
    def test_decorator_return_original_result_when_skip_precheck(self, eval_config_attr, precheck_value, server_info):
        """测试没有eval_config、没有precheck配置、precheck不是列表、precheck为空列表、缺少服务器信息时跳过预检查"""

        @model_output_precheck
        def test_func(self):
            return "original_result"

        class TestClass:
            def __init__(self):
                if eval_config_attr:
                    self.eval_config = MagicMock()
                    self.eval_config.precheck = precheck_value
                    for key, value in server_info.items():
                        setattr(self.eval_config, key, value)

        obj = TestClass()
        result = test_func(obj)
        assert result == "original_result"  # 校验返回原函数结果

    @patch('msmodelslim.infra.evaluation.precheck.prechecker.ModelOutputPrechecker')
    def test_decorator_return_original_result_when_precheck_passed(self, mock_prechecker_class):
        """测试预检查通过时返回原函数结果"""
        mock_prechecker = MagicMock()
        mock_prechecker.check.return_value = None
        mock_prechecker_class.return_value = mock_prechecker

        @model_output_precheck
        def test_func(self):
            return "original_result"

        class TestClass:
            def __init__(self):
                self.eval_config = MagicMock()
                self.eval_config.precheck = [{"type": "expected_answer"}]
                self.eval_config.host = "localhost"
                self.eval_config.port = 8000
                self.eval_config.served_model_name = "test_model"
                self.datasets = ["test_dataset"]

        obj = TestClass()
        result = test_func(obj)
        assert result == "original_result"  # 校验返回原函数结果
        mock_prechecker.check.assert_called_once()  # 校验调用check方法

    @patch('msmodelslim.infra.evaluation.precheck.prechecker.ModelOutputPrechecker')
    def test_decorator_return_failed_results_when_precheck_failed(self, mock_prechecker_class):
        """测试预检查失败时返回失败结果且不执行原函数"""
        mock_prechecker = MagicMock()
        mock_prechecker.check.return_value = [EvaluateAccuracy(dataset="test_dataset", accuracy=Decimal('0'))]
        mock_prechecker_class.return_value = mock_prechecker

        @model_output_precheck
        def test_func(self):
            return "original_result"

        class TestClass:
            def __init__(self):
                self.eval_config = MagicMock()
                self.eval_config.precheck = [{"type": "expected_answer"}]
                self.eval_config.host = "localhost"
                self.eval_config.port = 8000
                self.eval_config.served_model_name = "test_model"
                self.datasets = ["test_dataset"]

        obj = TestClass()
        result = test_func(obj)
        assert result is mock_prechecker.check.return_value  # 校验返回预检查结果列表
        assert isinstance(result, list)
        assert len(result) == 1  # 校验结果数量为1
        assert result[0].accuracy == Decimal('0')  # pylint: disable=no-member  # 校验accuracy为0.0
        mock_prechecker.check.assert_called_once()  # 校验调用check方法
