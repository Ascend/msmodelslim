#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

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

msmodelslim.infra.evaluation.precheck.base 模块的单元测试
"""

from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.utils.exception import SchemaValidateError

from msmodelslim.core.tune_strategy import EvaluateAccuracy
from msmodelslim.infra.evaluation.precheck.base import (
    BasePrecheckConfig,
    BasePrecheckRule,
    validate_base_precheck_config_list,
)
from msmodelslim.infra.evaluation.precheck.expected_answer_rule import ExpectedAnswerPrecheckConfig


class _ConcretePrecheckRule(BasePrecheckRule):
    """单测用具体规则：仅实现抽象方法，便于验证基类公共逻辑"""

    def check(
        self, host: str, port: int, served_model_name: str, datasets: List[str]
    ) -> Optional[List[EvaluateAccuracy]]:
        return None


def _base_config(**overrides) -> ExpectedAnswerPrecheckConfig:
    """通过具体子类实例化，验证 BasePrecheckConfig 的继承字段行为。"""
    config_kwargs = {"test_cases": [{"What is 2+2?": "4"}]}
    config_kwargs.update(overrides)
    return ExpectedAnswerPrecheckConfig(**config_kwargs)


class TestBasePrecheckConfig:
    """测试 BasePrecheckConfig 的公开字段默认值与校验"""

    def test_default_fields(self):
        config = _base_config()
        assert config.type == "expected_answer"
        assert config.max_tokens == 512
        assert config.timeout == 60.0

    @pytest.mark.parametrize("field,bad_value", [("max_tokens", 0), ("timeout", 0.0)])
    def test_non_positive_field_rejected(self, field, bad_value):
        with pytest.raises(SchemaValidateError):
            _base_config(**{field: bad_value})


class TestValidateBasePrecheckConfigList:
    """测试 validate_base_precheck_config_list 的输入归一化与校验"""

    def test_none_returns_empty_list(self):
        assert validate_base_precheck_config_list(None) == []

    def test_single_dict_with_type_wrapped_into_list(self):
        result = validate_base_precheck_config_list({"type": "expected_answer", "max_tokens": 16})
        assert len(result) == 1
        assert isinstance(result[0], BasePrecheckConfig)
        assert result[0].type == "expected_answer"
        assert result[0].max_tokens == 16

    def test_existing_config_instances_preserved(self):
        config = _base_config()
        result = validate_base_precheck_config_list([config])
        assert result == [config]

    def test_dict_item_validated(self):
        result = validate_base_precheck_config_list([{"type": "garbled_text", "timeout": 5.0}])
        assert len(result) == 1
        assert result[0].type == "garbled_text"
        assert result[0].timeout == 5.0

    def test_non_list_rejected(self):
        with pytest.raises(ValueError):
            validate_base_precheck_config_list("expected_answer")

    def test_invalid_item_type_rejected(self):
        with pytest.raises(ValueError):
            validate_base_precheck_config_list([123])


class TestBasePrecheckRuleTestChatViaApi:
    """测试 BasePrecheckRule.test_chat_via_api 的请求构造与响应解析"""

    @patch("msmodelslim.infra.evaluation.precheck.base.safe_post")
    @patch("msmodelslim.infra.evaluation.precheck.base.build_safe_url")
    def test_returns_message_content_and_builds_request(self, mock_build_url, mock_post):
        mock_build_url.return_value = "http://localhost:8000/v1/chat/completions"
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
        mock_post.return_value = response

        rule = _ConcretePrecheckRule(_base_config(max_tokens=64, timeout=12.0))
        content = rule.test_chat_via_api("localhost", 8000, "test_model", "hi there")

        assert content == "hello"
        mock_build_url.assert_called_once_with(
            host="localhost", port=8000, endpoint="/v1/chat/completions", scheme="http"
        )
        response.raise_for_status.assert_called_once()
        assert mock_post.call_args.kwargs["url"] == "http://localhost:8000/v1/chat/completions"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "test_model"
        assert payload["messages"] == [{"role": "user", "content": "hi there"}]
        assert payload["max_tokens"] == 64
        assert payload["temperature"] == 0.1
        assert mock_post.call_args.kwargs["timeout"] == 12.0

    @patch("msmodelslim.infra.evaluation.precheck.base.safe_post")
    @patch("msmodelslim.infra.evaluation.precheck.base.build_safe_url")
    def test_max_tokens_argument_overrides_config(self, mock_build_url, mock_post):
        mock_build_url.return_value = "http://localhost:8000/v1/chat/completions"
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_post.return_value = response

        rule = _ConcretePrecheckRule(_base_config(max_tokens=64))
        content = rule.test_chat_via_api("localhost", 8000, "test_model", "hi", max_tokens=8)

        assert content == "ok"
        assert mock_post.call_args.kwargs["json"]["max_tokens"] == 8

    @patch("msmodelslim.infra.evaluation.precheck.base.safe_post")
    @patch("msmodelslim.infra.evaluation.precheck.base.build_safe_url")
    def test_invalid_response_format_raises_value_error(self, mock_build_url, mock_post):
        mock_build_url.return_value = "http://localhost:8000/v1/chat/completions"
        response = MagicMock()
        response.json.return_value = {"choices": []}
        mock_post.return_value = response

        rule = _ConcretePrecheckRule(_base_config())
        with pytest.raises(ValueError, match="Invalid response format"):
            rule.test_chat_via_api("localhost", 8000, "test_model", "hi")
