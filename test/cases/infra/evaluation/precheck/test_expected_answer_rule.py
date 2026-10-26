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

msmodelslim.infra.evaluation.precheck.expected_answer_rule 模块的单元测试
"""

# msmodelslim.infra.evaluation.precheck.expected_answer_rule 模块的单元测试
from decimal import Decimal
from unittest.mock import patch

import pytest

from msmodelslim.infra.evaluation.precheck.expected_answer_rule import (
    ExpectedAnswerRule,
    ExpectedAnswerPrecheckConfig,
    parse_test_case,
)
from msmodelslim.infra.evaluation.precheck.expected_answer_rule import TestCase as ExpectedAnswerTestCase


class TestExpectedAnswerRule:
    """测试ExpectedAnswerRule类"""

    @pytest.mark.parametrize(
        "text,expected_answer,should_match",
        [
            ("This is the answer", "answer", True),
            ("This is wrong", "answer", False),
            ("This is answer1", ["answer1", "answer2"], True),
            ("This is answer2", ["answer1", "answer2"], True),
            ("This is wrong", ["answer1", "answer2"], False),
            ("This is the answer", "ANSWER", True),
            ("This is the ANSWER", "answer", True),
            ("any response", None, True),
        ],
    )
    def test_contains_expected_answer_return_match_when_string_list_case_insensitive_none(
        self, text, expected_answer, should_match
    ):
        """测试字符串、列表、大小写不敏感、None值匹配期望答案"""
        rule = ExpectedAnswerRule(
            ExpectedAnswerPrecheckConfig(test_cases=[{"message": "test", "expected_answer": None}])
        )
        assert rule.contains_expected_answer(text, expected_answer) is should_match  # 校验匹配结果正确

    @patch('msmodelslim.infra.evaluation.precheck.expected_answer_rule.BasePrecheckRule.test_chat_via_api')
    def test_check_return_none_when_all_passed(self, mock_test_chat):
        """测试所有测试用例通过时返回None"""
        mock_test_chat.side_effect = ["The answer is 4", "The answer is 6"]

        config = ExpectedAnswerPrecheckConfig(
            test_cases=[
                {"What is 2+2?": "4"},
                {"What is 3+3?": "6"},
            ]
        )
        rule = ExpectedAnswerRule(config)

        result = rule.check(host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"])
        assert result is None  # 校验返回None
        # 校验调用次数正确
        assert mock_test_chat.call_count == 2

    @patch('msmodelslim.infra.evaluation.precheck.expected_answer_rule.BasePrecheckRule.test_chat_via_api')
    def test_check_return_failed_results_when_one_failed(self, mock_test_chat):
        """测试一个测试用例失败时返回失败结果"""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "The answer is 4"
            else:
                return "I don't know"

        mock_test_chat.side_effect = side_effect

        config = ExpectedAnswerPrecheckConfig(
            test_cases=[
                {"What is 2+2?": "4"},
                {"What is 3+3?": "6"},
            ]
        )
        rule = ExpectedAnswerRule(config)

        result = rule.check(host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"])

        assert isinstance(result, list)  # 校验返回失败结果列表
        assert len(result) == 1  # 校验结果数量为1
        assert result[0].accuracy == Decimal('0')  # 校验accuracy为0.0

    @patch('msmodelslim.infra.evaluation.precheck.expected_answer_rule.BasePrecheckRule.test_chat_via_api')
    def test_check_return_none_when_no_expected_answer(self, mock_test_chat):
        """测试没有期望答案的测试用例时返回None"""
        mock_test_chat.return_value = "Any response"

        config = ExpectedAnswerPrecheckConfig(test_cases=[{"What is 2+2?": None}])
        rule = ExpectedAnswerRule(config)

        result = rule.check(host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"])
        assert result is None  # 校验返回None


class TestParseTestCase:
    """测试解析测试用例函数"""

    @pytest.mark.parametrize(
        "input_data,expected_message,expected_answer",
        [
            ({"What is 2+2?": "4"}, "What is 2+2?", "4"),
            ({"test": ""}, "test", None),
            ({"test": []}, "test", None),
            ({"message": "What is 2+2?", "expected_answer": ["4", "four"]}, "What is 2+2?", ["4", "four"]),
        ],
    )
    def test_parse_test_case_return_test_case_when_valid_format(self, input_data, expected_message, expected_answer):
        """测试解析有效格式的测试用例"""
        test_case = parse_test_case(input_data)
        assert isinstance(test_case, ExpectedAnswerTestCase)  # 校验返回TestCase实例
        assert test_case.message == expected_message  # 校验message正确
        assert test_case.expected_answer == expected_answer  # 校验expected_answer正确

    @pytest.mark.parametrize(
        "invalid_input",
        [
            123,
            {"key1": "val1", "key2": "val2"},  # 多键值对且无 message 键
        ],
    )
    def test_parse_test_case_raise_value_error_when_invalid_format(self, invalid_input):
        """测试解析无效格式的测试用例时抛出ValueError异常"""
        with pytest.raises(ValueError):
            parse_test_case(invalid_input)
