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

msmodelslim.infra.evaluation.precheck.garbled_text_rule 模块的单元测试
"""

# msmodelslim.infra.evaluation.precheck.garbled_text_rule 模块的单元测试
from decimal import Decimal
from unittest.mock import patch

from msmodelslim.infra.evaluation.precheck.garbled_text_rule import (
    GarbledTextRule,
    GarbledTextPrecheckConfig,
    EmptyTextCheckItem,
    RepeatedCharCheckItem,
    NormalCharRatioCheckItem,
    ControlCharCheckItem,
    RepeatedPatternCheckItem,
    get_all_check_item_names,
)


class TestGarbledTextRule:
    """测试GarbledTextRule类"""

    @patch('msmodelslim.infra.evaluation.precheck.garbled_text_rule.BasePrecheckRule.test_chat_via_api')
    def test_check_return_none_when_all_passed(self, mock_test_chat):
        """测试所有测试用例通过时返回None"""
        mock_test_chat.return_value = "This is a normal response."

        config = GarbledTextPrecheckConfig(test_cases=[{"message": "How are you?", "items": ["empty_text"]}])
        rule = GarbledTextRule(config)

        result = rule.check(host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"])
        assert result is None  # 校验返回None

    @patch('msmodelslim.infra.evaluation.precheck.garbled_text_rule.BasePrecheckRule.test_chat_via_api')
    def test_check_return_failed_results_when_garbled_detected(self, mock_test_chat):
        """测试检测到乱码时返回失败结果"""
        mock_test_chat.return_value = ""

        config = GarbledTextPrecheckConfig(test_cases=[{"message": "How are you?", "items": ["empty_text"]}])
        rule = GarbledTextRule(config)

        result = rule.check(host="localhost", port=8000, served_model_name="test_model", datasets=["test_dataset"])

        assert isinstance(result, list)  # 校验返回失败结果列表
        assert len(result) == 1  # 校验结果数量为1
        assert result[0].accuracy == Decimal('0')  # 校验accuracy为0.0


class TestGarbledTextCheckItems:
    """测试乱码检测检查项"""

    def test_empty_text_check_item_return_true_when_empty_or_whitespace(self):
        """测试检查空文本"""
        item = EmptyTextCheckItem()
        assert item.name == "empty_text"  # 校验name为empty_text
        assert item.check("") is True  # 校验空字符串返回True
        assert item.check("   ") is True  # 校验空白字符串返回True
        assert item.check("valid text") is False  # 校验有效文本返回False

    def test_repeated_char_check_item_return_true_when_repeated_chars_exceed_threshold(self):
        """测试检查重复字符"""
        item = RepeatedCharCheckItem(threshold=0.3)
        assert item.name == "repeated_char"  # 校验name为repeated_char
        assert item.check("a" * 100) is True  # 校验大量重复字符返回True
        assert item.check("This is a normal text.") is False  # 校验正常文本返回False

    def test_normal_char_ratio_check_item_return_true_when_ratio_below_min(self):
        """测试检查正常字符比例"""
        item = NormalCharRatioCheckItem(min_ratio=0.5)
        assert item.name == "normal_char_ratio"  # 校验name为normal_char_ratio
        assert item.check("\x00\x01\x02" * 50) is True  # 校验大量非正常字符返回True
        assert item.check("This is a normal text with Chinese 中文.") is False  # 校验正常文本返回False

    def test_control_char_check_item_return_true_when_ratio_exceed_max(self):
        """测试检查控制字符"""
        item = ControlCharCheckItem(max_ratio=0.1)
        assert item.name == "control_char"  # 校验name为control_char
        assert item.check("\x00\x01\x02" * 50) is True  # 校验大量控制字符返回True
        assert item.check("Normal text\nwith newline\tand tab.") is False  # 校验正常文本返回False

    def test_repeated_pattern_check_item_return_true_when_pattern_repeated(self):
        """测试检查重复模式"""
        item = RepeatedPatternCheckItem(min_pattern_count=3, min_pattern_ratio=0.5)
        assert item.name == "repeated_pattern"  # 校验name为repeated_pattern
        assert item.check("abc" * 20) is True  # 校验重复模式返回True
        assert item.check("This is a normal text without patterns.") is False  # 校验正常文本返回False
        assert item.check("ab") is False  # 校验太短文本返回False

    def test_get_all_check_item_names_return_list_with_all_names(self):
        """测试获取所有检查项名称"""
        names = get_all_check_item_names()
        assert isinstance(names, list)  # 校验返回列表类型
        assert "empty_text" in names  # 校验包含empty_text
        assert "repeated_char" in names  # 校验包含repeated_char
        assert "normal_char_ratio" in names  # 校验包含normal_char_ratio
        assert "control_char" in names  # 校验包含control_char
        assert "repeated_pattern" in names  # 校验包含repeated_pattern
