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
# pylint: disable=implicit-str-concat,logging-fstring-interpolation

import re
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import List, Optional, Any, Literal, Dict, Annotated

from pydantic import BaseModel, Field, field_validator, AfterValidator

from msmodelslim.core.tune_strategy import EvaluateAccuracy
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.validation.pydantic import validate_str_length, non_empty_string

from .base import BasePrecheckRule, BasePrecheckConfig


class GarbledTextCheckItem(ABC):
    """乱码检测检查项接口（责任链模式）"""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回检查项的名称"""
        pass

    @abstractmethod
    def check(self, text: str) -> bool:
        """检查文本是否为乱码，返回 True 如果是乱码"""
        pass


class EmptyTextCheckItem(GarbledTextCheckItem):
    """空文本检查项"""

    @property
    def name(self) -> str:
        return "empty_text"

    def check(self, text: str) -> bool:
        """检查文本是否为空"""
        text = text.strip()
        if not text:
            get_logger().debug(f"[{self.name}] Empty text detected")
            return True
        return False


class RepeatedCharCheckItem(GarbledTextCheckItem):
    """连续重复字符检查项"""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "repeated_char"

    def check(self, text: str) -> bool:
        """检查是否存在大量连续重复字符"""
        text_len = len(text)
        if text_len == 0:
            return False

        max_repeat = 0
        current_char = None
        current_count = 0
        for char in text:
            if char == current_char:
                current_count += 1
                max_repeat = max(max_repeat, current_count)
            else:
                current_char = char
                current_count = 1

        if max_repeat > text_len * self.threshold:
            get_logger().debug(f"[{self.name}] Excessive repeated characters detected: {max_repeat}/{text_len}")
            return True
        return False


class NormalCharRatioCheckItem(GarbledTextCheckItem):
    """正常字符比例检查项"""

    def __init__(self, min_ratio: float = 0.5):
        self.min_ratio = min_ratio

    @property
    def name(self) -> str:
        return "normal_char_ratio"

    def check(self, text: str) -> bool:
        """检查正常字符比例是否过低"""
        text_len = len(text)
        if text_len == 0:
            return False

        # 匹配中英文、数字、常见标点符号
        normal_chars = re.findall(r'[\u4e00-\u9fff\w\s，。！？、；：""' '（）【】《》.,!?;:()\\[\\]{}"\'-]', text)
        normal_ratio = len(normal_chars) / text_len

        if normal_ratio < self.min_ratio:
            get_logger().debug(f"[{self.name}] Low normal character ratio detected: {normal_ratio:.2f}")
            return True
        return False


class ControlCharCheckItem(GarbledTextCheckItem):
    """控制字符检查项"""

    def __init__(self, max_ratio: float = 0.1):
        self.max_ratio = max_ratio

    @property
    def name(self) -> str:
        return "control_char"

    def check(self, text: str) -> bool:
        """检查是否包含大量控制字符"""
        text_len = len(text)
        if text_len == 0:
            return False

        # 统计控制字符（排除换行、回车、制表符）
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        control_ratio = control_chars / text_len

        if control_ratio > self.max_ratio:
            get_logger().debug(f"[{self.name}] Excessive control characters detected: {control_ratio:.2f}")
            return True
        return False


class RepeatedPatternCheckItem(GarbledTextCheckItem):
    """重复模式检查项"""

    def __init__(self, min_pattern_count: int = 3, min_pattern_ratio: float = 0.5):
        self.min_pattern_count = min_pattern_count
        self.min_pattern_ratio = min_pattern_ratio

    @property
    def name(self) -> str:
        return "repeated_pattern"

    def check(self, text: str) -> bool:
        """检查是否存在明显的重复模式"""
        text_len = len(text)
        if text_len < 6:
            return False

        max_pattern_len = min(10, text_len // 3)
        for pattern_len in range(2, max_pattern_len):
            # 检查文本开头的模式是否在文本中重复出现
            pattern = text[:pattern_len]
            count = text.count(pattern)
            # 如果模式重复次数达到阈值，且重复部分占总长度的比例达到阈值，可能是乱码
            if count >= self.min_pattern_count and len(pattern) * count >= text_len * self.min_pattern_ratio:
                get_logger().debug(f"[{self.name}] Repeated pattern detected: pattern_len={pattern_len}, count={count}")
                return True
        return False


_CHECK_ITEM_REGISTRY: Dict[str, GarbledTextCheckItem] = {
    "empty_text": EmptyTextCheckItem(),
    "repeated_char": RepeatedCharCheckItem(),
    "normal_char_ratio": NormalCharRatioCheckItem(),
    "control_char": ControlCharCheckItem(),
    "repeated_pattern": RepeatedPatternCheckItem(),
}


def get_all_check_item_names() -> List[str]:
    """获取所有可用的检查项名称"""
    return list(_CHECK_ITEM_REGISTRY.keys())


class TestCaseConfig(BaseModel):
    """测试用例配置"""

    message: Annotated[str, AfterValidator(non_empty_string), AfterValidator(validate_str_length())] = Field(
        description="Test message, must be a non-empty string"
    )
    items: List[str] = Field(
        default_factory=lambda: list(_CHECK_ITEM_REGISTRY.keys()),
        description="List of check item names to enable for this test case. If not specified, all items are enabled.",
    )

    @field_validator('items', mode='after')
    @classmethod
    def validate_items(cls, v: List[str]) -> List[str]:
        available_items = get_all_check_item_names()
        for item_name in v:
            if item_name not in available_items:
                raise SchemaValidateError(
                    f"Unknown check item: '{item_name}'. Available items: {available_items}",
                    action=f"Please use one of the available check items: {available_items}",
                )
        return v


def parse_test_case(value: Any) -> TestCaseConfig:
    """解析测试用例，只支持字典格式：{"message": "...", "items": [...]}"""
    if not isinstance(value, dict):
        raise ValueError("Test case must be a dict with 'message' and 'items' fields")

    message = value.get("message")
    if not message:
        raise ValueError("Test case must contain 'message' field")

    items = value.get("items")
    if items is None:
        items = list(_CHECK_ITEM_REGISTRY.keys())
    elif isinstance(items, str):
        items = [items]
    elif not isinstance(items, list):
        raise ValueError(f"Items must be a list. Available: {get_all_check_item_names()}")

    return TestCaseConfig(message=str(message), items=items)


class GarbledTextPrecheckConfig(BasePrecheckConfig):
    """基本乱码检测预检查配置"""

    type: Literal["garbled_text"] = "garbled_text"
    test_cases: Optional[List[TestCaseConfig]] = Field(
        default=None, description="List of test cases. Each case contains a message and enabled check items."
    )

    @field_validator('test_cases', mode='before')
    @classmethod
    def parse_test_cases(cls, v: Any) -> List[TestCaseConfig]:
        if v is None:
            return [TestCaseConfig(message="How are you?", items=list(_CHECK_ITEM_REGISTRY.keys()))]
        if isinstance(v, list):
            return [parse_test_case(item) for item in v]
        return [parse_test_case(v)]

    @field_validator('test_cases', mode='after')
    @classmethod
    def validate_test_cases(cls, v: List[TestCaseConfig]) -> List[TestCaseConfig]:
        if not v:
            raise SchemaValidateError(
                "test_cases must contain at least one test case", action="Please provide at least one test case"
            )
        return v


class GarbledTextRule(BasePrecheckRule):
    """基本乱码检测预检查规则，基于配置针对不同的文本进行多检查项检测"""

    def __init__(self, config: GarbledTextPrecheckConfig):
        super().__init__(config)
        self.config: GarbledTextPrecheckConfig = config

    def is_garbled_text(self, text: str, check_items: List[str]) -> bool:
        for item_name in check_items:
            check_item = _CHECK_ITEM_REGISTRY.get(item_name)
            if check_item and check_item.check(text):
                get_logger().debug(f"Garbled text detected by check item: {item_name}")
                return True
        return False

    def check(
        self, host: str, port: int, served_model_name: str, datasets: List[str]
    ) -> Optional[List[EvaluateAccuracy]]:
        """执行乱码检测预检查"""
        if not self.config.test_cases:
            get_logger().warning("No test cases configured for garbled text precheck. Skipping.")
            return None

        get_logger().info(
            f"Testing model with {len(self.config.test_cases)} test case(s) for garbled text detection..."
        )

        for idx, test_case in enumerate(self.config.test_cases, 1):
            try:
                get_logger().debug(
                    f"Running garbled text test case {idx}/{len(self.config.test_cases)}: "
                    f"message='{test_case.message}', items={test_case.items}"
                )

                test_response = self.test_chat_via_api(
                    host=host,
                    port=port,
                    served_model_name=served_model_name,
                    test_message=test_case.message,
                    max_tokens=self.config.max_tokens,
                )
                get_logger().debug(f"test_response: {test_response}")

                if self.is_garbled_text(test_response, test_case.items):
                    get_logger().warning(
                        f"Garbled text test case {idx} failed: Garbled output detected for message "
                        f"'{test_case.message}' with check items {test_case.items}. Skipping evaluation stage."
                    )
                    return [EvaluateAccuracy(dataset=dataset, accuracy=Decimal('0')) for dataset in datasets]

                get_logger().debug(f"Garbled text test case {idx} passed")

            except Exception as e:
                get_logger().warning(
                    f"Failed to execute garbled text test case {idx} "
                    f"(message: '{test_case.message}', error: {e}). Proceeding with evaluation."
                )
                continue

        get_logger().info("All garbled text test cases passed. Proceeding with evaluation.")
        return None


def get_plugin():
    """
    获取 garbled_text 预检查插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (GarbledTextPrecheckConfig, GarbledTextRule) 元组
    """
    return GarbledTextPrecheckConfig, GarbledTextRule
