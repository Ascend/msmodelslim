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
# pylint: disable=consider-using-in,logging-fstring-interpolation

from decimal import Decimal
from typing import List, Optional, Union, Any, Literal, Annotated

from pydantic import BaseModel, Field, field_validator, AfterValidator

from msmodelslim.core.tune_strategy import EvaluateAccuracy
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.validation.pydantic import validate_str_length, non_empty_string

from .base import BasePrecheckRule, BasePrecheckConfig


class TestCase(BaseModel):
    """预检查测试用例"""

    message: Annotated[str, AfterValidator(non_empty_string), AfterValidator(validate_str_length(max_len=4096))] = (
        Field(description="Test message, must be a non-empty string")
    )
    expected_answer: Optional[
        Union[
            Annotated[str, AfterValidator(validate_str_length())],
            List[Annotated[str, AfterValidator(validate_str_length())]],
        ]
    ] = Field(
        default=None,
        description="Expected content that should be included in the response. Can be a string or a list of strings.",
    )


def parse_test_case(value: Any) -> TestCase:
    """解析测试用例，支持字典格式（键值对）"""
    if isinstance(value, dict):
        if "message" in value:
            expected_answer = value.get("expected_answer")
            if expected_answer == "" or expected_answer == []:
                expected_answer = None
            return TestCase(message=value["message"], expected_answer=expected_answer)
        if len(value) == 1:
            message, expected_answer = next(iter(value.items()))
            if expected_answer == "" or expected_answer == []:
                expected_answer = None
            return TestCase(message=message, expected_answer=expected_answer)
        raise ValueError(
            f"Invalid test case format: {value}. Must contain exactly one key-value pair or message/expected_answer keys"
        )
    raise ValueError(
        f"Invalid test case type: {type(value)}. Supported formats: dict (key-value or message/expected_answer)"
    )


def _default_expected_answer_test_cases() -> List[TestCase]:
    """返回默认的期望答案验证测试用例"""
    return [TestCase(message="What is 2+2?", expected_answer="4")]


class ExpectedAnswerPrecheckConfig(BasePrecheckConfig):
    """期望答案验证预检查配置"""

    type: Literal["expected_answer"] = "expected_answer"
    test_cases: List[TestCase] = Field(
        default_factory=_default_expected_answer_test_cases,
        min_length=1,
        description="List of test cases in dictionary format (key-value pairs), English only. Must contain at least one test case.",
    )

    @field_validator('test_cases', mode='before')
    @classmethod
    def parse_test_cases(cls, v: Any) -> List[TestCase]:
        if isinstance(v, list):
            return [parse_test_case(item) for item in v]
        elif v is not None:
            return [parse_test_case(v)]
        return v

    @field_validator('test_cases', mode='after')
    @classmethod
    def validate_test_cases(cls, v: List[TestCase]) -> List[TestCase]:
        if not v:
            raise SchemaValidateError(
                "test_cases must contain at least one test case", action="Please provide at least one test case"
            )
        return v


class ExpectedAnswerRule(BasePrecheckRule):
    """期望答案验证预检查规则，验证模型输出是否包含期望的答案内容"""

    def __init__(self, config: ExpectedAnswerPrecheckConfig):
        super().__init__(config)
        self.config: ExpectedAnswerPrecheckConfig = config

    def contains_expected_answer(self, response: str, expected_answer: Union[str, List[str]]) -> bool:
        """检查回答中是否包含期待的内容"""
        if expected_answer is None:
            return True

        expected_list = [expected_answer] if isinstance(expected_answer, str) else expected_answer
        response_lower = response.lower()
        return any(exp.lower() in response_lower for exp in expected_list)

    def check(
        self, host: str, port: int, served_model_name: str, datasets: List[str]
    ) -> Optional[List[EvaluateAccuracy]]:
        """
        执行期望答案验证预检查。

        对配置中的所有测试用例进行验证，如果任何一个测试用例失败，则预检查失败。

        如果检查失败，返回不达标的结果列表；如果检查通过，返回 None
        """
        if not self.config.test_cases:
            get_logger().warning("No test cases configured for expected answer precheck. Skipping.")
            return None

        get_logger().info(
            f"Testing model with {len(self.config.test_cases)} test case(s) for expected answer validation..."
        )

        # 遍历所有测试用例
        for idx, test_case in enumerate(self.config.test_cases, 1):
            try:
                get_logger().debug(
                    f"Running expected answer test case {idx}/{len(self.config.test_cases)}: {test_case.message}"
                )
                test_response = self.test_chat_via_api(
                    host=host,
                    port=port,
                    served_model_name=served_model_name,
                    test_message=test_case.message,
                    max_tokens=self.config.max_tokens,
                )
                get_logger().debug(f"test_response: {test_response}")

                # 检查期望答案
                if test_case.expected_answer is None:
                    get_logger().debug(f"Expected answer test case {idx} has no expected_answer set. Skipping.")
                    continue

                has_expected_answer = self.contains_expected_answer(test_response, test_case.expected_answer)
                if not has_expected_answer:
                    get_logger().warning(
                        f"Expected answer test case {idx} failed: Expected answer not found in response for message '{test_case.message}'. "
                        f"Expected: {test_case.expected_answer}. Skipping evaluation stage."
                    )
                    return [EvaluateAccuracy(dataset=dataset, accuracy=Decimal('0')) for dataset in datasets]
                get_logger().debug(f"Expected answer test case {idx} passed: Expected answer found in response")

            except Exception as e:
                get_logger().warning(
                    f"Failed to execute expected answer test case {idx} (message: '{test_case.message}', error: {e}). "
                    "Proceeding with evaluation."
                )
                # 如果测试用例执行失败，继续执行下一个测试用例
                continue

        get_logger().info("All expected answer test cases passed. Proceeding with evaluation.")
        return None


def get_plugin():
    """
    获取 expected_answer 预检查插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (ExpectedAnswerPrecheckConfig, ExpectedAnswerRule) 元组
    """
    return ExpectedAnswerPrecheckConfig, ExpectedAnswerRule
