#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from enum import Enum
from typing import Annotated, Dict, List, Optional

from pydantic import Field, BaseModel, AfterValidator, model_validator, model_serializer, ValidationError
from pydantic_core import PydanticCustomError

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.utils.validation.pydantic import validate_str_length, in_range


class ScenarioTagMatch(str, Enum):
    """
    Scenario tag match result for practice selection.

    - NO_MATCH: verified scenarios are absent or none matches requested tags
    - MATCH: found a scenario that contains all requested tags
    - STANDBY: there are verified scenarios, but none matches; can be used as standby config
    """

    NO_MATCH = "no_match"
    MATCH = "match"
    STANDBY = 'standby'


class Metadata(BaseModel):
    """量化配置元数据：标识配置的 ID、评分、标签与已验证的模型/场景。"""

    config_id: Annotated[str, AfterValidator(validate_str_length())] = Field(
        default='Unknown', description="量化配置 ID，例如 'Qwen3-32B W8A8'"
    )
    score: Annotated[float, AfterValidator(in_range(min_val=0))] = Field(
        default=100.0, description="量化配置评分，用于排序，必须 >= 0"
    )
    label: dict = Field(
        default_factory=dict,
        description="量化配置标签，用于过滤，例如 {'w_bit': 8, 'a_bit': 8, 'is_sparse': True, 'kv_cache': True}",
    )
    verified_model_types: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=list, description="已验证的模型类型列表，例如 ['LLaMa3.1-70B', 'Qwen2.5-72B']"
    )
    verified_tags: Dict[str, List[List[str]]] = Field(
        default_factory=dict,
        description="已验证场景标签：键为模型类型，值为场景标签列表（每个场景是一组标签，如 ['MindIE','Atlas_A2_Inference']）",
    )

    def matches_scenario_tags(self, model_type: str, scenario_tags: Optional[List[str]]) -> ScenarioTagMatch:
        """
        Match scenario tags against verified tags for a model.

        Returns:
            - ScenarioTagMatch.NO_MATCH: no verified scenarios are available for this model_type.
            - ScenarioTagMatch.MATCH: at least one verified scenario contains all requested scenario_tags.
            - ScenarioTagMatch.STANDBY: verified scenarios exist, but none match requested tags.
        """
        scenarios = self.verified_tags.get(model_type, [])
        if not scenarios:
            return ScenarioTagMatch.NO_MATCH
        if not scenario_tags:
            return ScenarioTagMatch.MATCH

        user_lower = [t.lower() for t in scenario_tags]
        for scenario_tags_list in scenarios:
            scenario_lower = [str(t).lower() for t in scenario_tags_list]
            if all(ut in scenario_lower for ut in user_lower):
                return ScenarioTagMatch.MATCH
        return ScenarioTagMatch.STANDBY


class PracticeConfig(BaseModel):
    """完整最佳实践量化任务配置：metadata + task（task ≡ apiversion + spec）。

    由自动调优策略生成/使用的实践配置。
    概念上 practice = metadata + task，实现上 task 为 BaseQuantConfig（插件判别基类）。
    """

    metadata: Metadata = Field(
        default_factory=Metadata, description="量化配置元数据（config_id/score/label/verified_*）"
    )
    task: BaseQuantConfig = Field(default_factory=BaseQuantConfig, description="量化任务描述（apiversion + spec）")

    @model_validator(mode='before')
    @classmethod
    def _normalize(cls, value):
        """预处理：旧格式 {apiversion, metadata, spec, ...} → {metadata, task:{apiversion, spec, ...}}。"""
        if isinstance(value, dict) and 'task' not in value:
            task = {k: v for k, v in value.items() if k != 'metadata'}
            task.setdefault('apiversion', 'Unknown')
            value = {'metadata': value.get('metadata', {}), 'task': task}
        return value

    @model_validator(mode='wrap')
    @classmethod
    def _strip_task_prefix(cls, value, handler):
        """剥除聚合后 task. 前缀，使错误路径与 yaml 的 spec. 一致。"""
        try:
            return handler(value)
        except ValidationError as e:
            errors = e.errors()
            line_errors = []
            for err in errors:
                loc = list(err['loc'])
                if loc and loc[0] == 'task':
                    err['loc'] = tuple(loc[1:])
                # from_exception_data 重建仅接受内置错误类型；自定义类型（如
                # invalid_processor_item）需用 PydanticCustomError 包装以保留消息
                line_errors.append(
                    {
                        'loc': err['loc'],
                        'type': PydanticCustomError(err['type'], err['msg'], err.get('ctx')),
                    }
                )
            raise ValidationError.from_exception_data(title=e.title, line_errors=line_errors)

    @model_serializer(mode='wrap')
    def _flatten(self, handler, info):
        """序列化时展平：{metadata, task} → {metadata, apiversion, spec, ...}，保持导出兼容。"""
        data = handler(self)
        task = data.pop('task', {})
        data.update(task)
        return data

    def extract_quant_config(self) -> BaseQuantConfig:
        """提取量化任务配置（apiversion + spec，不含 metadata）。"""
        return self.task

    @property
    def apiversion(self) -> str:
        """兼容旧访问：delegate 到 task.apiversion。"""
        return self.task.apiversion

    @property
    def spec(self) -> object:
        """兼容旧访问：delegate 到 task.spec。"""
        return self.task.spec

    @spec.setter
    def spec(self, value: object) -> None:
        """兼容旧写入：delegate 到 task.spec。"""
        self.task.spec = value
