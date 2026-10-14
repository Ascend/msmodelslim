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

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, AfterValidator
from typing import Annotated

from msmodelslim.core.practice import Metadata
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.processor.base import AutoProcessorConfigList
from msmodelslim.utils.validation.pydantic import validate_str_length


class QuantizationConfig(BaseModel):
    """
    量化配置构造产物

    对应 YAML 中的完整量化配置结构，包含 apiversion、metadata、spec（process/dataset/save）
    等组件，作为 Builder 的 build() 产出。当前仅支持 modelslim_v1，spec 为 ModelslimV1ServiceConfig。
    """

    apiversion: str = Field(description="量化服务 API 版本，如 modelslim_v1")
    metadata: Metadata = Field(default_factory=Metadata, description="配置元数据：config_id、label 等")
    spec: ModelslimV1ServiceConfig = Field(description="量化规格：process、dataset、save、runner")


class TuningSearchSpace(BaseModel):
    """
    调优搜索空间

    限定策略在调优时的搜索范围，不同策略使用其中不同子集
    新增约束时增加可选字段或通过 extra 传递，无需破坏已有逻辑。

    当前/预留字段示例：
    - anti_outlier_strategies: 离群值抑制策略候选列表（摸高算法使用）
    - allowed_datasets: 允许使用的数据集名称列表（预留）
    - max_rollback_layers: 回退层数上限（预留）
    """

    model_config = ConfigDict(extra="allow")

    anti_outlier_strategies: Optional[List[AutoProcessorConfigList]] = Field(
        default=None, description="离群值抑制策略候选（每项为 AutoProcessorConfigList），摸高算法使用"
    )
    allowed_datasets: Optional[List[Annotated[str, AfterValidator(validate_str_length())]]] = Field(
        default=None, description="允许的数据集列表"
    )
    max_rollback_layers: Optional[int] = Field(default=None, description="最大回退层数")
