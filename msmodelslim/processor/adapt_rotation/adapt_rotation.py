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

from typing import Literal, Union, Any

from torch import nn
from pydantic import Field, model_validator

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger

from .adapt_rotation_stage1 import AdaptRotationStage1Processor, AdaptRotationStage1ProcessorConfig
from .adapt_rotation_stage2 import AdaptRotationStage2Processor, AdaptRotationStage2ProcessorConfig


class AdaptRotationProcessorConfig(AutoProcessorConfig):
    """自适应旋转（adapt_rotation）处理器配置。

    位于 `spec.process[]`，由 `type: adapt_rotation` 分派；通过 `stage`（1 或 2）选择
    具体的旋转适配阶段，对应字段放在 `stage_config` 嵌套对象中。stage1 做激活分布
    适配与量化适配，stage2 在 stage1 基础上支持在线旋转。
    """

    type: Literal["adapt_rotation"] = Field(
        default="adapt_rotation", description="处理器类型，固定为 `adapt_rotation`。"
    )
    stage: Literal[1, 2] = Field(description="旋转适配阶段：1 或 2，决定使用哪个阶段配置。")
    stage_config: Union[AdaptRotationStage1ProcessorConfig, AdaptRotationStage2ProcessorConfig] = Field(
        description="阶段配置对象（内部自动组装字段，必选但由 before-validator 根据 `stage` 自动生成，用户无需在 YAML 中配置）；YAML 中不要直接配置该字段，请把阶段字段（如 steps、quant_dtype）平铺在处理器下，见《AdaptRotationStage1ProcessorConfig 配置说明》/《AdaptRotationStage2ProcessorConfig 配置说明》。"
    )

    @model_validator(mode='before')
    @classmethod
    def _build_stage_config(cls, data: Any) -> Any:
        """按 stage 把扁平字段组装为 stage_config：仅允许对应阶段字段，多余字段报错。"""
        if not isinstance(data, dict):
            return data
        stage_val = data.get("stage")
        if data.get("type") != "adapt_rotation" or stage_val not in (1, 2):
            return data
        s1 = set(AdaptRotationStage1ProcessorConfig.model_fields) - {"type"}
        s2 = set(AdaptRotationStage2ProcessorConfig.model_fields) - {"type"}
        allowed = (s1 | {"type", "stage"}) if stage_val == 1 else (s2 | {"type", "stage"})
        disallowed = set(data) - allowed
        if disallowed:
            raise SchemaValidateError(
                f"stage={stage_val} allows only {sorted(allowed)}; got extra: {sorted(disallowed)}"
            )
        if stage_val == 1:
            stage_config = AdaptRotationStage1ProcessorConfig.model_validate(
                {"type": "_adapt_rotation_stage1", **{k: data[k] for k in s1 if k in data}}
            )
        else:
            stage_config = AdaptRotationStage2ProcessorConfig.model_validate(
                {"type": "_adapt_rotation_stage2", **{k: data[k] for k in s2 if k in data}}
            )
        return {"type": "adapt_rotation", "stage": stage_val, "stage_config": stage_config}

    def __getattr__(self, name: str) -> Any:
        if name in ("type", "stage", "stage_config"):
            raise AttributeError(name)
        return getattr(self.stage_config, name)


@QABCRegistry.register(dispatch_key=AdaptRotationProcessorConfig, abc_class=AutoSessionProcessor)
class AdaptRotationProcessor(AutoSessionProcessor):
    """
    Upper-level processor for adapt_rotation: identified by type="adapt_rotation".
    Dispatches to AdaptRotationStage1Processor or AdaptRotationStage2Processor
    based on the stage field in config.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdaptRotationProcessorConfig,
        adapter: object,
        **kwargs,
    ) -> None:
        super().__init__(model)
        self.config = config
        if config.stage == 1:
            self._inner = AdaptRotationStage1Processor(model, config.stage_config, adapter, **kwargs)
        else:
            self._inner = AdaptRotationStage2Processor(model, config.stage_config, adapter, **kwargs)
        get_logger().debug("AdaptRotationProcessor delegating to %s", self._inner.__class__.__name__)

    def support_distributed(self) -> bool:
        return self._inner.support_distributed()

    def is_data_free(self) -> bool:
        return self._inner.is_data_free()

    def need_kv_cache(self):
        return self._inner.need_kv_cache()

    def preprocess(self, request: BatchProcessRequest) -> None:
        self._inner.preprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._inner.postprocess(request)

    def pre_run(self) -> None:
        self._inner.pre_run()

    def post_run(self) -> None:
        self._inner.post_run()

    def process(self, request: BatchProcessRequest) -> None:
        self._inner.process(request)
