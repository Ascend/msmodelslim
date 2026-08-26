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

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field, AfterValidator
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import get_current_context
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.analysis.binary_operator_layer_wise.metrics.factory import LayerWiseMethodFactory
from msmodelslim.processor.analysis.distributed_utils import (
    check_distributed_analysis_supported,
    maybe_barrier_before_linear_quant,
    publish_layer_analysis_result,
)
from msmodelslim.processor.base import AutoProcessorConfig, AutoProcessorConfigList, AutoSessionProcessor
from msmodelslim.utils.exception import UnexpectedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.validation.pydantic import validate_str_length


logger = get_logger()


class BinaryOperatorLayerWiseProcessorConfig(AutoProcessorConfig):
    """逐层敏感度分析处理器配置（对比逐块浮点与量化输出）。"""

    type: Literal["binary_operator_layer_wise"] = Field(
        default="binary_operator_layer_wise", description="处理器类型，固定为 `binary_operator_layer_wise`。"
    )
    metrics: str = Field(
        default="mse_layer_wise",
        description="分析指标，如 `mse_layer_wise`。",
    )
    quant_modules: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=lambda: ["*"],
        description=(
            "与 linear_quant.include、CLI --quant_modules 一致（YAML 占位 ${quant_modules}）；"
            "用于层敏感结果展示名后缀，如 model.layers.2 (mod1, mod2)。"
        ),
    )
    configs: AutoProcessorConfigList = Field(
        default_factory=list,
        description="用于执行量化-反量化路径的量化子处理器配置列表",
    )


@QABCRegistry.register(dispatch_key=BinaryOperatorLayerWiseProcessorConfig, abc_class=AutoSessionProcessor)
class BinaryOperatorLayerWiseProcessor(AutoSessionProcessor):
    """Per-block float vs quant output comparison."""

    def __init__(
        self,
        model: nn.Module,
        config: BinaryOperatorLayerWiseProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.quant_processors = [AutoSessionProcessor.from_config(model, cfg, adapter) for cfg in config.configs]
        self._analysis_method = LayerWiseMethodFactory.create_method(config.metrics, adapter=self.adapter)

        self._layer_scores: List[Dict[str, Any]] = []

    def support_distributed(self) -> bool:
        return all(processor.support_distributed() for processor in self.quant_processors)

    def pre_run(self) -> None:
        ctx = get_current_context()
        if ctx is None:
            raise UnexpectedError("No context is working.")
        check_distributed_analysis_supported(
            self._analysis_method.supports_distributed,
            self._analysis_method.name,
        )
        for processor in self.quant_processors:
            processor.pre_run()

    def process(self, request: BatchProcessRequest) -> None:
        maybe_barrier_before_linear_quant()

        # Safer restore: snapshot float parameters/buffers on CPU to avoid device memory spike.
        float_state = {k: v.detach().cpu().clone() for k, v in request.module.state_dict().items()}

        # Quant forward (same inputs)
        for processor in self.quant_processors:
            processor.preprocess(request)
            processor.process(request)
            processor.postprocess(request)

        self._run_forward_if_need(request)
        quant_outputs = list(request.outputs)

        # Restore float params/buffers for current block.
        try:
            request.module.load_state_dict(float_state, strict=False)
        except Exception as exc:
            raise UnexpectedError(
                "Failed to restore float state_dict after quant forward. "
                "This may indicate the quant processors changed module structure."
            ) from exc
        finally:
            del float_state

        # Float forward (same inputs)
        self._run_forward_if_need(request)
        float_outputs = list(request.outputs)

        # Compute score for current block immediately
        score = self._analysis_method.compute_score(float_outputs, quant_outputs)
        self._layer_scores.append({"name": request.name, "score": score})

        del quant_outputs, float_outputs

    def post_run(self) -> None:
        for processor in self.quant_processors:
            processor.post_run()

        self._layer_scores = publish_layer_analysis_result(
            list(self._layer_scores),
            self._analysis_method.name,
            quant_modules=self.config.quant_modules,
        )

        logger.info(
            "BinaryOperatorLayerWiseProcessor post_run: %d layer scores (%s), quant_modules=%s",
            len(self._layer_scores),
            self._analysis_method.name,
            self.config.quant_modules,
        )

        if not self._layer_scores:
            get_logger().warning(
                "No statistics collected. This may be caused by empty calibration data "
                "or the processor not running on any blocks."
            )
