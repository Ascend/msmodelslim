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
import functools
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import get_current_context
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.processor.analysis.unary_operator.metrics.factory import UnaryAnalysisMethodFactory
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import UnexpectedError


class UnaryAnalysisProcessorConfig(AutoProcessorConfig):
    """Configuration for unary layer sensitivity analysis (std/quantile/kurtosis)."""

    type: Literal["unary_analysis"] = "unary_analysis"
    metrics: str = Field(
        default="kurtosis",
        description="Analysis method: quantile | std | kurtosis",
    )
    patterns: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Layer name patterns to analyze",
    )


@QABCRegistry.register(dispatch_key=UnaryAnalysisProcessorConfig, abc_class=AutoSessionProcessor)
class UnaryAnalysisProcessor(AutoSessionProcessor):
    """
    Layer sensitivity analysis using unary activation (single forward; std/quantile/kurtosis).
    preprocess: register hook; process: forward; postprocess: remove hook, compute_score;
    post_run: write layer_scores to ctx['layer_analysis'].state['layer_scores'].
    """

    def __init__(
        self,
        model: nn.Module,
        config: UnaryAnalysisProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self._analysis_method = UnaryAnalysisMethodFactory.create_method(config.metrics)
        self._target_layers: List[str] = []
        self._layer_stats: Dict[str, Any] = {}
        self._layer_scores: List[Dict[str, Any]] = []
        self._hook_handles: Dict[str, Any] = {}

    def pre_run(self) -> None:
        # 校验上下文机制
        ctx = get_current_context()
        if ctx is None:
            raise UnexpectedError("No context is working.")

    def preprocess(self, request: BatchProcessRequest) -> None:
        all_layers = self._analysis_method.get_target_layers(request.module, request.name)
        self._target_layers = self._analysis_method.filter_layers_by_patterns(
            all_layers, self.config.patterns
        )
        get_logger().debug(
            "UnaryAnalysisProcessor preprocess: %d target layers (metrics=%s)",
            len(self._target_layers),
            self._analysis_method.name,
        )

        if len(self._target_layers) == 0:
            get_logger().warning(
                f"No target layers/modules found matching the specified patterns for {request.name}. "
                f"Please check the patterns {self.config.patterns} and the model structure, to ensure it meets expectations."
                )

        # Runner 按块下发 request（如 request.name='model.layers.0'），target_layers 是叶子 Linear 全名
        # 遍历当前块下属于 _target_layers 的 nn.Linear 子模块，逐个注册 hook
        hook_fn = self._analysis_method.get_hook()
        for sub_name, sub_module in request.module.named_modules(prefix=request.name):
            if sub_name not in self._target_layers:
                continue
            if not isinstance(sub_module, nn.Linear):
                continue
            bound_hook = functools.partial(
                hook_fn,
                layer_name=sub_name,
                stats_dict=self._layer_stats,
            )
            handle = sub_module.register_forward_hook(bound_hook)
            self._hook_handles[sub_name] = handle

    def process(self, request: BatchProcessRequest) -> None:
        super().process(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        # 移除当前块下所有已注册的 hook，并计算各叶子层的 score
        keys_to_remove = [k for k in self._hook_handles if k == request.name or k.startswith(request.name + ".")]
        for k in keys_to_remove:
            handle = self._hook_handles.pop(k, None)
            if handle is not None:
                handle.remove()
            if k in self._layer_stats and k in self._target_layers:
                score = self._analysis_method.compute_score(self._layer_stats[k])
                self._layer_scores.append({"name": k, "score": score})
                get_logger().debug(f"{k}: {score}")
                # 分数计算完成后删除激活状态，及时释放内存
                del self._layer_stats[k]

    def post_run(self) -> None:
        ctx = get_current_context()
        ctx["layer_analysis"].debug["layer_scores"] = self._layer_scores
        ctx["layer_analysis"].debug["method"] = self._analysis_method.name
        ctx["layer_analysis"].debug["patterns"] = self.config.patterns

        get_logger().info(
            "UnaryAnalysisProcessor post_run: %d layer scores computed (%s)",
            len(self._layer_scores),
            self._analysis_method.name,
        )

        if not ctx["layer_analysis"].debug["layer_scores"] or len(ctx["layer_analysis"].debug["layer_scores"]) == 0:
            get_logger().warning(
                f"No statistics collected. This may be caused by empty calibration data "
                f"or incompatible patterns with the model structure."
                )

    def get_layer_scores(self) -> List[Dict[str, Any]]:
        """Return the computed layer scores (for tests or when context is not used)."""
        return self._layer_scores
