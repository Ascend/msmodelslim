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
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import Field
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import get_current_context
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoProcessorConfigList, AutoSessionProcessor
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import UnexpectedError
from msmodelslim.processor.analysis.binary_operator.metrics.factory import BinaryAnalysisMethodFactory


class BinaryAnalysisProcessorConfig(AutoProcessorConfig):
    """Configuration for binary layer sensitivity analysis (attention_mse)."""

    type: Literal["binary_analysis"] = "binary_analysis"
    metrics: str = Field(
        default="attention_mse",
        description="Analysis method: attention_mse",
    )
    patterns: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Layer name patterns to analyze",
    )
    configs: AutoProcessorConfigList = Field(
        default_factory=list,
        description="List of quant sub-processor configs",
    )


@QABCRegistry.register(dispatch_key=BinaryAnalysisProcessorConfig, abc_class=AutoSessionProcessor)
class BinaryAnalysisProcessor(AutoSessionProcessor):
    """
    Layer sensitivity analysis using binary activation (two forwards, e.g. float vs quant; MSE).
    preprocess: register hook, first forward; postprocess: quant, second forward, remove hook, compute_score;
    post_run: write layer_scores to ctx['layer_analysis'].state['layer_scores'].
    """

    def __init__(
        self,
        model: nn.Module,
        config: BinaryAnalysisProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.quant_processors = [AutoSessionProcessor.from_config(model, cfg, adapter) for cfg in config.configs]
        self._analysis_method = BinaryAnalysisMethodFactory.create_method(
            config.metrics, adapter=self.adapter
        )
        self._target_layers: List[str] = []
        self._float_layer_stats: Dict[str, Any] = {}
        self._quant_layer_stats: Dict[str, Any] = {}
        self._layer_scores: List[Dict[str, Any]] = []
        self._hook_handles: Dict[str, Any] = {}

    def pre_run(self) -> None:
        # 校验上下文机制
        ctx = get_current_context()
        if ctx is None:
            raise UnexpectedError("No context is working.")

        for processor in self.quant_processors:
            processor.pre_run()

    def preprocess(self, request: BatchProcessRequest) -> None:
        all_layers = self._analysis_method.get_target_layers(request.module, request.name)
        self._target_layers = self._analysis_method.filter_layers_by_patterns(
            all_layers, self.config.patterns
        )
        get_logger().debug(
            "BinaryAnalysisProcessor preprocess: %d target layers (metrics=%s)",
            len(self._target_layers),
            self._analysis_method.name,
        )

        if len(self._target_layers) == 0:
            get_logger().warning(
                f"No target layers/modules found matching the specified patterns for {request.name}. "
                f"Please check the patterns {self.config.patterns} and the model structure, to ensure it meets expectations."
                )

        hook_fn = self._analysis_method.get_hook()
        self._register_hooks_for_request(request, hook_fn, self._float_layer_stats)

        # 跑一次前向，采集浮点权重的激活状态
        super().process(request)

        # 移除钩子，但不计算敏感分数
        keys_to_remove = [k for k in self._hook_handles if k == request.name or k.startswith(request.name + ".")]
        for k in keys_to_remove:
            handle = self._hook_handles.pop(k, None)
            if handle is not None:
                handle.remove()

    def process(self, request: BatchProcessRequest) -> None:
        for processor in self.quant_processors:
            processor.preprocess(request)
            processor.process(request)
            processor.postprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        hook_fn = self._analysis_method.get_hook()
        self._register_hooks_for_request(request, hook_fn, self._quant_layer_stats)

        # 再跑一次前向，采集量化权重的激活状态
        super().process(request)

        # 移除当前块下所有已注册的 hook，并计算各叶子层的 score
        keys_to_remove = [k for k in self._hook_handles if k == request.name or k.startswith(request.name + ".")]
        for k in keys_to_remove:
            handle = self._hook_handles.pop(k, None)
            if handle is not None:
                handle.remove()
            if k in self._float_layer_stats and k in self._quant_layer_stats and k in self._target_layers:
                score = self._analysis_method.compute_score(self._float_layer_stats[k], self._quant_layer_stats[k])
                self._layer_scores.append({"name": k, "score": score})
                get_logger().debug(f"{k}: {score}")
                # 分数计算完成后删除激活状态，及时释放内存
                del self._float_layer_stats[k]
                del self._quant_layer_stats[k]

    def post_run(self) -> None:
        for processor in self.quant_processors:
            processor.post_run()

        ctx = get_current_context()
        ctx["layer_analysis"].debug["layer_scores"] = self._layer_scores
        ctx["layer_analysis"].debug["method"] = self._analysis_method.name
        ctx["layer_analysis"].debug["patterns"] = self.config.patterns

        get_logger().info(
            "BinaryAnalysisProcessor post_run: %d layer scores computed (%s)",
            len(self._layer_scores),
            self._analysis_method.name,
        )

        if not ctx["layer_analysis"].debug["layer_scores"] or len(ctx["layer_analysis"].debug["layer_scores"]) == 0:
            get_logger().warning(
                f"No statistics collected. This may be caused by empty calibration data "
                f"or incompatible patterns with the model structure."
                )

    def _register_hooks_for_request(self, 
                                    request: BatchProcessRequest, 
                                    hook_fn: Callable, 
                                    stats_dict: Dict[str, Any]) -> None:
        """
        为当前 request 块中属于 _target_layers 且非 nn.Linear 的子模块注册前向 hook。
        采集到的激活状态存储到stats_dict中。
        """
        for sub_name, sub_module in request.module.named_modules(prefix=request.name):
            if sub_name not in self._target_layers:
                continue
            if isinstance(sub_module, nn.Linear):
                continue
            bound_hook = functools.partial(
                hook_fn,
                layer_name=sub_name,
                stats_dict=stats_dict,
            )
            handle = sub_module.register_forward_hook(bound_hook)
            self._hook_handles[sub_name] = handle
