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
from msmodelslim.processor.analysis.binary_operator_layer_wise.metrics.factory import LayerWiseMethodFactory
from msmodelslim.processor.base import AutoProcessorConfig, AutoProcessorConfigList, AutoSessionProcessor
from msmodelslim.utils.exception import UnexpectedError
from msmodelslim.utils.logging import get_logger


logger = get_logger()


class BinaryOperatorLayerWiseProcessorConfig(AutoProcessorConfig):
    """Configuration for decoder-layer level sensitivity analysis (mse_layer_wise)."""

    type: Literal["layer_analysis"] = "layer_analysis"
    metrics: str = Field(
        default="mse_layer_wise",
        description="Analysis method for layer-level sensitivity, currently only 'mse_layer_wise' is supported.",
    )
    patterns: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Decoder layer name patterns to analyze, e.g. 'model.layers.*'.",
    )
    configs: AutoProcessorConfigList = Field(
        default_factory=list,
        description="List of quant sub-processor configs used to build W4A4_MXFP4 quantization context.",
    )


@QABCRegistry.register(dispatch_key=BinaryOperatorLayerWiseProcessorConfig, abc_class=AutoSessionProcessor)
class BinaryOperatorLayerWiseProcessor(AutoSessionProcessor):
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
        self._analysis_method = LayerWiseMethodFactory.create_method(
            config.metrics,
            adapter=self.adapter,
        )
        self._target_decoder_layers: List[str] = []
        self._target_linear_names: List[str] = []
        self._float_layer_stats: Dict[str, Any] = {}
        self._quant_layer_stats: Dict[str, Any] = {}
        self._decoder_layer_scores: List[Dict[str, Any]] = []
        self._hook_handles: Dict[str, Any] = {}

    def pre_run(self) -> None:
        ctx = get_current_context()
        if ctx is None:
            raise UnexpectedError("No context is working.")

        for processor in self.quant_processors:
            processor.pre_run()

    def preprocess(self, request: BatchProcessRequest) -> None:
        model = request.module
        self._target_decoder_layers = self._analysis_method.filter_layers_by_patterns(
            [request.name], self.config.patterns
        )
        if not self._target_decoder_layers:
            return

        logger.debug(
            "LayerAnalysisProcessor preprocess: decoder layer %s (metrics=%s)",
            self._target_decoder_layers[0],
            self._analysis_method.name,
        )

        if len(self._target_decoder_layers) == 0:
            logger.warning(
                f"No target layers found matching the specified patterns for {request.name}. "
                f"Please check the patterns {self.config.patterns} and the model structure, to ensure it meets expectations."
                )

        hook_fn = self._analysis_method.get_hook()

        for decoder_name in self._target_decoder_layers:
            self._target_linear_names.extend(
                self._analysis_method.get_linear_layers_for_decoder(model, decoder_name)
            )

        for sub_name, sub_module in model.named_modules(prefix=request.name):
            if sub_name not in self._target_linear_names:
                continue

            bound_hook = functools.partial(
                hook_fn,
                layer_name=sub_name,
                stats_dict=self._float_layer_stats,
            )
            handle = sub_module.register_forward_hook(bound_hook)
            self._hook_handles[sub_name] = handle

        super().process(request)
        self._remove_hooks_for_request(request)

    def process(self, request: BatchProcessRequest) -> None:
        for processor in self.quant_processors:
            processor.preprocess(request)
            processor.process(request)
            processor.postprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        hook_fn = self._analysis_method.get_hook()
        model = request.module

        for sub_name, sub_module in model.named_modules(prefix=request.name):
            if sub_name not in self._target_linear_names:
                continue

            bound_hook = functools.partial(
                hook_fn,
                layer_name=sub_name,
                stats_dict=self._quant_layer_stats,
            )
            handle = sub_module.register_forward_hook(bound_hook)
            self._hook_handles[sub_name] = handle

        super().process(request)
        self._remove_hooks_for_request(request)

        decoder_scores = self._analysis_method.compute_decoder_layer_scores(
            self._float_layer_stats,
            self._quant_layer_stats,
            self._target_decoder_layers,
        )
        self._decoder_layer_scores.extend(decoder_scores)

        for decoder_name in self._target_decoder_layers:
            for layer_name in list(self._float_layer_stats.keys()):
                if layer_name.startswith(decoder_name + ".") or layer_name == decoder_name:
                    self._float_layer_stats.pop(layer_name, None)
                    self._quant_layer_stats.pop(layer_name, None)

    def post_run(self) -> None:
        for processor in self.quant_processors:
            processor.post_run()

        ctx = get_current_context()
        ctx["layer_analysis"].debug["layer_scores"] = self._decoder_layer_scores
        ctx["layer_analysis"].debug["method"] = self._analysis_method.name
        ctx["layer_analysis"].debug["patterns"] = self.config.patterns

        logger.info(
            "LayerAnalysisProcessor post_run: %d decoder-layer scores computed (%s)",
            len(self._decoder_layer_scores),
            self._analysis_method.name,
        )

        if not ctx["layer_analysis"].debug["layer_scores"] or len(ctx["layer_analysis"].debug["layer_scores"]) == 0:
            get_logger().warning(
                f"No statistics collected. This may be caused by empty calibration data "
                f"or incompatible patterns with the model structure."
                )

    def _remove_hooks_for_request(self, request: BatchProcessRequest) -> None:
        keys_to_remove = [k for k in self._hook_handles if k == request.name or k.startswith(request.name + ".")]
        for k in keys_to_remove:
            handle = self._hook_handles.pop(k, None)
            if handle is not None:
                handle.remove()
