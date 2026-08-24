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

import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.context import ContextManager
from msmodelslim.core.context.interface import IContextFactory
from msmodelslim.utils.logging import logger_setter, get_logger

from .pipeline_loader_infra import AnalysisPipelineLoaderInfra
from ..interface import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisScope,
    IAnalysisService,
)


@logger_setter()
class PipelineAnalysisService(IAnalysisService):
    """Analysis service for layer sensitivity evaluation using various methods"""

    def __init__(
        self,
        dataset_loader: DatasetLoaderInfra,
        context_factory: IContextFactory,
        pipeline_loader: AnalysisPipelineLoaderInfra,
    ):
        self.dataset_loader = dataset_loader
        self.context_factory = context_factory
        self.pipeline_loader = pipeline_loader

    def analyze(
        self,
        model_adapter: PipelineInterface,
        analysis_config: AnalysisConfig,
        device: DeviceType = DeviceType.NPU,
    ):
        """
        Analyze layer sensitivity based on configuration.
        """
        get_logger().info("==========ANALYSIS: Starting Layer Analysis==========")
        get_logger().info("Analysis scope: %s", analysis_config.scope.value)
        get_logger().info("Analysis metrics: %s", analysis_config.metrics)
        if analysis_config.scope == AnalysisScope.LINEAR:
            get_logger().info("linear_pattern: %s", analysis_config.linear_pattern)
        elif analysis_config.scope == AnalysisScope.LAYER:
            get_logger().info("quant_modules: %s", analysis_config.quant_modules)
        else:
            get_logger().info("attn: all attention modules")

        if device is DeviceType.NPU:
            torch.npu.set_compile_mode(jit_compile=False)

        calib_data = self.dataset_loader.get_dataset_by_name(analysis_config.calib_dataset)
        if calib_data is None:
            get_logger().warning("No calibration dataset specified. Analysis aborted.")
            return None

        runner = LayerWiseRunner(adapter=model_adapter)
        ctx = self.context_factory.create()

        with ContextManager(ctx=ctx):
            builder = self.pipeline_loader.get_pipeline_builder(analysis_config.metrics)
            processor_configs = builder.template_modules(analysis_config.template_substitute_list()).create()
            for cfg in processor_configs:
                runner.add_processor(cfg)

            runner.run(calib_data=calib_data, device=device)

        # Get layer scores from context
        layer_scores = ctx['layer_analysis'].debug['layer_scores']
        method = ctx['layer_analysis'].debug['method']
        dbg = ctx['layer_analysis'].debug
        if 'quant_modules' in dbg:
            result_patterns = list(dbg['quant_modules'])
        elif 'patterns' in dbg:
            result_patterns = list(dbg['patterns'])
        else:
            result_patterns = analysis_config.template_substitute_list()

        # Create result
        result = AnalysisResult(layer_scores=layer_scores, method=method, patterns=result_patterns)

        return result
