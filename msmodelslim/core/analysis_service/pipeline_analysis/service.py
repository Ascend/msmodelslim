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

from typing import Any, List, Optional

import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.context import ContextManager
from msmodelslim.core.context.interface import IContextFactory
from msmodelslim.processor.analysis.distributed_utils import read_layer_analysis_result
from msmodelslim.utils.exception import InvalidDatasetError, MisbehaviorError
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
        vlm_dataset_loader: Optional[DatasetLoaderInfra] = None,
    ):
        self.dataset_loader = dataset_loader
        self.vlm_dataset_loader = vlm_dataset_loader
        self.context_factory = context_factory
        self.pipeline_loader = pipeline_loader

    def _iter_dataset_loaders(self) -> List[DatasetLoaderInfra]:
        """Candidate loaders in try order; not bound to a specific model adapter."""
        loaders: List[DatasetLoaderInfra] = []
        if self.dataset_loader is not None:
            loaders.append(self.dataset_loader)
        if self.vlm_dataset_loader is not None:
            loaders.append(self.vlm_dataset_loader)
        return loaders

    def _load_calib_dataset(self, calib_dataset: str) -> Any:
        """
        Try each registered dataset loader until one loads successfully.

        Loaders are format routers (text jsonl / multimodal dir / ...), not tied to
        model adapter type. First non-None result wins.

        Only expected load failures (MisbehaviorError: InvalidDataset /
        SchemaValidate / SecurityError from path checks, etc.) fall through to
        the next loader; programming errors propagate. If every loader fails,
        raise InvalidDatasetError so the CLI exits non-zero.
        """
        errors: List[str] = []
        for loader in self._iter_dataset_loaders():
            loader_name = loader.__class__.__name__
            try:
                calib_data = loader.get_dataset_by_name(calib_dataset)
            except MisbehaviorError as exc:
                get_logger().info(
                    "Dataset loader %s could not load %s: %s",
                    loader_name,
                    calib_dataset,
                    exc,
                )
                errors.append(f"{loader_name}: {exc}")
                continue
            if calib_data is None:
                get_logger().info(
                    "Dataset loader %s returned None for %s, trying next loader",
                    loader_name,
                    calib_dataset,
                )
                errors.append(f"{loader_name}: returned None")
                continue
            get_logger().info(
                "Loaded calibration dataset %s with %s (%d samples)",
                calib_dataset,
                loader_name,
                len(calib_data) if hasattr(calib_data, "__len__") else -1,
            )
            return calib_data

        detail = "; ".join(errors) if errors else "no dataset loaders registered"
        raise InvalidDatasetError(
            f"Failed to load calibration dataset {calib_dataset}: {detail}",
            action="Please check --calib_dataset path/format, or register a compatible loader.",
        )

    def analyze(
        self,
        model_adapter: PipelineInterface,
        analysis_config: AnalysisConfig,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
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
        elif analysis_config.scope == AnalysisScope.ATTN_HEAD:
            get_logger().info("attn_head: attention head analysis (ra_compress)")
        else:
            get_logger().info("attn: all attention modules")

        if device is DeviceType.NPU:
            torch.npu.set_compile_mode(jit_compile=False)

        calib_data = self._load_calib_dataset(analysis_config.calib_dataset)

        use_dp = device_indices is not None and len(device_indices) > 1
        if use_dp:
            from msmodelslim.core.runner.dp_layer_wise_runner import DPLayerWiseRunner

            get_logger().info(
                "Using DPLayerWiseRunner for multi-device analysis on devices %s",
                device_indices,
            )
            runner = DPLayerWiseRunner(adapter=model_adapter)
        else:
            runner = LayerWiseRunner(adapter=model_adapter)

        ctx = self.context_factory.create(is_distributed=use_dp)

        with ContextManager(ctx=ctx):
            builder = self.pipeline_loader.get_pipeline_builder(analysis_config.metrics)
            processor_configs = builder.template_modules(analysis_config.template_substitute_list()).create()
            for cfg in processor_configs:
                runner.add_processor(cfg)

            runner.run(calib_data=calib_data, device=device, device_indices=device_indices)

        # Prefer shared state (DP parent ↔ child); debug is process-local only.
        published = read_layer_analysis_result(ctx)
        layer_scores = published["layer_scores"]
        method = published["method"]
        if published.get("quant_modules") is not None:
            result_patterns = list(published["quant_modules"])
        elif published.get("patterns") is not None:
            result_patterns = list(published["patterns"])
        else:
            result_patterns = analysis_config.template_substitute_list()

        result = AnalysisResult(
            layer_scores=layer_scores,
            method=method,
            patterns=result_patterns,
        )

        return result
