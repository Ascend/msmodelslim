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

import os
from pathlib import Path

from msmodelslim.app.analysis import LayerAnalysisApplication
from msmodelslim.app.analysis.application import (
    AnalysisMetrics,
    AttnArgs,
    AttnHeadArgs,
    LayerArgs,
    LinearArgs,
)
from msmodelslim.core.analysis_service import PipelineAnalysisService
from msmodelslim.core.context import ContextFactory
from msmodelslim.infra.file_dataset_loader import FileDatasetLoader
from msmodelslim.infra.analysis_pipeline_loader import YamlAnalysisPipelineLoader
from msmodelslim.infra.logging_analysis_result_displayer import AnalysisResultDisplayerFactory
from msmodelslim.model import PluginModelFactory
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security.path import get_valid_read_path


def get_dataset_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_calib_dir = os.path.abspath(os.path.join(cur_dir, '../../lab_calib'))
    lab_calib_dir = get_valid_read_path(lab_calib_dir, is_dir=True)
    return Path(lab_calib_dir)


def main(args):
    """Main function for layer analysis CLI"""
    try:
        # Get dataset directory
        dataset_dir = get_dataset_dir()
        # Create dataset loader
        dataset_loader = FileDatasetLoader(dataset_dir)

        # Create pipeline loader
        pipeline_loader = YamlAnalysisPipelineLoader()

        # Create analysis service
        analysis_service = PipelineAnalysisService(
            dataset_loader, context_factory=ContextFactory(enable_debug=True), pipeline_loader=pipeline_loader
        )
        # Create model factory
        model_factory = PluginModelFactory()

        metrics = AnalysisMetrics(str(args.metrics).strip().lower())
        save_path = args.save_path

        if args.scope == 'linear':
            scope_args = LinearArgs(pattern=list(args.pattern), metrics=metrics)
        elif args.scope == 'layer':
            scope_args = LayerArgs(quant_modules=list(args.quant_modules), metrics=metrics)
        elif args.scope == 'attn':
            scope_args = AttnArgs(metrics=metrics)
        elif args.scope == 'attn_head':
            scope_args = AttnHeadArgs(metrics=metrics)
        else:
            raise ValueError(f"Unsupported analyze scope: {args.scope}")

        # Create result manager (按 metrics 选择对应展示器)
        result_manager = AnalysisResultDisplayerFactory.create(metrics.value)

        # Create analysis app
        analysis_app = LayerAnalysisApplication(
            analysis_service=analysis_service,
            model_factory=model_factory,
            result_manager=result_manager,
        )

        # topk 仅对 linear/layer/attn 子命令可用，attn_head 无此参数
        topk = getattr(args, 'topk', 15)

        # Run analysis
        result = analysis_app.analyze(
            model_type=args.model_type,
            model_path=args.model_path,
            scope_args=scope_args,
            device=args.device,
            calib_dataset=args.calib_dataset,
            topk=topk,
            trust_remote_code=args.trust_remote_code,
            save_path=save_path,
        )
        return result

    except Exception as e:
        get_logger().error("Layer analysis failed: %s", str(e))
        raise
