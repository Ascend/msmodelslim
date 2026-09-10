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

import os
from pathlib import Path

from msmodelslim.app.eval import InferenceApplication
from msmodelslim.cli.utils import parse_device_string
from msmodelslim.core.context import ContextFactory
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VLMDatasetLoader
from msmodelslim.infra.file_dataset_loader import FileDatasetLoader
from msmodelslim.infra.logging_inference_result_displayer import LoggingInferenceResultDisplayer
from msmodelslim.model import PluginModelFactory
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security.path import get_valid_read_path


def get_dataset_dir() -> Path:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_calib_dir = os.path.abspath(os.path.join(cur_dir, "../../lab_calib"))
    return Path(get_valid_read_path(lab_calib_dir, is_dir=True))


def main(args):
    try:
        device_type, device_indices = parse_device_string(args.device)
        if getattr(args, "device_id", None):
            device_indices = list(args.device_id)
        dataset_loader = FileDatasetLoader(get_dataset_dir())
        vlm_dataset_loader = VLMDatasetLoader(get_dataset_dir())
        app = InferenceApplication(
            model_factory=PluginModelFactory(),
            dataset_loader=dataset_loader,
            context_factory=ContextFactory(enable_debug=False),
            result_displayer=LoggingInferenceResultDisplayer(),
            vlm_dataset_loader=vlm_dataset_loader,
        )
        return app.run(
            model_type=args.model_type,
            model_path=args.model_path,
            device=device_type,
            device_indices=device_indices,
            prompt_file=args.prompt_file,
            max_new_tokens=getattr(args, "max_new_tokens", 1),
        )
    except Exception as exc:
        get_logger().error("Evaluation failed: %s", exc)
        raise
