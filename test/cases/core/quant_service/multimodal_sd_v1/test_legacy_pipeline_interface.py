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

import inspect

import pytest

from msmodelslim.core.quant_service.multimodal_sd_v1.legacy_pipeline_interface import (
    LegacyMultimodalPipelineInterface,
)
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.utils.exception import ToDoError


def test_legacy_pipeline_interface_is_abstract_and_extends_runner_interface():
    assert issubclass(LegacyMultimodalPipelineInterface, PipelineInterface)
    assert inspect.isabstract(LegacyMultimodalPipelineInterface)
    assert LegacyMultimodalPipelineInterface.__abstractmethods__ == {
        "apply_quantization",
        "enable_kv_cache",
        "generate_model_forward",
        "generate_model_visit",
        "handle_dataset",
        "init_model",
        "load_pipeline",
        "run_calib_inference",
        "set_model_args",
        "model_path",
        "model_type",
        "trust_remote_code",
    }


def test_unimplemented_legacy_methods_raise_todo_error_with_action():
    cases = [
        (LegacyMultimodalPipelineInterface.run_calib_inference, (object(),), "run_calib_inference"),
        (LegacyMultimodalPipelineInterface.apply_quantization, (object(), object()), "apply_quantization"),
        (LegacyMultimodalPipelineInterface.load_pipeline, (object(),), "load_pipeline"),
        (LegacyMultimodalPipelineInterface.set_model_args, (object(), object()), "set_model_args"),
    ]

    for method, args, name in cases:
        with pytest.raises(ToDoError, match=f"This model does not support {name}") as exc_info:
            method(*args)
        assert exc_info.value.action == f"Please implement {name} for your model."
