#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

# pylint: disable=no-name-in-module

import pytest

from msmodelslim.core.quant_service.multimodal_sd_v1.legacy_pipeline_interface import (
    LegacyMultimodalPipelineInterface,
)
from msmodelslim.utils.exception import ToDoError


class LegacyFullImplementedBase(LegacyMultimodalPipelineInterface):
    model_path = "mock_path"
    model_type = "mock_type"
    trust_remote_code = False

    def handle_dataset(self, dataset, device=None):
        return []

    def init_model(self, device=None):
        return {}

    def generate_model_visit(self, model):
        yield object()

    def generate_model_forward(self, model, inputs):
        yield object()

    def enable_kv_cache(self, model, need_kv_cache):
        pass

    def run_calib_inference(self):
        pass

    def apply_quantization(self, quant_model_func):
        quant_model_func()

    def load_pipeline(self):
        pass

    def set_model_args(self, override_model_config):
        pass


class TestLegacyMultimodalPipelineExceptions:
    def test_run_calib_inference_throws_todo_error(self):
        class TestSubclass(LegacyFullImplementedBase):
            def run_calib_inference(self):
                super(LegacyFullImplementedBase, self).run_calib_inference()

        with pytest.raises(ToDoError) as exc_info:
            TestSubclass().run_calib_inference()
        assert "run_calib_inference" in str(exc_info.value)
