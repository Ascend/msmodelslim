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

# pylint: disable=no-member,abstract-class-instantiated

import pytest
from contextlib import nullcontext
from pydantic import BaseModel

from msmodelslim.core.quant_service.multimodal_sd_v1.pipeline_interface import MultimodalPipelineInterface
from msmodelslim.utils.exception import ToDoError


class _DummyInferenceConfig(BaseModel):
    size: str = "720p"


class FullImplementedBase(MultimodalPipelineInterface):
    """完全实现所有抽象方法的基础类，用于测试"""

    model_path = "mock_path"
    model_type = "mock_type"
    trust_remote_code = False

    def handle_dataset(self, dataset, device=None):
        return []

    def init_model(self, device=None):
        return None

    def generate_model_visit(self, model):
        yield object()

    def generate_model_forward(self, model, inputs):
        yield object()

    def enable_kv_cache(self, model, need_kv_cache):
        pass

    def get_inference_config_class(self):
        return _DummyInferenceConfig

    def inference_dump_calib_data(self, dataset=None, inference_config=None):
        pass

    def prepare_calib_data(self, models, dump_config, save_path, dataset, inference_config):
        return {k: None for k in models}

    def quantization_context(self):
        return nullcontext()

    def configure_runtime(self, inference_config=None):
        pass


class TestMultimodalPipelineExceptions:
    """测试MultimodalPipelineInterface的异常抛出"""

    def test_inference_dump_calib_data_throws_todo_error(self):
        class TestSubclass(FullImplementedBase):
            def inference_dump_calib_data(self, dataset=None, inference_config=None):
                super(FullImplementedBase, self).inference_dump_calib_data(dataset, inference_config)

        instance = TestSubclass()
        with pytest.raises(ToDoError) as exc_info:
            instance.inference_dump_calib_data()
        assert "This model does not support inference_dump_calib_data." in str(exc_info.value)

    def test_quantization_context_throws_todo_error(self):
        class TestSubclass(FullImplementedBase):
            def quantization_context(self):
                super(FullImplementedBase, self).quantization_context()

        instance = TestSubclass()
        with pytest.raises(ToDoError) as exc_info:
            instance.quantization_context()
        assert "This model does not support quantization_context." in str(exc_info.value)

    def test_prepare_calib_data_throws_todo_error(self):
        class TestSubclass(FullImplementedBase):
            def prepare_calib_data(self, models, dump_config, save_path, dataset, inference_config):
                super(FullImplementedBase, self).prepare_calib_data(
                    models, dump_config, save_path, dataset, inference_config
                )

        instance = TestSubclass()
        with pytest.raises(ToDoError) as exc_info:
            instance.prepare_calib_data({}, None, None, [], None)
        assert "This model does not support prepare_calib_data." in str(exc_info.value)

    def test_get_inference_config_class_throws_todo_error(self):
        class TestSubclass(FullImplementedBase):
            def get_inference_config_class(self):
                super(FullImplementedBase, self).get_inference_config_class()

        instance = TestSubclass()
        with pytest.raises(ToDoError) as exc_info:
            instance.get_inference_config_class()
        assert "This model does not support get_inference_config_class." in str(exc_info.value)

    def test_configure_runtime_throws_todo_error(self):
        class TestSubclass(FullImplementedBase):
            def configure_runtime(self, inference_config=None):
                super(FullImplementedBase, self).configure_runtime(inference_config)

        instance = TestSubclass()
        with pytest.raises(ToDoError) as exc_info:
            instance.configure_runtime({})
        assert "This model does not support configure_runtime." in str(exc_info.value)
