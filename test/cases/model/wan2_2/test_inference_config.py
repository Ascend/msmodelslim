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

# pylint: disable=no-name-in-module

from unittest.mock import Mock

import pytest

from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import (
    DumpConfig,
    MultimodalSDConfig,
    validate_inference_config,
)
from msmodelslim.model.wan2_2.t2v.model_adapter import Wan2_2T2VModelAdapter
from msmodelslim.model.wan2_2.i2v.model_adapter import Wan2_2I2VModelAdapter
from msmodelslim.model.wan2_2.ti2v.model_adapter import Wan2_2TI2VModelAdapter
from msmodelslim.utils.exception import SchemaValidateError


def _mock_adapter(config_cls, model_type: str):
    adapter = Mock()
    adapter.model_type = model_type
    adapter.get_inference_config_class.return_value = config_cls
    return adapter


def _sd_config(inference_config: dict) -> MultimodalSDConfig:
    return MultimodalSDConfig(dump_config=DumpConfig(), inference_config=inference_config)


class TestValidateInferenceConfigWan22T2V:
    @staticmethod
    def test_validate_inference_config_returns_t2v_config_when_valid():
        adapter = _mock_adapter(Wan2_2T2VModelAdapter.Wan2_2T2VInferenceConfig, "Wan2.2-T2V-A14B")
        cfg = validate_inference_config(adapter, _sd_config({"sample_steps": 20}))
        assert cfg.sample_steps == 20

    @staticmethod
    def test_validate_inference_config_raises_schema_error_when_task_mismatched():
        adapter = _mock_adapter(Wan2_2T2VModelAdapter.Wan2_2T2VInferenceConfig, "Wan2.2-T2V-A14B")
        with pytest.raises(SchemaValidateError):
            validate_inference_config(adapter, _sd_config({"task": "i2v-A14B"}))

    @staticmethod
    def test_validate_inference_config_raises_schema_error_when_unknown_field():
        adapter = _mock_adapter(Wan2_2T2VModelAdapter.Wan2_2T2VInferenceConfig, "Wan2.2-T2V-A14B")
        with pytest.raises(SchemaValidateError):
            validate_inference_config(adapter, _sd_config({"use_attentioncache": True}))

    @staticmethod
    def test_validate_inference_config_raises_schema_error_when_guide_scale_is_tuple():
        adapter = _mock_adapter(Wan2_2T2VModelAdapter.Wan2_2T2VInferenceConfig, "Wan2.2-T2V-A14B")
        with pytest.raises(SchemaValidateError):
            validate_inference_config(adapter, _sd_config({"sample_guide_scale": [3.0, 4.0]}))


class TestValidateInferenceConfigWan22I2V:
    @staticmethod
    def test_validate_inference_config_raises_schema_error_when_task_mismatched():
        adapter = _mock_adapter(Wan2_2I2VModelAdapter.Wan2_2I2VInferenceConfig, "Wan2.2-I2V-A14B")
        with pytest.raises(SchemaValidateError):
            validate_inference_config(adapter, _sd_config({"task": "t2v-A14B"}))


class TestValidateInferenceConfigWan22TI2V:
    @staticmethod
    def test_validate_inference_config_returns_ti2v_config_when_valid():
        adapter = _mock_adapter(Wan2_2TI2VModelAdapter.Wan2_2TI2VInferenceConfig, "Wan2.2-TI2V-5B")
        cfg = validate_inference_config(adapter, _sd_config({"sample_steps": 50}))
        assert cfg.sample_steps == 50

    @staticmethod
    def test_validate_inference_config_raises_schema_error_when_task_mismatched():
        adapter = _mock_adapter(Wan2_2TI2VModelAdapter.Wan2_2TI2VInferenceConfig, "Wan2.2-TI2V-5B")
        with pytest.raises(SchemaValidateError):
            validate_inference_config(adapter, _sd_config({"task": "t2v-A14B"}))
