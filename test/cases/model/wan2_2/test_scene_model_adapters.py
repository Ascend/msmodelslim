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

场景化 Wan2.2 适配器（T2V / I2V / TI2V）单测。

InferenceConfig 与 validate_inference_config 的 schema 用例见 test_inference_config.py；
get_expert_adapter 未绑定/回退用例见 test_get_expert_adapter.py。
"""

# pylint: disable=no-name-in-module

import argparse
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.model.wan2_2.constants import TASK_TYPES
from msmodelslim.model.wan2_2.expert_sub_adapter import (
    Wan2_2ExpertSubAdapter,
    Wan2_2HighNoiseSubAdapter,
    Wan2_2LowNoiseSubAdapter,
)
from msmodelslim.model.wan2_2.i2v.model_adapter import Wan2_2I2VModelAdapter
from msmodelslim.model.wan2_2.t2v.model_adapter import Wan2_2T2VModelAdapter
from msmodelslim.model.wan2_2.ti2v.model_adapter import Wan2_2TI2VModelAdapter
from msmodelslim.utils.exception import SchemaValidateError


def _scene_adapter(cls, model_path):
    adapter = cls.__new__(cls)
    adapter.model_path = model_path
    adapter._expert_adapters = {}
    adapter.low_noise_model = None
    adapter.high_noise_model = None
    adapter.transformer = None
    return adapter


class _YamlInferenceConfig:
    def __init__(self, fields: dict):
        self._fields = fields

    def model_dump(self, exclude_none=True):
        _ = exclude_none
        return dict(self._fields)


def _patch_wan_configs_for_configure_runtime(scene_task: str):
    mock_wan_configs = MagicMock()
    mock_wan_configs.WAN_CONFIGS = {scene_task: MagicMock(param_dtype="bfloat16")}
    return patch.dict(sys.modules, {"wan": MagicMock(configs=mock_wan_configs), "wan.configs": mock_wan_configs})


class TestWan2_2T2VValidateCalibSamples:
    @pytest.fixture
    def adapter(self, tmp_path):
        return _scene_adapter(Wan2_2T2VModelAdapter, tmp_path)

    def test_validate_calib_samples_returns_samples_when_text_only_valid(self, adapter):
        samples = [VlmCalibSample(text="hello")]
        assert adapter.validate_calib_samples(samples) == samples

    def test_validate_calib_samples_raises_schema_validate_error_when_image_present(self, adapter):
        samples = [VlmCalibSample(text="hello", image="/path/img.jpg")]
        with pytest.raises(SchemaValidateError, match="must not include image"):
            adapter.validate_calib_samples(samples)

    def test_validate_calib_samples_raises_schema_validate_error_when_text_empty(self, adapter):
        with pytest.raises(SchemaValidateError, match="non-empty text"):
            adapter.validate_calib_samples([VlmCalibSample(text="   ")])


class TestWan2_2I2VValidateCalibSamples:
    @pytest.fixture
    def adapter(self, tmp_path):
        return _scene_adapter(Wan2_2I2VModelAdapter, tmp_path)

    def test_validate_calib_samples_returns_samples_when_text_and_image_valid(self, adapter):
        samples = [VlmCalibSample(text="prompt", image="/data/frame.jpg")]
        assert adapter.validate_calib_samples(samples) == samples

    def test_validate_calib_samples_raises_schema_validate_error_when_image_missing(self, adapter):
        with pytest.raises(SchemaValidateError, match="requires image"):
            adapter.validate_calib_samples([VlmCalibSample(text="prompt")])

    def test_validate_calib_samples_raises_schema_validate_error_when_image_blank(self, adapter):
        with pytest.raises(SchemaValidateError, match="requires image"):
            adapter.validate_calib_samples([VlmCalibSample(text="prompt", image="  ")])


class TestWan2_2TI2VValidateCalibSamples:
    @pytest.fixture
    def adapter(self, tmp_path):
        return _scene_adapter(Wan2_2TI2VModelAdapter, tmp_path)

    def test_validate_calib_samples_accepts_text_only_for_t2v_style_calibration(self, adapter):
        samples = [VlmCalibSample(text="text only")]
        assert adapter.validate_calib_samples(samples) == samples

    def test_validate_calib_samples_accepts_text_with_image_for_i2v_style_calibration(self, adapter):
        samples = [VlmCalibSample(text="prompt", image="/img.png")]
        assert adapter.validate_calib_samples(samples) == samples

    def test_validate_calib_samples_raises_schema_validate_error_when_image_set_but_blank(self, adapter):
        with pytest.raises(SchemaValidateError, match="non-empty path"):
            adapter.validate_calib_samples([VlmCalibSample(text="prompt", image="")])


class TestWan2_2ConfigureRuntime:
    @pytest.fixture
    def t2v_adapter(self, tmp_path):
        return _scene_adapter(Wan2_2T2VModelAdapter, tmp_path)

    def test_configure_runtime_raises_schema_validate_error_when_illegal_yaml_key(self, t2v_adapter):
        allowed = frozenset({"sample_steps", "size"})
        with (
            _patch_wan_configs_for_configure_runtime("t2v-A14B"),
            patch.object(t2v_adapter, "_allowed_generate_config_keys", return_value=allowed),
        ):
            with pytest.raises(SchemaValidateError, match="illegal config attributes"):
                t2v_adapter.configure_runtime(
                    _YamlInferenceConfig({"use_attentioncache": True, "sample_steps": 40}),
                )

    def test_configure_runtime_writes_sample_steps_to_model_args_when_yaml_override(self, t2v_adapter):
        merged = argparse.Namespace(
            sample_steps=55,
            task="t2v-A14B",
            ckpt_dir=str(t2v_adapter.model_path),
        )
        allowed = frozenset({"sample_steps"})
        with (
            _patch_wan_configs_for_configure_runtime("t2v-A14B"),
            patch.object(t2v_adapter, "_allowed_generate_config_keys", return_value=allowed),
            patch.object(t2v_adapter, "_parse_args_from_generate", return_value=merged) as mock_parse,
        ):
            t2v_adapter.configure_runtime(_YamlInferenceConfig({"sample_steps": 55}))
            cli_args = mock_parse.call_args[0][0]
            assert "--sample_steps" in cli_args
            assert "55" in cli_args
        assert t2v_adapter.model_args.sample_steps == 55
        assert t2v_adapter.model_args.task_config == TASK_TYPES["t2v-A14B"]
        assert not hasattr(t2v_adapter, "inference_config")

    def test_configure_runtime_forces_scene_task_on_cli_argv_when_yaml_task_differs(self, tmp_path):
        """scene_task 由适配器固定，CLI 末尾 --task 覆盖 YAML 中的错误 task。"""
        adapter = _scene_adapter(Wan2_2I2VModelAdapter, tmp_path)
        merged = argparse.Namespace(sample_steps=40, task="i2v-A14B")
        allowed = frozenset({"sample_steps", "task"})
        with (
            _patch_wan_configs_for_configure_runtime("i2v-A14B"),
            patch.object(adapter, "_allowed_generate_config_keys", return_value=allowed),
            patch.object(adapter, "_parse_args_from_generate", return_value=merged) as mock_parse,
        ):
            adapter.configure_runtime(
                _YamlInferenceConfig({"sample_steps": 40, "task": "t2v-A14B"}),
            )
            cli_args = mock_parse.call_args[0][0]
            task_positions = [i for i, arg in enumerate(cli_args) if arg == "--task"]
            assert task_positions, "expected --task on generate CLI argv"
            last_task = task_positions[-1]
            assert cli_args[last_task + 1] == "i2v-A14B"
        assert adapter.model_args.task_config == TASK_TYPES["i2v-A14B"]


class TestWan2_2InitModelExpertBinding:
    def test_init_model_binds_dual_expert_sub_adapters_when_t2v_pipeline_loaded(self, tmp_path):
        adapter = _scene_adapter(Wan2_2T2VModelAdapter, tmp_path)
        low = Mock(spec=nn.Module)
        high = Mock(spec=nn.Module)

        def _fake_load():
            adapter.low_noise_model = low
            adapter.high_noise_model = high

        with patch.object(adapter, "_load_pipeline", side_effect=_fake_load):
            experts = adapter.init_model(DeviceType.NPU)

        assert experts == {"low_noise_model": low, "high_noise_model": high}
        assert isinstance(adapter.get_expert_adapter("low_noise_model"), Wan2_2LowNoiseSubAdapter)
        assert isinstance(adapter.get_expert_adapter("high_noise_model"), Wan2_2HighNoiseSubAdapter)

    def test_init_model_binds_dual_expert_sub_adapters_when_i2v_pipeline_loaded(self, tmp_path):
        adapter = _scene_adapter(Wan2_2I2VModelAdapter, tmp_path)
        low = Mock(spec=nn.Module)
        high = Mock(spec=nn.Module)

        def _fake_load():
            adapter.low_noise_model = low
            adapter.high_noise_model = high

        with patch.object(adapter, "_load_pipeline", side_effect=_fake_load):
            experts = adapter.init_model()

        assert set(experts.keys()) == {"low_noise_model", "high_noise_model"}
        assert adapter.get_expert_adapter("low_noise_model").expert_name == "low_noise_model"

    def test_init_model_binds_single_expert_sub_adapter_when_ti2v_pipeline_loaded(self, tmp_path):
        adapter = _scene_adapter(Wan2_2TI2VModelAdapter, tmp_path)
        dit = Mock(spec=nn.Module)

        def _fake_load():
            adapter.transformer = dit

        with patch.object(adapter, "_load_pipeline", side_effect=_fake_load):
            experts = adapter.init_model()

        assert experts == {"": dit}
        sub = adapter.get_expert_adapter("")
        assert isinstance(sub, Wan2_2ExpertSubAdapter)
        assert sub is not adapter


class TestWan2_2RuntimeValue:
    def test_runtime_value_reads_from_inference_config_model_when_pydantic(self, tmp_path):
        adapter = _scene_adapter(Wan2_2T2VModelAdapter, tmp_path)
        adapter.model_args = argparse.Namespace(sample_steps=10)
        cfg = Wan2_2T2VModelAdapter.Wan2_2T2VInferenceConfig(sample_steps=55)
        assert adapter._runtime_value(cfg, "sample_steps") == 55

    def test_runtime_value_reads_from_dict_when_inference_config_is_mapping(self, tmp_path):
        adapter = _scene_adapter(Wan2_2T2VModelAdapter, tmp_path)
        adapter.model_args = argparse.Namespace(sample_steps=10)
        assert adapter._runtime_value({"sample_steps": 55}, "sample_steps") == 55

    def test_runtime_value_falls_back_to_model_args_when_inference_config_is_none(self, tmp_path):
        adapter = _scene_adapter(Wan2_2T2VModelAdapter, tmp_path)
        adapter.model_args = argparse.Namespace(sample_steps=10)
        assert adapter._runtime_value(None, "sample_steps") == 10
