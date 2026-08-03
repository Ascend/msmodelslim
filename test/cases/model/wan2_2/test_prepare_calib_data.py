#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Wan2.2 prepare_calib_data 与 enable_dump 配置联动单测。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import DumpConfig
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.model.wan2_2.t2v.model_adapter import Wan2_2T2VModelAdapter


def _t2v_adapter(tmp_path):
    adapter = Wan2_2T2VModelAdapter.__new__(Wan2_2T2VModelAdapter)
    adapter.model_args = MagicMock(task_config="t2v-A14B")
    return adapter


class TestWan2_2PrepareCalibData:
    """Wan2_2BaseModelAdapter.prepare_calib_data"""

    @staticmethod
    def test_prepare_calib_data_returns_none_per_expert_when_enable_dump_false(tmp_path):
        adapter = _t2v_adapter(tmp_path)
        models = {"low_noise_model": MagicMock(spec=nn.Module), "high_noise_model": MagicMock(spec=nn.Module)}
        dump_config = DumpConfig(enable_dump=False)

        with (
            patch(
                "msmodelslim.model.wan2_2.base_model_adapter.load_cached_data_for_models",
            ) as mock_load,
            patch.object(adapter, "release_auxiliary_models") as mock_release,
        ):
            result = adapter.prepare_calib_data(
                models=models,
                dump_config=dump_config,
                save_path=Path(tmp_path),
                dataset=[VlmCalibSample(text="hello")],
                inference_config=None,
            )

        mock_load.assert_not_called()
        mock_release.assert_called_once()
        assert result == {"low_noise_model": None, "high_noise_model": None}

    @staticmethod
    def test_prepare_calib_data_calls_load_cached_when_enable_dump_true(tmp_path):
        adapter = _t2v_adapter(tmp_path)
        models = {"transformer": MagicMock(spec=nn.Module)}
        dump_config = DumpConfig(enable_dump=True, dump_data_dir=str(tmp_path))
        expected = {"transformer": {"tensor": 1}}

        with (
            patch(
                "msmodelslim.model.wan2_2.base_model_adapter.load_cached_data_for_models",
                return_value=expected,
            ) as mock_load,
            patch.object(adapter, "release_auxiliary_models") as mock_release,
        ):
            result = adapter.prepare_calib_data(
                models=models,
                dump_config=dump_config,
                save_path=Path(tmp_path),
                dataset=[VlmCalibSample(text="hello")],
                inference_config=None,
            )

        mock_load.assert_called_once()
        mock_release.assert_called_once()
        assert result == expected


class TestWan2_2ReleaseAuxiliaryModels:
    """Wan2_2BaseModelAdapter.release_auxiliary_models"""

    @staticmethod
    def test_release_clears_text_encoder_and_vae():
        adapter = _t2v_adapter(Path("/tmp"))
        text_encoder = MagicMock()
        text_encoder.model = MagicMock()
        vae = MagicMock()
        vae.model = MagicMock()
        vae.mean = torch.zeros(1)
        vae.std = torch.ones(1)
        pipeline = MagicMock()
        pipeline.text_encoder = text_encoder
        pipeline.vae = vae
        adapter.wan_t2v = pipeline
        adapter.wan_ti2v = None
        adapter.wan_i2v = None
        adapter.pipeline = None

        with (
            patch("msmodelslim.model.wan2_2.base_model_adapter.gc.collect"),
            patch(
                "msmodelslim.model.wan2_2.base_model_adapter.get_device_allocated_memory",
                return_value=0,
            ),
            patch(
                "msmodelslim.model.wan2_2.base_model_adapter.get_device_reserved_memory",
                return_value=0,
            ),
        ):
            adapter.release_auxiliary_models()

        assert pipeline.text_encoder is None
        assert pipeline.vae is None
        text_encoder.model.cpu.assert_called_once()
        vae.model.cpu.assert_called_once()

    @staticmethod
    def test_release_noop_when_no_pipeline():
        adapter = _t2v_adapter(Path("/tmp"))
        adapter.wan_t2v = None
        adapter.wan_ti2v = None
        adapter.wan_i2v = None
        adapter.pipeline = None

        with patch("msmodelslim.model.wan2_2.base_model_adapter.gc.collect") as mock_gc:
            adapter.release_auxiliary_models()

        mock_gc.assert_not_called()
