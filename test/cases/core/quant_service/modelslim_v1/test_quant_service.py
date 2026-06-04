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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.core.quant_service.modelslim_v1.quant_service import (
    ModelslimV1QuantService,
    ModelslimV1QuantServiceConfig,
    get_plugin,
)
from msmodelslim.core.quant_service.modelslim_v1.save import AscendV1Config, MindIEFormatConfig
from msmodelslim.format.base import QuantFormatConfig
from msmodelslim.processor.save.processor import QuantSaveProcessorConfig


class TestModelslimV1QuantService:
    """Tests for ModelslimV1QuantService._build_save_processor_config."""

    @pytest.fixture
    def service(self):
        return ModelslimV1QuantService(
            quant_service_config=MagicMock(),
            dataset_loader=MagicMock(),
            context_factory=MagicMock(),
        )

    def _format_mock(self, fmt_type: str):
        fmt = MagicMock()
        fmt.type = fmt_type
        fmt.model_dump.return_value = {"type": fmt_type, "part_file_size": 4}
        return fmt

    def test_build_save_processor_config_returns_ascendv1_when_type_ascendv1_saver(self, service):
        """场景：format type 为 ascendv1_saver。预期：返回 AscendV1Config 并设置 save_directory。"""
        save_path = Path("/tmp/save")
        proc = service._build_save_processor_config(self._format_mock("ascendv1_saver"), save_path)
        assert isinstance(proc, AscendV1Config)
        assert proc.save_directory == str(save_path)

    def test_build_save_processor_config_returns_mindie_when_type_mindie_format_saver(self, service):
        """场景：format type 为 mindie_format_saver。预期：返回 MindIEFormatConfig。"""
        out = Path("/out")
        proc = service._build_save_processor_config(self._format_mock("mindie_format_saver"), out)
        assert isinstance(proc, MindIEFormatConfig)
        assert Path(proc.save_directory) == out

    def test_build_save_processor_config_returns_quant_save_when_unknown_type(self, service):
        """场景：未知 format type。预期：回退为 QuantSaveProcessorConfig。"""
        fmt = QuantFormatConfig()
        fallback = Path("/fallback")
        proc = service._build_save_processor_config(fmt, fallback)
        assert isinstance(proc, QuantSaveProcessorConfig)
        assert proc.format is fmt
        assert Path(proc.save_directory) == fallback


class TestModelslimV1QuantServiceChooseRunner:
    """Tests for _choose_runner_type."""

    @pytest.fixture
    def service(self):
        return ModelslimV1QuantService(
            quant_service_config=MagicMock(),
            dataset_loader=MagicMock(),
            context_factory=MagicMock(),
        )

    def _spec(self, runner):
        spec = MagicMock()
        spec.runner = runner
        return spec

    def test_choose_runner_returns_model_wise_when_configured(self, service):
        """场景：runner=MODEL_WISE。预期：返回 MODEL_WISE。"""
        from msmodelslim.core.const import RunnerType

        cfg = MagicMock()
        cfg.spec = self._spec(RunnerType.MODEL_WISE)
        assert service._choose_runner_type(cfg, MagicMock()) == RunnerType.MODEL_WISE

    def test_choose_runner_returns_dp_layer_wise_when_auto_multi_device(self, service):
        """场景：AUTO 且多 device_indices。预期：DP_LAYER_WISE。"""
        from msmodelslim.core.const import RunnerType

        cfg = MagicMock()
        cfg.spec = self._spec(RunnerType.AUTO)
        assert service._choose_runner_type(cfg, MagicMock(), device_indices=[0, 1]) == RunnerType.DP_LAYER_WISE


class TestModelslimV1QuantServiceQuantizeValidation:
    """Tests for quantize input validation."""

    @pytest.fixture
    def service(self):
        return ModelslimV1QuantService(
            quant_service_config=MagicMock(),
            dataset_loader=MagicMock(),
            context_factory=MagicMock(),
        )

    def test_quantize_raises_schema_error_when_not_base_quant_config(self, service):
        """场景：quant_config 类型错误。预期：SchemaValidateError。"""
        from msmodelslim.utils.exception import SchemaValidateError

        with pytest.raises(SchemaValidateError, match="NOT BaseQuantConfig"):
            service.quantize(MagicMock(), MagicMock())


class TestModelslimV1QuantServiceQuantProcess:
    """Tests for quant_process with mocked runner and context."""

    @pytest.fixture
    def service(self):
        ctx = MagicMock()
        factory = MagicMock()
        factory.create.return_value = ctx
        svc = ModelslimV1QuantService(
            quant_service_config=MagicMock(),
            dataset_loader=MagicMock(),
            context_factory=factory,
        )
        svc.dataset_loader.get_dataset_by_name.return_value = []
        return svc

    def _quant_config(self):
        from msmodelslim.core.const import RunnerType
        from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig

        spec = MagicMock()
        spec.runner = RunnerType.LAYER_WISE
        spec.prior = []
        spec.process = []
        spec.save = []
        spec.dataset = "ds"
        cfg = MagicMock(spec=ModelslimV1QuantConfig)
        cfg.spec = spec
        return cfg

    @patch("msmodelslim.core.quant_service.modelslim_v1.quant_service.LayerWiseRunner")
    @patch("msmodelslim.core.quant_service.modelslim_v1.quant_service.ContextManager")
    def test_quant_process_runs_layer_wise_runner_when_configured(self, mock_ctx_mgr, mock_runner_cls, service):
        """场景：LAYER_WISE runner。预期：runner.run 被调用。"""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        adapter = MagicMock()
        service.quant_process(self._quant_config(), adapter, None, device=MagicMock())
        mock_runner.run.assert_called_once()
        mock_ctx_mgr.assert_called_once()

    @patch("msmodelslim.core.quant_service.modelslim_v1.quant_service.seed_all")
    @patch("msmodelslim.core.quant_service.modelslim_v1.quant_service.LayerWiseRunner")
    @patch("msmodelslim.core.quant_service.modelslim_v1.quant_service.ContextManager")
    def test_quant_process_clears_safetensors_when_save_path_exists(
        self, mock_ctx_mgr, mock_runner_cls, mock_seed, service, tmp_path
    ):
        """场景：save_path 下已有 safetensors。预期：清理后执行量化。"""
        stale = tmp_path / "old.safetensors"
        stale.write_bytes(b"x")
        mock_runner_cls.return_value = MagicMock()
        service.quant_process(self._quant_config(), MagicMock(), tmp_path, device=MagicMock())
        assert not stale.exists()

    @patch("msmodelslim.core.quant_service.modelslim_v1.quant_service.seed_all")
    @patch("msmodelslim.core.runner.dp_layer_wise_runner.DPLayerWiseRunner")
    @patch("msmodelslim.core.quant_service.modelslim_v1.quant_service.ContextManager")
    def test_quant_process_uses_dp_runner_when_dp_layer_wise(self, mock_ctx_mgr, mock_dp_cls, mock_seed, service):
        """场景：runner=DP_LAYER_WISE。预期：DPLayerWiseRunner 被实例化。"""
        from msmodelslim.core.const import RunnerType

        cfg = self._quant_config()
        cfg.spec.runner = RunnerType.DP_LAYER_WISE
        mock_dp_cls.return_value = MagicMock()
        service.quant_process(cfg, MagicMock(), None, device=MagicMock(), device_indices=[0, 1])
        mock_dp_cls.assert_called_once()


def test_get_plugin_returns_config_and_service_classes():
    """场景：调用 get_plugin。预期：返回配置类与服务类。"""
    cfg_cls, svc_cls = get_plugin()
    assert cfg_cls is ModelslimV1QuantServiceConfig
    assert svc_cls is ModelslimV1QuantService
