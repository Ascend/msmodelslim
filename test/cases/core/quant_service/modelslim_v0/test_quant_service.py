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

import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub llm_ptq imports so quant_service loads without transformers/accelerate
_mock_anti_pkg = MagicMock()
_mock_anti_pkg.AntiOutlier = MagicMock()
_mock_anti_pkg.AntiOutlierConfig = MagicMock()
sys.modules["msmodelslim.pytorch.llm_ptq.anti_outlier"] = _mock_anti_pkg

_mock_ptq_tools = MagicMock()
_mock_ptq_tools.Calibrator = MagicMock()
_mock_ptq_tools.QuantConfig = MagicMock()
sys.modules["msmodelslim.pytorch.llm_ptq.llm_ptq_tools"] = _mock_ptq_tools

from msmodelslim.core.const import DeviceType  # noqa: E402
from msmodelslim.core.quant_service.interface import BaseQuantConfig  # noqa: E402
from msmodelslim.core.quant_service.modelslim_v0.quant_config import ModelslimV0QuantConfig  # noqa: E402
from msmodelslim.core.quant_service.modelslim_v0.pipeline_interface import PipelineInterface  # noqa: E402
from msmodelslim.core.quant_service.modelslim_v0.quant_service import (  # noqa: E402
    ModelslimV0QuantService,
    ModelslimV0QuantServiceConfig,
)
from msmodelslim.utils.exception import SchemaValidateError  # noqa: E402


class TestModelslimV0QuantService:
    """Tests for ModelslimV0QuantService."""

    @pytest.fixture
    def service(self):
        return ModelslimV0QuantService(
            quant_service_config=ModelslimV0QuantServiceConfig(),
            dataset_loader=MagicMock(),
            context_factory=MagicMock(),
        )

    def test_backend_name_is_modelslim_v0_when_initialized(self, service):
        """场景：初始化服务。预期：backend_name 为 modelslim_v0。"""
        assert service.backend_name == "modelslim_v0"

    def test_quantize_raises_schema_validate_error_when_quant_config_invalid(self, service):
        """场景：quant_config 非 BaseQuantConfig。预期：SchemaValidateError。"""
        with pytest.raises(SchemaValidateError, match="BaseTask"):
            service.quantize(quant_config=MagicMock(), model_adapter=MagicMock())

    @patch("msmodelslim.core.quant_service.modelslim_v0.quant_service.get_logger")
    @patch.object(ModelslimV0QuantService, "quant_process")
    def test_quantize_warns_and_delegates_when_device_indices_set(self, mock_quant_process, mock_get_logger, service):
        """场景：quantize 传入 device_indices。预期：warning 后委托 quant_process。"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        quant_config = BaseQuantConfig(apiversion="modelslim_v0", spec={})
        adapter = MagicMock(spec=PipelineInterface)

        service.quantize(
            quant_config=quant_config,
            model_adapter=adapter,
            device_indices=[0, 1],
        )

        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[0][1] == "modelslim_v0"
        mock_quant_process.assert_called_once()

    def test_quant_process_skips_persist_when_save_path_none(self, service):
        """场景：save_path 为 None。预期：校准后跳过 persist 并返回。"""
        mock_calibrator = MagicMock()
        mock_save_method = MagicMock()
        mock_calibrator.save = mock_save_method
        dataset_loader = MagicMock()
        dataset_loader.get_dataset_by_name.return_value = []
        service.dataset_loader = dataset_loader

        model_adapter = MagicMock(spec=PipelineInterface)
        model_adapter.load_model.return_value = MagicMock()
        model_adapter.handle_dataset.return_value = []

        quant_config = ModelslimV0QuantConfig.from_base(
            BaseQuantConfig(
                apiversion="modelslim_v0",
                spec={"calib_cfg": {}, "calib_dataset": "calib.jsonl"},
            )
        )

        import msmodelslim.core.quant_service.modelslim_v0.quant_service as v0_svc

        with patch.object(v0_svc, "Calibrator", return_value=mock_calibrator):
            service.quant_process(quant_config, model_adapter, save_path=None, device=DeviceType.CPU)

        mock_calibrator.run.assert_called_once()
        mock_save_method.assert_not_called()
