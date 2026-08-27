#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quant_service.modelslim_convert.quant_service 模块的单元测试
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.modelslim_convert.quant_service import (
    ModelslimConvertQuantService,
    ModelslimConvertQuantServiceConfig,
    get_plugin,
)


class TestModelslimConvertQuantService:
    """测试 ModelslimConvertQuantService 类"""

    def test_get_plugin_return_config_and_service_classes(self):
        cfg_cls, svc_cls = get_plugin()
        assert cfg_cls is ModelslimConvertQuantServiceConfig
        assert svc_cls is ModelslimConvertQuantService

    def test_plugin_registered_in_entry_points_when_installed(self):
        from importlib.metadata import entry_points

        from msmodelslim.core.quant_service.interface import QUANT_SERVICE_PLUGIN_GROUP

        names = [e.name for e in entry_points().select(group=QUANT_SERVICE_PLUGIN_GROUP)]
        assert "modelslim_convert" in names

    def test_quantize_raise_error_when_save_path_none(self):
        service = ModelslimConvertQuantService(ModelslimConvertQuantServiceConfig())
        model_adapter = MagicMock()
        model_adapter.model_path = Path("/tmp/model")
        quant_config = BaseQuantConfig.model_validate(
            {
                "apiversion": "modelslim_convert",
                "spec": {"linears": [], "save": [{"type": "ascend_v1"}]},
            }
        )
        with pytest.raises(ValueError, match="requires save_path"):
            service.quantize(quant_config, model_adapter, save_path=None)

    @patch("msmodelslim.core.quant_service.modelslim_convert.quant_service.create_convert_application")
    def test_quantize_delegate_to_convert_application_when_save_path_given(self, mock_factory):
        mock_app = MagicMock()
        mock_factory.return_value = mock_app
        service = ModelslimConvertQuantService(ModelslimConvertQuantServiceConfig())
        model_adapter = MagicMock()
        model_adapter.model_path = Path("/data/model")
        model_adapter.model_type = "qwen3_5_moe"
        quant_config = BaseQuantConfig.model_validate(
            {
                "apiversion": "modelslim_convert",
                "spec": {
                    "preprocess": [],
                    "linears": [
                        {
                            "match": ["layers.*.q_proj"],
                            "target": "FLOAT",
                            "route": "auto",
                        },
                    ],
                    "save": [{"type": "ascend_v1"}],
                },
            }
        )
        service.quantize(
            quant_config,
            model_adapter,
            save_path=Path("/data/out"),
            device=DeviceType.CPU,
        )
        mock_factory.assert_called_once()
        mock_app.run.assert_called_once()
        convert_cfg = mock_app.run.call_args[0][0]
        assert convert_cfg.model_path == "/data/model"
        assert convert_cfg.save_path == "/data/out"
        assert convert_cfg.model_family == "qwen3_5_moe"
