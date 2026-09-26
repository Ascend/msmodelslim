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
        # --device cpu：忽略 device_id 并清空卡号，走 CPU 路径
        assert convert_cfg.parallel.device_indices == []

    @patch("msmodelslim.core.quant_service.modelslim_convert.quant_service.create_convert_application")
    def test_quantize_default_card_zero_when_device_npu_and_no_indices(self, mock_factory):
        """场景：device=npu（CLI 默认）且未传 --device_id。
        预期：对齐 quant 语义，默认卡 0 走 NPU 路径。
        """
        mock_app = MagicMock()
        mock_factory.return_value = mock_app
        service = ModelslimConvertQuantService(ModelslimConvertQuantServiceConfig())
        model_adapter = MagicMock()
        model_adapter.model_path = Path("/data/model")
        model_adapter.model_type = "qwen3_5_moe"
        quant_config = BaseQuantConfig.model_validate(
            {
                "apiversion": "modelslim_convert",
                "spec": {"linears": [], "save": [{"type": "ascend_v1"}]},
            }
        )
        service.quantize(quant_config, model_adapter, save_path=Path("/data/out"), device=DeviceType.NPU)
        convert_cfg = mock_app.run.call_args[0][0]
        assert convert_cfg.parallel.device_indices == [0]

    @patch("msmodelslim.core.quant_service.modelslim_convert.quant_service.create_convert_application")
    def test_quantize_pass_indices_when_device_npu_and_indices_given(self, mock_factory):
        """场景：device=npu 且显式传卡号。
        预期：卡号原样透传。
        """
        mock_app = MagicMock()
        mock_factory.return_value = mock_app
        service = ModelslimConvertQuantService(ModelslimConvertQuantServiceConfig())
        model_adapter = MagicMock()
        model_adapter.model_path = Path("/data/model")
        model_adapter.model_type = "qwen3_5_moe"
        quant_config = BaseQuantConfig.model_validate(
            {
                "apiversion": "modelslim_convert",
                "spec": {"linears": [], "save": [{"type": "ascend_v1"}]},
            }
        )
        service.quantize(
            quant_config,
            model_adapter,
            save_path=Path("/data/out"),
            device=DeviceType.NPU,
            device_indices=[1, 3],
        )
        convert_cfg = mock_app.run.call_args[0][0]
        assert convert_cfg.parallel.device_indices == [1, 3]
