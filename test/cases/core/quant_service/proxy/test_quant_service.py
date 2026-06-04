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

from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.proxy.quant_service import (
    QuantServiceProxy,
    QuantServiceProxyConfig,
    _QUANT_SERVICE_FACTORY,
)


class TestQuantServiceProxy:
    """Tests for QuantServiceProxy."""

    @patch.object(_QUANT_SERVICE_FACTORY, "create")
    @patch("msmodelslim.core.quant_service.proxy.quant_service.load_plugin_config_class")
    def test_quantize_creates_backend_via_factory_when_first_call(self, mock_load_config_class, mock_factory_create):
        """场景：首次 quantize 某 apiversion。预期：TypedFactory.create 被调用并委托 quantize。"""
        mock_backend_config_class = MagicMock(return_value=MagicMock())
        mock_load_config_class.return_value = mock_backend_config_class

        backend_service = MagicMock()
        mock_factory_create.return_value = backend_service

        dataset_loader = MagicMock()
        proxy = QuantServiceProxy(
            quant_service_config=QuantServiceProxyConfig(),
            dataset_loader=dataset_loader,
        )
        quant_config = BaseQuantConfig(apiversion="modelslim_v1", spec={})
        model_adapter = MagicMock()
        save_path = Path("/tmp/out")

        proxy.quantize(
            quant_config=quant_config,
            model_adapter=model_adapter,
            save_path=save_path,
            device=DeviceType.NPU,
            device_indices=[0],
        )

        mock_load_config_class.assert_called_once()
        mock_factory_create.assert_called_once()
        backend_service.quantize.assert_called_once_with(
            quant_config=quant_config,
            model_adapter=model_adapter,
            save_path=save_path,
            device=DeviceType.NPU,
            device_indices=[0],
        )

    @patch.object(_QUANT_SERVICE_FACTORY, "create")
    @patch("msmodelslim.core.quant_service.proxy.quant_service.load_plugin_config_class")
    def test_quantize_reuses_cached_backend_when_same_apiversion(self, mock_load_config_class, mock_factory_create):
        """场景：相同 apiversion 二次 quantize。预期：factory.create 仅调用一次。"""
        mock_load_config_class.return_value = MagicMock(return_value=MagicMock())
        backend_service = MagicMock()
        mock_factory_create.return_value = backend_service

        proxy = QuantServiceProxy(
            quant_service_config=QuantServiceProxyConfig(),
            dataset_loader=MagicMock(),
        )
        quant_config = BaseQuantConfig(apiversion="modelslim_v0", spec={})

        proxy.quantize(quant_config=quant_config, model_adapter=MagicMock())
        proxy.quantize(quant_config=quant_config, model_adapter=MagicMock())

        assert mock_factory_create.call_count == 1

    def test_dataset_loader_for_apiversion_returns_vlm_loader_when_multimodal_vlm(self):
        """场景：apiversion 为 multimodal_vlm_modelslim_v1。预期：使用 vlm_dataset_loader。"""
        vlm_loader = MagicMock()
        default_loader = MagicMock()
        proxy = QuantServiceProxy(
            quant_service_config=QuantServiceProxyConfig(),
            dataset_loader=default_loader,
            vlm_dataset_loader=vlm_loader,
        )
        loader = proxy._dataset_loader_for_apiversion("multimodal_vlm_modelslim_v1")
        assert loader is vlm_loader
