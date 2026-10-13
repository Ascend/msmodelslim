#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
from typing import Optional, Any, List, Dict, Literal, Tuple

from msmodelslim.core.const import DeviceType
from msmodelslim.core.context.interface import IContextFactory
from msmodelslim.core.quant_service import KeyInfoPersistenceInfra
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.plugin.plugin_utils import load_plugin_config_class
from msmodelslim.utils.plugin.typed_factory import TypedFactory
from ..dataset_loader_infra import DatasetLoaderInfra
from ..interface import (
    IQuantService,
    BaseQuantConfig,
    QuantServiceConfig,
    QUANT_SERVICE_PLUGIN_GROUP,
)
from ..modelslim_v0.pipeline_interface import PipelineInterface as ModelslimV0PipelineInterface

# 模块级工厂：按 apiversion 加载插件后端（proxy 不参与插件，仅做调度）
_QUANT_SERVICE_FACTORY = TypedFactory[IQuantService](config_base_class=QuantServiceConfig)


class QuantServiceProxyConfig(QuantServiceConfig):
    """proxy 调度入口的配置，用于 CLI 等直接构造 QuantServiceProxy。proxy 不作为插件注册。"""

    apiversion: Literal["proxy"] = "proxy"


@logger_setter(prefix='msmodelslim.core.quant_service.proxy')
class QuantServiceProxy(IQuantService):
    """量化服务代理：根据 quant_config.apiversion 通过插件创建 IQuantService 并委托 quantize。"""

    def __init__(
        self,
        quant_service_config: QuantServiceProxyConfig,
        dataset_loader: DatasetLoaderInfra,
        vlm_dataset_loader: Optional[DatasetLoaderInfra] = None,
        context_factory: IContextFactory = None,
        debug_info_persistence: Optional[KeyInfoPersistenceInfra] = None,
        **kwargs,
    ):
        """QuantServiceConfig 与 dataset_loader、vlm_dataset_loader 分开传入。"""
        self.quant_service_config = quant_service_config
        self.dataset_loader = dataset_loader
        self.vlm_dataset_loader = vlm_dataset_loader
        self._service_cache: Dict[Tuple[str, int], IQuantService] = {}
        self.context_factory = context_factory
        self.debug_info_persistence = debug_info_persistence

    def quantize(
        self,
        quant_config: BaseQuantConfig,
        model_adapter: Any,
        save_path: Optional[Path] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        api_version = quant_config.apiversion
        dataset_loader = self._dataset_loader_for_apiversion(api_version, model_adapter)
        cache_key = (api_version, id(dataset_loader))
        if cache_key not in self._service_cache:
            backend_config_class = load_plugin_config_class(QUANT_SERVICE_PLUGIN_GROUP, api_version)
            backend_config = backend_config_class(apiversion=api_version)
            self._service_cache[cache_key] = _QUANT_SERVICE_FACTORY.create(
                backend_config,
                dataset_loader=dataset_loader,
                context_factory=self.context_factory,
                debug_info_persistence=self.debug_info_persistence,
            )
        quant_service = self._service_cache[cache_key]
        quant_service.quantize(
            quant_config=quant_config,
            model_adapter=model_adapter,
            save_path=save_path,
            device=device,
            device_indices=device_indices,
        )

    def _dataset_loader_for_apiversion(self, apiversion: str, model_adapter: Any = None) -> DatasetLoaderInfra:
        """Wire the loader required by a backend/model protocol.

        VLMDatasetLoader determines the actual modality from dataset contents;
        the proxy only injects the loader capability requested by the adapter.
        """
        is_vlm_backend = apiversion in {
            'multimodal_vlm_modelslim_v1',
            'multimodal_sd_modelslim_v1',
        }
        needs_vlm_loader = (
            apiversion == 'modelslim_v0'
            and isinstance(model_adapter, ModelslimV0PipelineInterface)
            and model_adapter.uses_image_calib()
        )
        if is_vlm_backend or needs_vlm_loader:
            if self.vlm_dataset_loader is None:
                raise SchemaValidateError(
                    "vlm_dataset_loader is required by the quantization task",
                    action="Please inject vlm_dataset_loader when constructing QuantServiceProxy.",
                )
            return self.vlm_dataset_loader
        return self.dataset_loader
