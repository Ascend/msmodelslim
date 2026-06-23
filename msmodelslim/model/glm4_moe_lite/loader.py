# -*- coding: UTF-8 -*-

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class Glm4MoeLiteFlashAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.glm4_moe_lite.model_adapter:GLM4MoeLiteFlashModelAdapter"
