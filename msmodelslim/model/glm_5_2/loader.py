# -*- coding: UTF-8 -*-

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class Glm52AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.glm_5_2.model_adapter:GLM52ModelAdapter"
