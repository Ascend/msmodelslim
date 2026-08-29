# -*- coding: UTF-8 -*-

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class Glm5NextAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.glm5_next.model_adapter:Glm5NextModelAdapter"
