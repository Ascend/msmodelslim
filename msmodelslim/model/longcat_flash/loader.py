# -*- coding: UTF-8 -*-

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class LongCatFlashAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.longcat_flash.model_adapter:LongCatFlashModelAdapter"
