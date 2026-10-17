# -*- coding: UTF-8 -*-
"""DeepSeek-V4.1-Flash adapter loader, registered via config.ini [ModelAdapterEntryPoints]."""

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class DeepSeekV41FlashAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.deepseek_v4_1_flash.model_adapter:DeepSeekV41FlashModelAdapter"

    def get_adapter_config_class(self):
        """pydantic model for the practice ``spec.model_adapter`` control params."""
        from msmodelslim.model.deepseek_v4_1_flash.model_adapter import DeepSeekV41FlashModelConfig

        return DeepSeekV41FlashModelConfig
