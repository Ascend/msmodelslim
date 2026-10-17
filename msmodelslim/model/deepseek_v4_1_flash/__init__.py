# -*- coding: UTF-8 -*-
"""DeepSeek-V4.1-Flash model adapter — checkpoint decode/reparameterization and standard
ModelAdapter interfaces. No remote code is required: the module graph is built
from the source safetensors layout, mirroring the checkpoint's own structure.
"""

from msmodelslim.model.deepseek_v4_1_flash.model_adapter import DeepSeekV41FlashModelAdapter

__all__ = ["DeepSeekV41FlashModelAdapter"]
