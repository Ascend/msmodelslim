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

Plugin loader for Qwen3.8-Flash-Next (qwen4_exp) msModelSlim adapter.
"""

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class Qwen4ExpFlashNextAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.qwen4_exp_flash_next.model_adapter:Qwen4ExpFlashNextModelAdapter"

    def get_adapter_config_class(self):
        """pydantic model for the practice ``spec.model_adapter`` control params."""
        from msmodelslim.model.qwen4_exp_flash_next.model_adapter import (
            Qwen4ExpFlashNextModelConfig,
        )

        return Qwen4ExpFlashNextModelConfig
