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

DeepSeek-V4.1-Flash model adapter — checkpoint decode/reparameterization and standard
ModelAdapter interfaces. No remote code is required: the module graph is built
from the source safetensors layout, mirroring the checkpoint's own structure.
"""

from msmodelslim.model.deepseek_v4_1_flash.model_adapter import DeepSeekV41FlashModelAdapter

__all__ = ["DeepSeekV41FlashModelAdapter"]
