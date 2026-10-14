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
# pylint: disable=pointless-string-statement

"""
Multimodal VLM V1 Quantization Service

A unified quantization service for multimodal vision-language models with:
- Automatic MoE fusion layer conversion
- Layer-wise loading and processing
- Multi-modal calibration dataset support
- Compatible with msmodelslim quant command

Supported models:
- Qwen3-VL-MoE
- Other multimodal VLM models (extensible)
"""

__all__ = [
    'MultimodalVLMModelslimV1QuantService',
    'MultimodalVLMModelslimV1QuantServiceConfig',
    'MultimodalVLMModelslimV1QuantConfig',
]

from .quant_config import MultimodalVLMModelslimV1QuantConfig  # noqa: E402
from .quant_service import MultimodalVLMModelslimV1QuantService, MultimodalVLMModelslimV1QuantServiceConfig  # noqa: E402
