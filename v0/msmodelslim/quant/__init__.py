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

__all__ = [
    "quant_model",
    "SessionConfig",
    "M3ProcessorConfig",
    "M4ProcessorConfig",
    "M6Config",
    "M6ProcessorConfig",
    "W8A8QuantConfig",
    "W8A8ProcessorConfig",
    "FA3ProcessorConfig",
    "W8A8DynamicQuantConfig",
    "W8A8DynamicProcessorConfig",
    "W8A8TimeStepQuantConfig",
    "W8A8TimeStepProcessorConfig",
    "SaveProcessorConfig",
]

try:
    from msmodelslim.quant.session.session import quant_model, SessionConfig
    from msmodelslim.quant.session.session import (
        M4ProcessorConfig,
        W8A8QuantConfig,
        W8A8ProcessorConfig,
        FA3ProcessorConfig,
        W8A8DynamicQuantConfig,
        W8A8DynamicProcessorConfig,
        W8A8TimeStepQuantConfig,
        W8A8TimeStepProcessorConfig,
        SaveProcessorConfig,
        M3ProcessorConfig,
        M6ProcessorConfig,
        M6Config,
    )
except ImportError:
    from msmodelslim.utils.logging import get_logger

    get_logger().warning(
        "The session module is imported failed, but it is not a critical error, because v1 does not use it"
    )
