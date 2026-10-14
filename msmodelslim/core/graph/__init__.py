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
Graph Module for MsModelSlim

This module provides graph-related utilities and configurations for model processing,
including subgraph types, adapter configurations, and mapping relationships.
"""

__all__ = [
    # Constants
    "SUPPORTED_SUBGRAPH_TYPES",
    # Configuration Classes
    "MappingConfig",
    "FusionConfig",
    "AdapterConfig",
]

# Import all components from adapter_types module
from .adapter_types import (  # noqa: E402
    SUPPORTED_SUBGRAPH_TYPES,
    MappingConfig,
    FusionConfig,
    AdapterConfig,
)
