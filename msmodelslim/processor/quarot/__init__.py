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
    'QuaRotProcessor',
    'QuaRotProcessorConfig',
    'QuaRotInterface',
    'LAOSOnlineRotationInterface',
    'RotatePair',
    'create_rot',
    'QuaRotMode',
    'OnlineQuaRotProcessor',
    'OnlineQuaRotProcessorConfig',
    'OnlineQuaRotInterface',
    'RotationConfig',
]

from .offline_quarot.quarot import QuaRotProcessor, QuaRotProcessorConfig
from .offline_quarot.quarot_interface import QuaRotInterface, LAOSOnlineRotationInterface, RotatePair
from .common.quarot_utils import create_rot, QuaRotMode
from .online_quarot.online_quarot import OnlineQuaRotProcessor, OnlineQuaRotProcessorConfig
from .online_quarot.online_quarot_interface import OnlineQuaRotInterface, RotationConfig
