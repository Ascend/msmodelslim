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

import sys
from pathlib import Path


_v0_root = Path(__file__).resolve().parent.parent / "v0"
if _v0_root.is_dir():
    _v0_root_str = str(_v0_root)
    if _v0_root_str not in sys.path:
        sys.path.insert(0, _v0_root_str)

    _v0_msmodelslim = _v0_root / "msmodelslim"
    if _v0_msmodelslim.is_dir():
        _v0_msmodelslim_str = str(_v0_msmodelslim)
        if _v0_msmodelslim_str not in __path__:
            __path__.append(_v0_msmodelslim_str)

__all__ = ['set_logger_level', 'logger']

from msmodelslim.utils.logging import logger, set_logger_level  # noqa: E402
from msmodelslim.utils.patch import patch_torch, patch_pydantic  # noqa: E402

patch_torch()
patch_pydantic()

OLD_PACKAGE_NAME = 'modelslim'
NEW_PACKAGE_NAME = 'msmodelslim'
