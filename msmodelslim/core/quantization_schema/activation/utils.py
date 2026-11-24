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

from msmodelslim.utils.logging import get_logger


_HIF8_FIXED_SCALES = (8, 15, 32768)
_HIF8_FIXED_DEFAULT = 8.0


def resolve_hif8_static_scale(scale: object) -> float:
    """Resolve the configured HiF8 fixed scale."""
    if scale in _HIF8_FIXED_SCALES:
        return float(scale)
    if scale is None:
        get_logger().info("HiF8 fixed scale is not configured; using default %g.", _HIF8_FIXED_DEFAULT)
    else:
        get_logger().warning("Unsupported HiF8 fixed scale %r; using default %g.", scale, _HIF8_FIXED_DEFAULT)
    return _HIF8_FIXED_DEFAULT
