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

from msmodelslim.common.low_rank_decompose.low_rank_decompose import (
    RankMethods,  # noqa: F401
    get_hidden_channels_by_layer_name,  # noqa: F401
    is_hidden_channels_valid,  # noqa: F401
    get_decompose_channels_2d,  # noqa: F401
    decompose_weight_2d_svd,  # noqa: F401
    get_decompose_channels_4d,  # noqa: F401
    decompose_weight_4d_tucker,  # noqa: F401
)
