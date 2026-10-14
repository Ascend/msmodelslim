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

from ascend_utils.common.utils import (
    get_attrs_of_obj,  # noqa: F401
    concatenate_name_in_network,  # noqa: F401
    FullPermutation,  # noqa: F401
    CallParams,  # noqa: F401
    ResListToRelease,  # noqa: F401
    amp_enabled,  # noqa: F401
    OperatorAttrName,  # noqa: F401
    check_model_backend,  # noqa: F401
    count_parameters,  # noqa: F401
)
