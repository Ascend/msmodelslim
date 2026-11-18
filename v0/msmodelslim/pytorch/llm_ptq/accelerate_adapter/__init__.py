#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
    'replace_device_align_hook_if_needed',
    'move_update_weight_hook_if_need',
    'get_state_dict_copy',
    'clear_unused_module',
    'PrepareWeight',

    'get_offloaded_dataset',

    'LazyTensor',
    'handle_lazy_tensor'
]

from .hook_adapter import (replace_device_align_hook_if_needed,
                           move_update_weight_hook_if_need,
                           get_state_dict_copy,
                           clear_unused_module,
                           PrepareWeight)
from .utils import get_offloaded_dataset
from .lazy_handler import LazyTensor, handle_lazy_tensor
