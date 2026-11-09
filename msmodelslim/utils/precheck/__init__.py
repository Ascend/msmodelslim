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

Precheck framework and reusable string/path rules.
"""

from .manager import (
    PrecheckManager,
    PrecheckReport,
    PrecheckResult,
    PrecheckRule,
    PrecheckStatus,
    precheck_rule,
)
from .rules import (
    DISK_SPACE_GROUP,
    NPU_INDEX_GROUP,
    NPU_OCCUPANCY_GROUP,
    PATH_PERMISSION_GROUP,
    STRING_FIELD_GROUP,
    check_disk_space,
    npu_index_check,
    npu_occupancy_check,
    readable_path_check,
    string_field_check,
    writable_path_check,
)

__all__ = [
    "DISK_SPACE_GROUP",
    "NPU_INDEX_GROUP",
    "NPU_OCCUPANCY_GROUP",
    "PATH_PERMISSION_GROUP",
    "STRING_FIELD_GROUP",
    "PrecheckManager",
    "PrecheckReport",
    "PrecheckResult",
    "PrecheckRule",
    "PrecheckStatus",
    "check_disk_space",
    "npu_index_check",
    "npu_occupancy_check",
    "precheck_rule",
    "readable_path_check",
    "string_field_check",
    "writable_path_check",
]
