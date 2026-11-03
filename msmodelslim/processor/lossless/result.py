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

from typing import Dict, Iterable


def new_lossless_check_result(include: Iterable[str], exclude: Iterable[str]) -> Dict:
    """Create a serializable result with an explicit unknown/pass/fail state."""

    return {
        "passed": True,
        "skipped": False,
        "skip_reason": None,
        "checks": [],
        "anomalies": [],
        "missing_targets": [],
        "skipped_targets": [],
        "include": list(include),
        "exclude": list(exclude),
        "matched_targets": [],
        "checked_targets": [],
        "unmatched_include": [],
        "unmatched_exclude": [],
        "total_checks": 0,
    }
