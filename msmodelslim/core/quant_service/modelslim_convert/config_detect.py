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

检测 YAML 是否为 modelslim_convert 任务配置。
"""

from __future__ import annotations

from pathlib import Path

from msmodelslim.utils.security import yaml_safe_load

MODELSLIM_CONVERT_APIVERSION = "modelslim_convert"


def is_modelslim_convert_config(config_path: str | Path) -> bool:
    """``config_path`` 指向的 YAML 是否 ``apiversion: modelslim_convert``。"""
    raw = yaml_safe_load(str(config_path))
    if not isinstance(raw, dict):
        return False
    return raw.get("apiversion") == MODELSLIM_CONVERT_APIVERSION
