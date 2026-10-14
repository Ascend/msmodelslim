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

from datetime import timedelta
from pathlib import Path
from typing import Union
import re

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.security import get_valid_read_path, get_write_directory


def convert_to_bool(obj: Union[str, bool], param_name="value") -> bool:
    if isinstance(obj, bool):
        return obj
    if obj == "True":
        return True
    if obj == "False":
        return False
    raise SchemaValidateError(
        f"{param_name} must be a string or bool, but got {type(obj)}",
        action="Please ensure the input is literally True or False",
    )


def convert_to_timedelta(obj: Union[str, timedelta], param_name="value") -> timedelta:
    if isinstance(obj, timedelta):
        return obj
    if isinstance(obj, str):
        # 必须按顺序 D H M S，可省略中间任意一段，例如 "1D2H", "30M", "45S"
        m = re.fullmatch(r'(?:(\d+)D)?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', obj)
        if not m or all(v is None for v in m.groups()):
            raise SchemaValidateError(
                f"{param_name} has invalid timedelta format: {obj!r}",
                action=(
                    "Please use '1D2H30M15S' style with units in fixed order D, H, M, S; "
                    "any segment can be omitted, e.g. '1D2H', '30M', or '10S'."
                ),
            )

        days, hours, minutes, seconds = (int(v) if v is not None else 0 for v in m.groups())
        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    raise SchemaValidateError(
        f"{param_name} must be a string or timedelta, but got {type(obj)}",
        action=(
            "Please ensure the input is a timedelta or a string like '1D', '2H30M', "
            "'45M10S', using D for days, H for hours, M for minutes, and S for seconds."
        ),
    )


def convert_to_readable_dir(obj: Union[str, Path], param_name="A readable dir") -> Path:
    if isinstance(obj, str):
        obj = get_valid_read_path(obj, is_dir=True)
        return Path(obj)
    if isinstance(obj, Path):
        obj = get_valid_read_path(str(obj), is_dir=True)
        return Path(obj)
    raise SchemaValidateError(
        f"{param_name} must be a string or Path, but got {type(obj)}",
        action="Please ensure the input is a string or Path",
    )


def convert_to_writable_dir(obj: Union[str, Path], param_name="A writable dir") -> Path:
    if isinstance(obj, str):
        obj = get_write_directory(obj)
        return Path(obj)
    if isinstance(obj, Path):
        obj = get_write_directory(str(obj))
        return Path(obj)
    raise SchemaValidateError(
        f"{param_name} must be a string or Path, but got {type(obj)}",
        action="Please ensure the input is a string or Path",
    )


def convert_to_readable_file(obj: Union[str, Path], param_name="A readable file") -> Path:
    if isinstance(obj, str):
        obj = get_valid_read_path(obj, is_dir=False)
        return Path(obj)
    if isinstance(obj, Path):
        obj = get_valid_read_path(str(obj), is_dir=False)
        return Path(obj)
    raise SchemaValidateError(
        f"{param_name} must be a string or Path, but got {type(obj)}",
        action="Please ensure the input is a string or Path",
    )
