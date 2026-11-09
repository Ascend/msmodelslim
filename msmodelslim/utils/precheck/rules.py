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

Reusable precheck rules for arguments, paths, disk, and NPU state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.utils.env import get_dir_size, get_free_space, get_npu_memory
from msmodelslim.utils.exception import EnvError, SchemaValidateError, SecurityError
from msmodelslim.utils.precheck.manager import precheck_rule
from msmodelslim.utils.security import check_read_permission, check_write_permission, get_valid_path
from msmodelslim.utils.validation.conversion import (
    convert_to_readable_dir,
    convert_to_readable_file,
)
from msmodelslim.utils.validation.device import validate_device_index
from msmodelslim.utils.validation.value import validate_str_length

STRING_FIELD_GROUP = "string.fields"
PATH_PERMISSION_GROUP = "path.permission"
DISK_SPACE_GROUP = "disk"
NPU_INDEX_GROUP = "npu.index"
NPU_OCCUPANCY_GROUP = "npu"


def _is_byte_count(value: Any) -> bool:
    """Return whether a value is a non-negative integer byte count."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@precheck_rule(group=STRING_FIELD_GROUP)
def string_field_check(value: Any, name: str = "string", skip_when_none: bool = False) -> bool:
    """Validate a string field's type and length.

    ``None`` is accepted only for optional fields, marked with ``skip_when_none=True``;
    such a field is reported as skipped instead of validated.
    """
    if value is None:
        if skip_when_none:
            return False
        raise SchemaValidateError(
            "%s must be a string, but got %s" % (name, type(value)),
            action="Please ensure %s is a string" % name,
        )
    if not isinstance(value, str):
        raise SchemaValidateError(
            "%s must be a string, but got %s" % (name, type(value)),
            action="Please ensure %s is a string" % name,
        )
    validate_str_length(input_str=value, str_name=name)
    return True


@precheck_rule(group=PATH_PERMISSION_GROUP)
def readable_path_check(path: Any, file: bool = False, skip_when_none: bool = False) -> bool:
    """Validate a readable file or directory after path normalization."""
    if path is None and skip_when_none:
        return False
    real_path = convert_to_readable_file(path) if file else convert_to_readable_dir(path)
    check_read_permission(str(real_path))
    return True


@precheck_rule(group=PATH_PERMISSION_GROUP)
def writable_path_check(path: Any) -> bool:
    """Validate an output directory without creating it."""
    if not isinstance(path, (str, Path)):
        raise SchemaValidateError(
            "save_path must be a string or Path, but got %s" % type(path),
            action="Please pass save_path as a string or Path.",
        )

    real_path = Path(get_valid_path(str(path)))
    if real_path.exists():
        if not real_path.is_dir():
            raise SecurityError(
                "Output path exists but is not a directory: %s" % path,
                action="Please choose a directory path for --save_path.",
            )
    else:
        ancestor = real_path.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise SecurityError(
                "An ancestor of the output path is not a directory: %s" % ancestor,
                action="Please choose a directory path for --save_path.",
            )

    check_write_permission(str(real_path))
    return True


@precheck_rule(group=DISK_SPACE_GROUP, rule_id=DISK_SPACE_GROUP, skippable=True)
def check_disk_space(
    model_path: Any, save_path: Any, size_ratio: float = 1.0, reserve_bytes: int = 0
) -> Union[bool, str]:
    """Require enough output space for the model plus configured reserve.

    A probe that cannot measure the model or the free space returns a pass, because
    the check has no verdict to report; only a measured shortfall raises.
    """
    try:
        model_bytes = get_dir_size(model_path)
    except Exception:
        return True
    if not _is_byte_count(model_bytes):
        return True

    required_bytes = int(model_bytes * size_ratio + reserve_bytes)
    try:
        free_bytes = get_free_space(save_path)
    except Exception:
        return True
    if not _is_byte_count(free_bytes):
        return True

    if required_bytes <= 0:
        raise EnvError(
            "No measurable files in the model directory",
            action="Please check that --model_path is not empty and the path is correct",
        )

    need_gb = required_bytes / (1024**3)
    free_gb = free_bytes / (1024**3)
    if free_bytes >= required_bytes:
        return "model directory size %.2f GB, output partition free %.2f GB" % (
            model_bytes / (1024**3),
            free_gb,
        )

    release_gb = (required_bytes - free_bytes) / (1024**3)
    raise EnvError(
        "Insufficient disk space: %.2f GB free, %.2f GB required, %.2f GB more needed"
        % (
            free_gb,
            need_gb,
            release_gb,
        ),
        action="Please free %.2f GB or change --save_path" % release_gb,
    )


@precheck_rule(group=NPU_INDEX_GROUP, rule_id=NPU_INDEX_GROUP)
def npu_index_check(device_type: Any, device_indices: Any = None) -> bool:
    """Validate device index types and availability.

    When the NPU runtime cannot report its device count the index is passed through,
    so the run itself reports the runtime problem instead of this precheck.
    """
    if not isinstance(device_type, DeviceType):
        return False

    if device_indices is not None and not isinstance(device_indices, list):
        raise SchemaValidateError(
            "device_indices must be a list or None, but got %s" % type(device_indices),
            action="Please pass device indices as a list of integers, e.g. [0, 1].",
        )
    indices = device_indices or ([0] if device_type == DeviceType.NPU else [])
    non_integer_indices = [idx for idx in indices if isinstance(idx, bool) or not isinstance(idx, int)]
    if non_integer_indices:
        raise SchemaValidateError(
            f"Device indices must be integers, but got: {non_integer_indices}",
            action="Please pass device indices as integers, e.g. --device_id 0 1.",
        )

    # Validate values that do not require querying the runtime before probing.
    if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
        validate_device_index(indices, device_type)

    if device_type == DeviceType.NPU:
        try:
            device_count = torch.npu.device_count()
        except Exception:
            return True
        if isinstance(device_count, bool) or not isinstance(device_count, int) or device_count < 0:
            return True

    try:
        validate_device_index(indices, device_type)
    except SchemaValidateError:
        raise
    except Exception:
        return True

    if device_type == DeviceType.CPU and any(idx > 0 for idx in indices):
        raise SchemaValidateError(f"CPU supports only device index 0, but got: {indices}")

    return True


@precheck_rule(group=NPU_OCCUPANCY_GROUP, rule_id=NPU_OCCUPANCY_GROUP, skippable=True)
def npu_occupancy_check(device_type: Any, device_index: int, free_ratio_min: float) -> Union[bool, str]:
    """Check whether an NPU has at least ``free_ratio_min`` free memory.

    A probe that cannot report the memory returns a pass; only a measured free ratio
    below ``free_ratio_min`` raises.
    """
    if device_type != DeviceType.NPU:
        return False
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        return False

    try:
        free_bytes, total_bytes = get_npu_memory(device_index)
    except Exception:
        return True
    if (
        not _is_byte_count(free_bytes)
        or not _is_byte_count(total_bytes)
        or total_bytes == 0
        or free_bytes > total_bytes
    ):
        return True

    free_ratio = free_bytes / total_bytes
    if free_ratio < free_ratio_min:
        raise EnvError(
            "NPU:%s free memory %.2f/%.2f GB, free ratio %.2f below threshold %.2f"
            % (
                device_index,
                free_bytes / (1024**3),
                total_bytes / (1024**3),
                free_ratio,
                free_ratio_min,
            ),
            action="Please switch to another card or wait for the running task to finish",
        )
    return "free memory %.2f/%.2f GB" % (free_bytes / (1024**3), total_bytes / (1024**3))
