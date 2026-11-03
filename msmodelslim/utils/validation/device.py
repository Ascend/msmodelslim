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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from typing import List, Optional

import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError


def validate_device_index(device_index: Optional[List[int]], device_type: DeviceType) -> None:
    """
    Validate device_index parameter.

    Args:
        device_index: Device indices to validate
        device_type: Device type for context validation

    Raises:
        SchemaValidateError: If device_index is invalid
    """
    if device_index is None:
        return

    # Value validation: check if indices are non-negative
    if any(idx < 0 for idx in device_index):
        negative_indices = [idx for idx in device_index if idx < 0]
        raise SchemaValidateError(
            f"Device indices must be non-negative integers, but got negative values: {negative_indices}"
        )

    # Value validation: check for duplicates
    if len(device_index) != len(set(device_index)):
        duplicates = [idx for idx in set(device_index) if device_index.count(idx) > 1]
        raise SchemaValidateError(f"Device indices must be unique, but found duplicates: {duplicates}")

    # CPU does not support multi-device
    if device_type == DeviceType.CPU and len(device_index) > 1:
        raise SchemaValidateError(
            f"CPU does not support multi-device configuration. "
            f"Got device indices: {device_index}. "
            f"Please use NPU for multi-device parallel, or use single CPU device."
        )

    # Value validation: check device availability
    if device_type == DeviceType.NPU:
        max_device_count = torch.npu.device_count()
    else:
        # CPU doesn't need device count validation
        max_device_count = None

    # Check if indices exceed available devices
    if max_device_count is not None:
        invalid_indices = [idx for idx in device_index if idx >= max_device_count]
        if invalid_indices:
            raise SchemaValidateError(
                f"Device indices {invalid_indices} exceed maximum available device index "
                f"({max_device_count - 1}). Available device indices: 0 to {max_device_count - 1}"
            )
