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

from typing import Tuple, Optional, List

from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError


def parse_device_string(device_str: str) -> Tuple[DeviceType, Optional[List[int]]]:
    """
    Parse device string into device type string and device indices list.

    Args:
        device_str: Device specification string (e.g., 'npu', 'npu:0,1,2,3', 'cpu')

    Returns:
        device_type: str, device type string
        device_indices: Optional[List[int]], list of device indices, None if not specified

    Raises:
        SchemaValidateError: If device_str is empty or device indices cannot be converted to integers

    Examples:
        >>> parse_device_string('npu')
        ('npu', None)
        >>> parse_device_string('npu:0,1,2,3')
        ('npu', [0, 1, 2, 3])
        >>> parse_device_string('cpu')
        ('cpu', None)
    """
    device_str = device_str.strip()
    if not device_str:
        raise SchemaValidateError("device string cannot be empty")

    # Split by colon to separate device type and indices
    parts = device_str.split(':', 1)
    device_type_str = parts[0].strip()

    # Validate and convert device type
    try:
        device_type = DeviceType(device_type_str)
    except ValueError as e:
        valid_types = ', '.join([f"'{dt.value}'" for dt in DeviceType])
        raise SchemaValidateError(
            f"Invalid device type: '{device_type_str}'. Supported device types: {valid_types}"
        ) from e

    # Parse device indices if provided
    device_indices = None
    if len(parts) > 1:
        indices_str = parts[1].strip()
        if indices_str:  # Only process if not empty
            # Split by comma and convert to integers
            indices_list = [idx.strip() for idx in indices_str.split(',') if idx.strip()]

            if not indices_list:
                raise SchemaValidateError(f"Device indices cannot be empty after parsing: '{indices_str}'")

            # Convert to integers
            try:
                device_indices = [int(idx) for idx in indices_list]
            except ValueError as e:
                raise SchemaValidateError(
                    f"Invalid device indices format: '{indices_str}'. "
                    f"Expected comma-separated integers (e.g., '0,1,2,3')"
                ) from e

    return device_type, device_indices
