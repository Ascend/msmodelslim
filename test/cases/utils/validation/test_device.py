#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from unittest.mock import patch

import pytest
import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.validation.device import validate_device_index


def test_validate_device_index_returns_when_indices_none():
    assert validate_device_index(None, DeviceType.NPU) is None


def test_validate_device_index_raises_when_negative():
    with pytest.raises(SchemaValidateError, match="non-negative"):
        validate_device_index([-1, 0], DeviceType.NPU)


def test_validate_device_index_raises_when_duplicates():
    with pytest.raises(SchemaValidateError, match="unique"):
        validate_device_index([0, 0], DeviceType.NPU)


def test_validate_device_index_raises_when_cpu_multi_device():
    with pytest.raises(SchemaValidateError, match="CPU does not support multi-device"):
        validate_device_index([0, 1], DeviceType.CPU)


def test_validate_device_index_raises_when_index_exceeds_npu_count():
    mock_npu = type("Npu", (), {"device_count": staticmethod(lambda: 2)})()
    with patch.object(torch, "npu", mock_npu, create=True):
        with pytest.raises(SchemaValidateError, match="exceed maximum"):
            validate_device_index([0, 2], DeviceType.NPU)


def test_validate_device_index_passes_when_cpu_single():
    assert validate_device_index([0], DeviceType.CPU) is None
