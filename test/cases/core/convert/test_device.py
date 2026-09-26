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

msmodelslim.core.convert.device 模块的单元测试
"""

from unittest.mock import patch

import pytest

from msmodelslim.core.convert.device import (
    npu_available,
    resolve_multi_worker_devices,
)
from msmodelslim.utils.exception import EnvError, SchemaValidateError


class TestNpuAvailable:
    """测试 npu_available 函数"""

    def test_npu_available_return_false_when_torch_has_no_npu(self):
        with patch("msmodelslim.core.convert.device.torch") as mock_torch:
            del mock_torch.npu
            assert npu_available() is False  # 校验无 npu 属性时返回 False


class TestResolveMultiWorkerDevices:
    """测试 resolve_multi_worker_devices 函数"""

    def test_resolve_multi_worker_devices_return_empty_when_none_or_empty(self):
        assert resolve_multi_worker_devices(None) == []  # 校验 None 不启用多卡
        assert resolve_multi_worker_devices([]) == []  # 校验空列表不启用多卡

    def test_resolve_multi_worker_devices_raise_when_npu_unavailable(self):
        """场景：指定卡号但 NPU 不可用。
        预期：对齐 quant 语义直接抛错，不静默回落 CPU。
        """
        with patch("msmodelslim.core.convert.device.npu_available", return_value=False):
            with pytest.raises(EnvError, match="no NPU is available"):
                resolve_multi_worker_devices([0, 1, 2])

    def test_resolve_multi_worker_devices_map_indices_to_npu_strings(self):
        with (
            patch("msmodelslim.core.convert.device.npu_available", return_value=True),
            patch("msmodelslim.core.convert.device._npu_device_count", return_value=8),
        ):
            assert resolve_multi_worker_devices([0, 1, 2]) == ["npu:0", "npu:1", "npu:2"]

    def test_resolve_multi_worker_devices_raise_when_duplicate_indices(self):
        with (
            patch("msmodelslim.core.convert.device.npu_available", return_value=True),
            patch("msmodelslim.core.convert.device._npu_device_count", return_value=8),
        ):
            with pytest.raises(SchemaValidateError, match="Duplicate device indices"):
                resolve_multi_worker_devices([0, 0])

    def test_resolve_multi_worker_devices_raise_when_index_out_of_range(self):
        with (
            patch("msmodelslim.core.convert.device.npu_available", return_value=True),
            patch("msmodelslim.core.convert.device._npu_device_count", return_value=2),
        ):
            with pytest.raises(SchemaValidateError, match="out of range"):
                resolve_multi_worker_devices([0, 5])
