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
    effective_convert_workers,
    npu_available,
    resolve_multi_worker_devices,
    resolve_worker_device,
)


class TestNpuAvailable:
    """测试 npu_available 函数"""

    def test_npu_available_return_false_when_torch_has_no_npu(self):
        with patch("msmodelslim.core.convert.device.torch") as mock_torch:
            del mock_torch.npu
            assert npu_available() is False  # 校验无 npu 属性时返回 False


class TestResolveWorkerDevice:
    """测试 resolve_worker_device 函数"""

    def test_resolve_worker_device_return_cpu_when_auto_and_no_npu(self):
        with patch("msmodelslim.core.convert.device.npu_available", return_value=False):
            assert resolve_worker_device("auto") == "cpu"  # 校验 auto 无 NPU 时回落 CPU

    def test_resolve_worker_device_return_cpu_when_explicit_cpu(self):
        assert resolve_worker_device("cpu") == "cpu"  # 校验显式 cpu

    def test_resolve_worker_device_return_npu0_when_auto_and_npu_available(self):
        with patch("msmodelslim.core.convert.device.npu_available", return_value=True):
            assert resolve_worker_device("auto") == "npu:0"  # 校验 auto 有 NPU 时用 npu:0

    def test_resolve_worker_device_return_cpu_when_npu_specified_but_unavailable(self):
        with patch("msmodelslim.core.convert.device.npu_available", return_value=False):
            assert resolve_worker_device("npu") == "cpu"  # 校验 npu 不可用时回落 CPU

    def test_resolve_worker_device_raise_error_when_unsupported_spec(self):
        with pytest.raises(ValueError, match="Unsupported worker_device"):
            resolve_worker_device("cuda:0")


class TestEffectiveConvertWorkers:
    """测试 effective_convert_workers 函数"""

    def test_effective_convert_workers_return_max_workers_when_cpu_device(self):
        assert effective_convert_workers(4, "cpu", npu_max_workers=1) == 4  # 校验 CPU 不压并发

    def test_effective_convert_workers_cap_workers_when_npu_device(self):
        assert effective_convert_workers(8, "npu:0", npu_max_workers=2) == 2  # 校验 NPU 受 npu_max_workers 限制

    def test_effective_convert_workers_return_one_when_max_workers_zero(self):
        assert effective_convert_workers(0, "cpu", npu_max_workers=1) == 1  # 校验至少 1 worker


class TestResolveMultiWorkerDevices:
    """测试 resolve_multi_worker_devices 函数"""

    def test_resolve_multi_worker_devices_return_empty_when_disabled(self):
        assert resolve_multi_worker_devices(None) == []
        assert resolve_multi_worker_devices([]) == []
        with patch("msmodelslim.core.convert.device.npu_available", return_value=False):
            assert resolve_multi_worker_devices([0, 1, 2]) == []

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
            with pytest.raises(ValueError, match="Duplicate device indices"):
                resolve_multi_worker_devices([0, 0])

    def test_resolve_multi_worker_devices_raise_when_index_out_of_range(self):
        with (
            patch("msmodelslim.core.convert.device.npu_available", return_value=True),
            patch("msmodelslim.core.convert.device._npu_device_count", return_value=2),
        ):
            with pytest.raises(ValueError, match="out of range"):
                resolve_multi_worker_devices([0, 5])
