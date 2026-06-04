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

from msmodelslim.core.const import DeviceType, QuantType, RunnerType


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_npu_value_equals_npu_when_accessed(self):
        """场景：访问 NPU 枚举成员。预期：值为 npu。"""
        assert DeviceType.NPU.value == "npu"

    def test_cpu_value_equals_cpu_when_accessed(self):
        """场景：访问 CPU 枚举成员。预期：值为 cpu。"""
        assert DeviceType.CPU.value == "cpu"


class TestQuantType:
    """Tests for QuantType enum."""

    def test_w8a8_value_equals_w8a8_when_accessed(self):
        """场景：访问 W8A8 枚举成员。预期：值为 w8a8。"""
        assert QuantType.W8A8.value == "w8a8"

    def test_w4a4_value_equals_w4a4_when_accessed(self):
        """场景：访问 W4A4 枚举成员。预期：值为 w4a4。"""
        assert QuantType.W4A4.value == "w4a4"


class TestRunnerType:
    """Tests for RunnerType enum."""

    def test_auto_value_equals_auto_when_accessed(self):
        """场景：访问 AUTO 枚举成员。预期：值为 auto。"""
        assert RunnerType.AUTO.value == "auto"

    def test_layer_wise_value_equals_layer_wise_when_accessed(self):
        """场景：访问 LAYER_WISE 枚举成员。预期：值为 layer_wise。"""
        assert RunnerType.LAYER_WISE.value == "layer_wise"
