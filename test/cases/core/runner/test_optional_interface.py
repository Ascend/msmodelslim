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

from msmodelslim.core.runner.optional_interface import LayerWiseOffloadOptionalInterface


class TestLayerWiseOffloadOptionalInterface:
    """Tests for LayerWiseOffloadOptionalInterface."""

    def test_get_layer_wise_offload_device_returns_none_when_default(self):
        """场景：使用默认实现。预期：返回 None。"""
        iface = LayerWiseOffloadOptionalInterface()
        assert iface.get_layer_wise_offload_device() is None
