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

import unittest
from test.cases.ir.test_activation_static import BaseActivationTestMixin
from msmodelslim.ir.fp8_activation_static import FP8FakeQuantActivationPerHead
from msmodelslim.ir.const import fp8_e4m3_per_head_sym
from msmodelslim.ir.qal import QDType


class TestFP8FakeQuantActivationPerHead(BaseActivationTestMixin, unittest.TestCase):
    """测试 FP8FakeQuantActivationPerHead 类"""

    module_class = FP8FakeQuantActivationPerHead
    scheme_const = fp8_e4m3_per_head_sym
    fake_quantize_path = "msmodelslim.ir.activation_static.fake_quantize"
    expected_dtype = QDType.FP8_E4M3


if __name__ == '__main__':
    unittest.main()
