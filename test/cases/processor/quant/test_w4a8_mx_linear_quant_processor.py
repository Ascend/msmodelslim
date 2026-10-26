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


from ._linear_quant_processor_contract import MXLinearQuantProcessorContract
from ..processor_test_helpers import TestProcessorBase
from msmodelslim.ir.qal import QDType, QScope


class TestLinearQuantProcessor(MXLinearQuantProcessorContract, TestProcessorBase):
    """Test the LinearQuantProcessor contract for W4A8 MX."""

    weight_dtype = QDType.MXFP4
    activation_dtype = QDType.MXFP8
    weight_scope = QScope.PER_BLOCK
    activation_scope = QScope.PER_BLOCK


if __name__ == '__main__':
    unittest.main()
