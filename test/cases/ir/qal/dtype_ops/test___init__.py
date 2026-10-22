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

from msmodelslim.ir.qal import dtype_ops
from msmodelslim.ir.qal.dtype_ops.fp8_ops import Fp8
from msmodelslim.ir.qal.dtype_ops.int4_ops import Int4
from msmodelslim.ir.qal.dtype_ops.int8_ops import Int8
from msmodelslim.ir.qal.dtype_ops.mxfp4_ops import Mxfp4
from msmodelslim.ir.qal.dtype_ops.mxfp8_ops import Mxfp8


class TestDtypeOpsInit(unittest.TestCase):
    """被测目标：dtype_ops 包导出。不重复测量化公式。"""

    def test_all_ends_with_sym_or_asym_when_exported(self):
        """场景：公开名。预期：64 个都以 _sym 或 _asym 结尾。"""
        self.assertEqual(len(dtype_ops.__all__), 64)
        for name in dtype_ops.__all__:
            self.assertTrue(name.endswith("_sym") or name.endswith("_asym"), name)

    def test_all_omits_pack_names_when_exported(self):
        """场景：公开名。预期：没有 pack / unpack。"""
        for name in dtype_ops.__all__:
            self.assertNotIn("pack", name)
            self.assertNotIn("unpack", name)

    def test_classes_not_in_all_when_exported(self):
        """场景：格式类。预期：Int8 等不进 __all__。"""
        for cls in (Int8, Int4, Fp8, Mxfp8, Mxfp4):
            self.assertNotIn(cls.__name__, dtype_ops.__all__)

    def test_alias_is_class_method_when_int8_quantize_per_tensor_sym(self):
        """场景：模块级别名。预期：与 Int8 方法是同一对象。"""
        self.assertIs(dtype_ops.int8_quantize_per_tensor_sym, Int8.quantize_per_tensor_sym)


if __name__ == "__main__":
    unittest.main()
