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

from msmodelslim.model.qwen3_5_moe.loader import Qwen3_5MoeAdapterLoader


class TestQwen3_5MoeAdapterLoader(unittest.TestCase):
    """测试Qwen3_5MoeAdapterLoader的功能"""

    def test_assert_adapter_class_path_when_defined(self):
        """测试ADAPTER_CLASS_PATH属性：应指向正确的适配器类路径"""
        self.assertEqual(
            Qwen3_5MoeAdapterLoader.ADAPTER_CLASS_PATH,
            "msmodelslim.model.qwen3_5_moe.model_adapter:Qwen3_5ModelAdapter",
        )

    def test_assert_is_subclass_when_inheritance_checked(self):
        """测试继承关系：应为BaseModelAdapterLoader的子类"""
        from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader

        self.assertTrue(issubclass(Qwen3_5MoeAdapterLoader, BaseModelAdapterLoader))

    def test_assert_instantiable_when_created(self):
        """测试实例化：应能正常创建实例"""
        loader = Qwen3_5MoeAdapterLoader()
        self.assertIsInstance(loader, Qwen3_5MoeAdapterLoader)


if __name__ == '__main__':
    unittest.main()
