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

# pylint: disable=abstract-class-instantiated

import unittest

from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.processor.anti_outlier.oasq.interface import OASQInterface


class ConcreteOASQInterface(OASQInterface):
    def get_adapter_config_for_subgraph(self):
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(source="norm", targets=["linear"]),
            )
        ]


class TestOASQInterface(unittest.TestCase):
    """OASQInterface is an abstract interface returning adapter configs."""

    def test_cannot_instantiate_interface_directly(self):
        with self.assertRaises(TypeError):
            OASQInterface()

    def test_interface_declares_abstract_method(self):
        self.assertTrue(hasattr(OASQInterface, "get_adapter_config_for_subgraph"))
        self.assertIn("get_adapter_config_for_subgraph", OASQInterface.__abstractmethods__)

    def test_subclass_returns_adapter_configs(self):
        configs = ConcreteOASQInterface().get_adapter_config_for_subgraph()
        self.assertEqual(len(configs), 1)
        self.assertIsInstance(configs[0], AdapterConfig)
        self.assertEqual(configs[0].subgraph_type, "norm-linear")


if __name__ == "__main__":
    unittest.main()
