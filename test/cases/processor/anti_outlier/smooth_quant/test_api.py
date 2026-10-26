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

Unit tests for the smooth_quant() dispatch wiring.
"""

import unittest
from unittest.mock import patch

from msmodelslim.ir.qal.qregistry import QFuncRegistry
from msmodelslim.processor.anti_outlier.common.smooth_types import SmoothQuantConfig
from msmodelslim.processor.anti_outlier.common.subgraph_type import NormLinearSubgraph
from msmodelslim.processor.anti_outlier.smooth_quant.api import smooth_quant


class _FakeSubgraph:
    """Sentinel subgraph type used only to assert dispatch wiring."""


class TestSmoothQuantDispatch(unittest.TestCase):
    """smooth_quant forwards the api name, (type, version) key and args."""

    def test_dispatch_forwards_name_key_and_args(self):
        subgraph = _FakeSubgraph()
        config = SmoothQuantConfig(version=1)
        context = object()
        with patch("msmodelslim.processor.anti_outlier.smooth_quant.api.QFuncRegistry.dispatch") as mock_dispatch:
            smooth_quant(subgraph, config, context)
        mock_dispatch.assert_called_once_with("smooth_quant", (_FakeSubgraph, 1), subgraph, config, context)

    def test_dispatch_key_includes_config_version(self):
        config = SmoothQuantConfig(version=3)
        with patch("msmodelslim.processor.anti_outlier.smooth_quant.api.QFuncRegistry.dispatch") as mock_dispatch:
            smooth_quant(_FakeSubgraph(), config, object())
        self.assertEqual(mock_dispatch.call_args.args[1], (_FakeSubgraph, 3))


class TestSmoothQuantRegistration(unittest.TestCase):
    """smooth_quant registers the norm-linear implementation for version 1."""

    def test_norm_linear_registered(self):
        impl_map = QFuncRegistry._registered_api["smooth_quant"].impl_map
        self.assertIn((NormLinearSubgraph, 1), impl_map)


if __name__ == "__main__":
    unittest.main()
