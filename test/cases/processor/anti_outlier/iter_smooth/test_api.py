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

Unit tests for the iter_smooth() dispatch entry point.
"""

import unittest
from unittest.mock import patch

from msmodelslim.ir.qal.qregistry import QFuncRegistry
from msmodelslim.processor.anti_outlier.common.smooth_types import IterSmoothConfig
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
)
from msmodelslim.processor.anti_outlier.iter_smooth.api import iter_smooth


class _FakeSubgraph:
    """Sentinel subgraph type used only to assert dispatch wiring."""


class TestIterSmoothDispatch(unittest.TestCase):
    """iter_smooth forwards the api name, (type, version) key and args to the registry."""

    def test_dispatch_forwards_name_key_and_args(self):
        subgraph = _FakeSubgraph()
        config = IterSmoothConfig(version=1)
        context = object()
        with patch("msmodelslim.processor.anti_outlier.iter_smooth.api.QFuncRegistry.dispatch") as mock_dispatch:
            iter_smooth(subgraph, config, context)
        mock_dispatch.assert_called_once_with("iter_smooth", (_FakeSubgraph, 1), subgraph, config, context)

    def test_dispatch_key_includes_config_version(self):
        config = IterSmoothConfig(version=2)
        with patch("msmodelslim.processor.anti_outlier.iter_smooth.api.QFuncRegistry.dispatch") as mock_dispatch:
            iter_smooth(_FakeSubgraph(), config, object())
        self.assertEqual(mock_dispatch.call_args.args[1], (_FakeSubgraph, 2))


class TestIterSmoothRegistrations(unittest.TestCase):
    """All supported subgraph types are registered under version 1."""

    def test_expected_subgraph_types_registered(self):
        impl_map = QFuncRegistry._registered_api["iter_smooth"].impl_map
        for subgraph_cls in (
            OVSubgraph,
            UpDownSubgraph,
            LinearLinearSubgraph,
            NormLinearSubgraph,
            NonFusionSubgraph,
        ):
            self.assertIn((subgraph_cls, 1), impl_map)


if __name__ == "__main__":
    unittest.main()
