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

Unit tests for the flex_smooth()/flex_awq_ssz() dispatch entry points.
"""

import unittest
from unittest.mock import patch

from msmodelslim.ir.qal.qregistry import QFuncRegistry
from msmodelslim.processor.anti_outlier.common.smooth_types import (
    FlexAWQSSZConfig,
    FlexSmoothQuantConfig,
)
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
)
from msmodelslim.processor.anti_outlier.flex_smooth.api import (
    flex_awq_ssz,
    flex_smooth_quant,
    get_optimal_alpha_beta_flex_smooth,
)


class _FakeSubgraph:
    """Sentinel subgraph type used only to assert dispatch wiring."""


class TestFlexSmoothQuantDispatch(unittest.TestCase):
    def test_dispatch_forwards_name_key_and_args(self):
        subgraph = _FakeSubgraph()
        config = FlexSmoothQuantConfig(version=1)
        context = object()
        with patch("msmodelslim.processor.anti_outlier.flex_smooth.api.QFuncRegistry.dispatch") as mock_dispatch:
            flex_smooth_quant(subgraph, config, context)
        mock_dispatch.assert_called_once_with("flex_smooth_quant", (_FakeSubgraph, 1), subgraph, config, context)


class TestFlexAwqSszDispatch(unittest.TestCase):
    def test_dispatch_forwards_name_key_and_args(self):
        subgraph = _FakeSubgraph()
        config = FlexAWQSSZConfig(version=1)
        context = object()
        with patch("msmodelslim.processor.anti_outlier.flex_smooth.api.QFuncRegistry.dispatch") as mock_dispatch:
            flex_awq_ssz(subgraph, config, context)
        mock_dispatch.assert_called_once_with("flex_awq_ssz", (_FakeSubgraph, 1), subgraph, config, context)


class TestFlexSmoothRegistrations(unittest.TestCase):
    def test_flex_smooth_quant_registers_all_subgraph_types(self):
        impl_map = QFuncRegistry._registered_api["flex_smooth_quant"].impl_map
        for subgraph_cls in (OVSubgraph, UpDownSubgraph, LinearLinearSubgraph, NormLinearSubgraph, NonFusionSubgraph):
            self.assertIn((subgraph_cls, 1), impl_map, msg=subgraph_cls.__name__)

    def test_flex_awq_ssz_registers_fusion_subgraph_types(self):
        impl_map = QFuncRegistry._registered_api["flex_awq_ssz"].impl_map
        for subgraph_cls in (OVSubgraph, UpDownSubgraph, LinearLinearSubgraph, NormLinearSubgraph):
            self.assertIn((subgraph_cls, 1), impl_map, msg=subgraph_cls.__name__)
        self.assertNotIn((NonFusionSubgraph, 1), impl_map)


class TestGetOptimalAlphaBeta(unittest.TestCase):
    def test_returns_provided_alpha_beta(self):
        config = FlexSmoothQuantConfig(version=1, alpha=0.25, beta=0.75)
        result = get_optimal_alpha_beta_flex_smooth(config, None, None)
        self.assertEqual(result, (0.25, 0.75))

    def test_runs_search_when_alpha_or_beta_missing(self):
        config = FlexSmoothQuantConfig(version=1, alpha=None, beta=None)
        with patch(
            "msmodelslim.processor.anti_outlier.flex_smooth.api.FlexSmoothAlphaBetaSearcher"
        ) as mock_searcher_cls:
            mock_searcher_cls.return_value.search_alpha_beta.return_value = (0.3, 0.7, 0.01)
            result = get_optimal_alpha_beta_flex_smooth(config, "act", "weights")
        self.assertEqual(result, (0.3, 0.7))
        mock_searcher_cls.return_value.search_alpha_beta.assert_called_once_with("act", "weights")


if __name__ == "__main__":
    unittest.main()
