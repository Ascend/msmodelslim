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

Unit tests for the alpha/beta phase search in flex_smooth.alpha_beta_search.
"""

# pylint: disable=abstract-class-instantiated

import unittest
from unittest.mock import MagicMock, patch

import torch

from msmodelslim.processor.anti_outlier.flex_smooth.alpha_beta_search import (
    BaseAlphaBetaSearcher,
    FlexAWQSSZAlphaBetaSearcher,
    FlexSmoothAlphaBetaSearcher,
)


class _AlphaStub(FlexSmoothAlphaBetaSearcher):
    """Evaluator with a single grid minimum at alpha=0.4."""

    def evaluate_alpha_beta(self, act, weights, alpha, beta, **kwargs):
        return abs(alpha - 0.4)


class _BetaStub(FlexSmoothAlphaBetaSearcher):
    """Evaluator with a single grid minimum at beta=0.3."""

    def evaluate_alpha_beta(self, act, weights, alpha, beta, **kwargs):
        return abs(beta - 0.3)


class TestFlexSmoothAlphaBetaSearcher(unittest.TestCase):
    """search_alpha/search_beta pick the grid value minimizing the evaluator."""

    def setUp(self):
        self.act = torch.randn(4, 3)
        self.weights = torch.randn(3, 3)

    def test_search_alpha_returns_best_alpha_and_mse(self):
        best_alpha, best_mse = _AlphaStub().search_alpha(self.act, self.weights)
        self.assertAlmostEqual(float(best_alpha), 0.4, places=6)
        self.assertAlmostEqual(float(best_mse), 0.0, places=6)

    def test_search_beta_returns_best_beta_and_mse(self):
        best_beta, best_mse = _BetaStub().search_beta(self.act, self.weights, alpha=0.5)
        self.assertAlmostEqual(float(best_beta), 0.3, places=6)
        self.assertAlmostEqual(float(best_mse), 0.0, places=6)

    def test_search_beta_respects_normal_mse_best_floor(self):
        # An already-good score still allows a strictly better beta to win.
        best_beta, best_mse = _BetaStub().search_beta(self.act, self.weights, alpha=0.5, normal_mse_best=0.05)
        self.assertAlmostEqual(float(best_beta), 0.3, places=6)
        self.assertLessEqual(float(best_mse), 0.05)

    def test_search_alpha_beta_returns_triple(self):
        result = _AlphaStub().search_alpha_beta(self.act, self.weights)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(float(result[0]), 0.4, places=6)


class TestFlexAWQSSZAlphaBetaSearcher(unittest.TestCase):
    """FlexAWQSSZ searcher keeps beta fixed at 0."""

    def test_init_stores_qconfig_and_step(self):
        qconfig = MagicMock()
        searcher = FlexAWQSSZAlphaBetaSearcher(qconfig=qconfig, search_step=0.1)
        self.assertIs(searcher.qconfig, qconfig)
        self.assertEqual(searcher.search_step, 0.1)

    def test_search_beta_returns_zero_beta_and_mse(self):
        searcher = FlexAWQSSZAlphaBetaSearcher(qconfig=MagicMock(), search_step=0.05)
        linear = MagicMock()
        with patch.object(searcher, "evaluate_alpha_beta", return_value=0.123) as mock_eval:
            best_beta, best_mse = searcher.search_beta(torch.randn(2, 3), linear, alpha=0.25)
        self.assertEqual(best_beta, 0.0)
        self.assertAlmostEqual(best_mse, 0.123, places=6)
        mock_eval.assert_called_once()
        self.assertEqual(mock_eval.call_args.kwargs.get("beta", 0.0), 0.0)


class TestBaseAlphaBetaSearcher(unittest.TestCase):
    """Base class is abstract and declares the search contract."""

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            BaseAlphaBetaSearcher()

    def test_abstract_methods(self):
        for name in ("search_alpha", "search_beta", "evaluate_alpha_beta"):
            self.assertIn(name, BaseAlphaBetaSearcher.__abstractmethods__)


if __name__ == "__main__":
    unittest.main()
