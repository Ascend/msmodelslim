#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

import torch

from msmodelslim.processor.analysis.binary_operator_model_wise.metrics.mse_model_wise import (
    MSEModelWiseAnalysisInterface,
    MSEModelWiseAnalysisMethod,
)
from msmodelslim.utils.exception import UnsupportedError


class TestMSEModelWiseAnalysisMethod(unittest.TestCase):
    def setUp(self):
        self.method = MSEModelWiseAnalysisMethod()

    def test_name_and_supports_distributed(self):
        self.assertEqual(self.method.name, "mse_model_wise")
        self.assertTrue(self.method.supports_distributed)

    def test_compute_score_returns_zero_when_no_valid_pairs(self):
        self.assertEqual(self.method.compute_score([], []), 0.0)
        self.assertEqual(
            self.method.compute_score([{"bad": 1}], [torch.randn(2, 3)]),
            0.0,
        )

    def test_compute_score_from_tensor_outputs(self):
        ref = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        cand = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        score = self.method.compute_score([ref], [cand])
        self.assertAlmostEqual(score, 0.25, places=5)

    def test_compute_score_averages_multiple_pairs(self):
        ref_a = torch.ones(2, 2)
        cand_a = torch.ones(2, 2) * 2.0
        ref_b = torch.zeros(2, 2)
        cand_b = torch.zeros(2, 2)
        score = self.method.compute_score([ref_a, ref_b], [cand_a, cand_b])
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_to_tensor_falls_back_to_nested_tuple_on_unsupported_error(self):
        tensor = torch.randn(2, 3)
        result = self.method._to_tensor((tensor,))
        self.assertTrue(torch.equal(result, tensor))

    def test_to_tensor_returns_none_for_empty_nested_tuple(self):
        self.assertIsNone(self.method._to_tensor(()))
        self.assertIsNone(self.method._to_tensor([]))

    def test_to_tensor_returns_none_for_non_tensor_leaf(self):
        self.assertIsNone(self.method._to_tensor({"unsupported": 1}))

    def test_uses_adapter_extract_hidden_states(self):
        class FakeAdapter(MSEModelWiseAnalysisInterface):
            def extract_hidden_states(self, value):
                return value["h"]

        method = MSEModelWiseAnalysisMethod(adapter=FakeAdapter())
        ref = {"h": torch.ones(2, 2)}
        cand = {"h": torch.ones(2, 2) * 2.0}
        self.assertAlmostEqual(method.compute_score([ref], [cand]), 1.0, places=5)

    def test_to_tensor_returns_none_when_adapter_raises_unsupported(self):
        class RejectAdapter(MSEModelWiseAnalysisInterface):
            def extract_hidden_states(self, value):
                raise UnsupportedError("bad", action="fix")

        method = MSEModelWiseAnalysisMethod(adapter=RejectAdapter())
        self.assertIsNone(method._to_tensor({"x": 1}))
