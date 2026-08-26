#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from types import SimpleNamespace

import torch

from msmodelslim.processor.analysis.binary_operator_model_wise.metrics.mse_model_wise.block_data import (
    DefaultMSEModelWiseBlockData,
    resolve_mse_model_wise_block_data,
)
from msmodelslim.processor.analysis.binary_operator_model_wise.metrics.mse_model_wise.interface import (
    MSEModelWiseAnalysisInterface,
)
from msmodelslim.utils.exception import UnsupportedError


class TestDefaultMSEModelWiseBlockData(unittest.TestCase):
    def setUp(self):
        self.block_data = DefaultMSEModelWiseBlockData()

    def test_extract_hidden_states_from_dict_with_pooler_output(self):
        tensor = torch.randn(4, 5)
        out = self.block_data.extract_hidden_states({"pooler_output": tensor})
        self.assertTrue(torch.equal(out, tensor))

    def test_extract_hidden_states_from_dict_with_hidden_states(self):
        tensor = torch.randn(3, 4)
        out = self.block_data.extract_hidden_states({"hidden_states": tensor})
        self.assertTrue(torch.equal(out, tensor))

    def test_extract_hidden_states_from_model_output_object(self):
        tensor = torch.randn(2, 3)
        out = self.block_data.extract_hidden_states(SimpleNamespace(last_hidden_state=tensor))
        self.assertTrue(torch.equal(out, tensor))

    def test_extract_hidden_states_from_forward_row(self):
        hidden = torch.randn(2, 3)
        out = self.block_data.extract_hidden_states(((hidden,), {}))
        self.assertTrue(torch.equal(out, hidden))

    def test_extract_hidden_states_from_forward_row_nested_tuple(self):
        hidden = torch.randn(2, 3)
        out = self.block_data.extract_hidden_states((((hidden,),), {}))
        self.assertTrue(torch.equal(out, hidden))

    def test_extract_hidden_states_from_forward_row_raises_when_args_empty(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self.block_data.extract_hidden_states(((), {}))
        self.assertIn("empty block args", str(ctx.exception))

    def test_extract_hidden_states_from_forward_row_raises_for_unsupported_args(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self.block_data.extract_hidden_states((("not-a-tensor",), {}))
        self.assertIn("args[0] type str", str(ctx.exception))

    def test_extract_unsupported_output_guides_custom_interface(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self.block_data.extract_hidden_states({"hidden": torch.randn(2, 4)})
        msg = str(ctx.exception)
        self.assertIn("inherit MSEModelWiseAnalysisInterface", msg)
        self.assertIn("DefaultMSEModelWiseBlockData", msg)

    def test_extract_hidden_states_from_empty_tuple_raises(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self.block_data.extract_hidden_states(())
        self.assertIn("empty tuple block output", str(ctx.exception))

    def test_extract_hidden_states_from_tuple_with_nested_dict(self):
        tensor = torch.randn(2, 4)
        out = self.block_data.extract_hidden_states(({"hidden_states": tensor},))
        self.assertTrue(torch.equal(out, tensor))

    def test_extract_hidden_states_from_tuple_raises_for_non_tensor_first(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self.block_data.extract_hidden_states(("bad", None))
        self.assertIn("first element of block output", str(ctx.exception))

    def test_extract_hidden_states_from_raw_tensor(self):
        tensor = torch.randn(2, 4)
        out = self.block_data.extract_hidden_states(tensor)
        self.assertTrue(torch.equal(out, tensor))

    def test_extract_hidden_states_from_tuple(self):
        tensor = torch.randn(2, 4)
        out = self.block_data.extract_hidden_states((tensor, None))
        self.assertTrue(torch.equal(out, tensor))

    def test_resolve_uses_adapter_when_it_implements_interface(self):
        class FakeAdapter(MSEModelWiseAnalysisInterface):
            def extract_hidden_states(self, value):
                return value["h"]

        adapter = FakeAdapter()
        self.assertIs(resolve_mse_model_wise_block_data(adapter), adapter)

    def test_resolve_falls_back_to_default(self):
        block_data = resolve_mse_model_wise_block_data(object())
        self.assertIsInstance(block_data, DefaultMSEModelWiseBlockData)
