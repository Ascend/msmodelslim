#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Contract tests for trainable_linear_quant interfaces."""

# pylint: disable=abstract-class-instantiated

import inspect
import unittest
from typing import List, get_type_hints

import torch

from msmodelslim.core.graph.adapter_types import AdapterConfig
from msmodelslim.processor.trainable_linear_quant.data import TLQBlockDataInterface
from msmodelslim.processor.trainable_linear_quant.interface import (
    TLQSubgraphAdapter,
)
from msmodelslim.processor.trainable_linear_quant import interface as interface_module


class _Adapter:
    def get_adapter_config_for_subgraph(self):
        return []


class TestTLQSubgraphAdapter(unittest.TestCase):
    def test_protocol_method_signature(self):
        hints = get_type_hints(TLQSubgraphAdapter.get_adapter_config_for_subgraph)
        self.assertEqual(hints["return"], List[AdapterConfig])

    def test_runtime_structural_usage(self):
        adapter: TLQSubgraphAdapter = _Adapter()
        self.assertEqual(adapter.get_adapter_config_for_subgraph(), [])


class TestTLQBlockDataInterfaceExport(unittest.TestCase):
    def test_interface_is_abstract(self):
        self.assertTrue(inspect.isabstract(TLQBlockDataInterface))
        with self.assertRaises(TypeError):
            TLQBlockDataInterface()

    def test_interface_requires_three_tlq_contract_methods(self):
        names = set(TLQBlockDataInterface.__abstractmethods__)
        self.assertEqual(names, {"extract_hidden_states", "inject_hidden_states", "get_loss_mask"})
        self.assertIs(get_type_hints(TLQBlockDataInterface.extract_hidden_states)["return"], torch.Tensor)

    def test_public_exports_are_stable(self):
        self.assertEqual(
            interface_module.__all__,
            ["TLQBlockDataInterface", "TLQSubgraphAdapter"],
        )


if __name__ == "__main__":
    unittest.main()
