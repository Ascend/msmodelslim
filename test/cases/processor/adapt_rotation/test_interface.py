#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

from msmodelslim.processor.adapt_rotation.interface import AdaptRotationInterface
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import QuaRotInterface


class _ConcreteAdaptRotation(AdaptRotationInterface):
    """Minimal concrete adapter exercising the abstract contract."""

    def get_hidden_dim(self) -> int:
        return 128

    def get_ln_fuse_map(self):
        return {"norm": ["q_proj"]}, {"norm": ["k_proj"]}

    def get_bake_names(self):
        return ["down_proj"], ["down_proj"]

    def get_rotate_map(self, block_size):
        return [], []


class TestAdaptRotationInterface(unittest.TestCase):
    def test_interface_extends_quarot_interface(self):
        self.assertTrue(issubclass(AdaptRotationInterface, QuaRotInterface))

    def test_base_get_hidden_dim_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            AdaptRotationInterface().get_hidden_dim()

    def test_concrete_subclass_exposes_hidden_dim(self):
        adapter = _ConcreteAdaptRotation()
        self.assertEqual(adapter.get_hidden_dim(), 128)

    def test_get_ln_fuse_map_returns_expected_tuple_when_concrete_adapter(self):
        adapter = _ConcreteAdaptRotation()
        self.assertEqual(adapter.get_ln_fuse_map(), ({"norm": ["q_proj"]}, {"norm": ["k_proj"]}))

    def test_get_bake_names_returns_expected_tuple_when_concrete_adapter(self):
        adapter = _ConcreteAdaptRotation()
        self.assertEqual(adapter.get_bake_names(), (["down_proj"], ["down_proj"]))

    def test_get_rotate_map_returns_empty_pair_when_block_size_is_minus_one(self):
        adapter = _ConcreteAdaptRotation()
        self.assertEqual(adapter.get_rotate_map(block_size=-1), ([], []))


if __name__ == "__main__":
    unittest.main()
