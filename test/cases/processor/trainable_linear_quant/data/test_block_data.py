#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

import torch
from torch import nn

from msmodelslim.processor.trainable_linear_quant.data.block_data import (
    DefaultTLQBlockDataInterface,
    TLQBlockDataInterface,
    propagate_outputs_to_inputs,
    resolve_tlq_block_data_interface,
)


class TinyBlock(nn.Module):
    def forward(self, x):
        return x * 2


class TestDefaultTLQBlockDataInterface(unittest.TestCase):
    def setUp(self):
        self.block_data = DefaultTLQBlockDataInterface()

    def test_inject_flat_args(self):
        hidden = torch.randn(2, 4)
        replacement = torch.randn(2, 4)
        sample = [[hidden.clone()], {}]
        self.block_data.inject_hidden_states(sample, replacement)
        self.assertTrue(torch.equal(sample[0][0], replacement))

    def test_inject_nested_args(self):
        hidden = torch.randn(2, 4)
        replacement = torch.randn(2, 4)
        sample = [[[hidden.clone()]], {}]
        self.block_data.inject_hidden_states(sample, replacement)
        self.assertTrue(torch.equal(sample[0][0][0], replacement))

    def test_propagate_outputs_to_inputs(self):
        src = torch.randn(2, 4)
        dst = torch.zeros(2, 4)
        sample = [[dst.clone()], {}]
        propagate_outputs_to_inputs(self.block_data, [sample], [src])
        self.assertTrue(torch.equal(sample[0][0], src))

    def test_extract_unsupported_output_guides_custom_interface(self):
        with self.assertRaises(Exception) as ctx:
            self.block_data.extract_hidden_states({"hidden": torch.randn(2, 4)})
        msg = str(ctx.exception)
        self.assertIn("inherit TLQBlockDataInterface", msg)
        self.assertIn("DefaultTLQBlockDataInterface", msg)

    def test_resolve_uses_adapter_when_it_implements_interface(self):
        class FakeAdapter(TLQBlockDataInterface):
            def extract_hidden_states(self, block_output):
                return block_output["h"]

            def inject_hidden_states(self, block_input, hidden):
                block_input[1]["h"] = hidden

            def get_loss_mask(self, block_input, hidden_states):
                return torch.ones(hidden_states.shape[:2], dtype=torch.bool)

        adapter = FakeAdapter()
        self.assertIs(resolve_tlq_block_data_interface(adapter), adapter)

    def test_resolve_falls_back_to_default(self):
        block_data = resolve_tlq_block_data_interface(object())
        self.assertIsInstance(block_data, DefaultTLQBlockDataInterface)

    def test_extract_hidden_states_from_tuple(self):
        tensor = torch.randn(2, 4)
        out = self.block_data.extract_hidden_states((tensor, None))
        self.assertTrue(torch.equal(out, tensor))

    def test_get_loss_mask_moves_attention_mask_to_hidden_device(self):
        hidden = torch.randn(2, 3, 4)
        # 2D mask on CPU while hidden is used as device reference
        attn = torch.ones(2, 3, dtype=torch.long)
        sample = ((hidden,), {"attention_mask": attn})
        mask = self.block_data.get_loss_mask(sample, hidden)
        self.assertEqual(mask.device, hidden.device)
        self.assertEqual(tuple(mask.shape), (2, 3))

    def test_get_loss_mask_ones_fallback_matches_hidden_device(self):
        hidden = torch.randn(2, 5, 4)
        sample = ((hidden,), {})
        mask = self.block_data.get_loss_mask(sample, hidden)
        self.assertEqual(mask.device, hidden.device)
        self.assertTrue(bool(mask.all()))
