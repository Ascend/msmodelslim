#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

import torch

from msmodelslim.processor.trainable_linear_quant.core.train.loss_functions import (
    custom_outlier_loss,
    get_elementwise_loss_fn,
    l1_elementwise_loss,
    loss_tensor_pairs,
    masked_mean_loss,
)
from msmodelslim.processor.trainable_linear_quant.data.block_data import (
    DefaultTLQBlockDataInterface,
)


class TestLossFunctions(unittest.TestCase):
    def test_l1_and_custom_outlier(self):
        output = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)
        l1 = l1_elementwise_loss(output, target)
        self.assertEqual(l1.shape, output.shape)
        outlier = custom_outlier_loss(output, target, l1_elementwise_loss)
        self.assertEqual(outlier.shape, output.shape)

    def test_masked_mean_loss(self):
        loss = torch.ones(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
        value = masked_mean_loss(loss, mask)
        self.assertTrue(torch.isfinite(value))

    def test_get_elementwise_loss_fn(self):
        self.assertIs(get_elementwise_loss_fn("l1"), l1_elementwise_loss)
        custom = get_elementwise_loss_fn("custom_outlier")
        out = custom(torch.ones(1, 2, 3), torch.zeros(1, 2, 3))
        self.assertEqual(out.shape, (1, 2, 3))
        with self.assertRaises(ValueError):
            get_elementwise_loss_fn("unknown")

    def test_loss_tensor_pairs_tuple_and_tensor(self):
        handler = DefaultTLQBlockDataInterface()
        q = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))
        f = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))
        pairs = loss_tensor_pairs(handler, q, f)
        self.assertEqual(len(pairs), 2)

        t_q = torch.randn(1, 2, 3)
        t_f = torch.randn(1, 2, 3)
        pairs = loss_tensor_pairs(handler, t_q, t_f)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(torch.equal(pairs[0][0], t_q))

    def test_loss_tensor_pairs_type_mismatches(self):
        handler = DefaultTLQBlockDataInterface()
        with self.assertRaises(TypeError):
            loss_tensor_pairs(handler, (torch.ones(1),), torch.ones(1))
        with self.assertRaises(TypeError):
            loss_tensor_pairs(handler, torch.ones(1), (torch.ones(1),))
        with self.assertRaises(ValueError):
            loss_tensor_pairs(handler, (torch.ones(1),), (torch.ones(1), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
