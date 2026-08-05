#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""TLQ SignSGD slim implementation smoke tests."""

import unittest

import torch

from msmodelslim.processor.trainable_linear_quant.core.train.sign_sgd import SignSGD


class TestTLQSignSGD(unittest.TestCase):
    def test_sign_update(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -1.0, 0.5]))
        opt = SignSGD([p], lr=0.1)
        p.grad = torch.tensor([2.0, -3.0, 0.0])
        opt.step()
        # Δ = -lr * sign(g) → [-0.1, +0.1, 0]
        self.assertTrue(torch.allclose(p.data, torch.tensor([0.9, -0.9, 0.5])))

    def test_param_group_lr(self):
        p1 = torch.nn.Parameter(torch.tensor([1.0]))
        p2 = torch.nn.Parameter(torch.tensor([1.0]))
        opt = SignSGD(
            [{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.2}],
            lr=0.05,
        )
        p1.grad = torch.tensor([1.0])
        p2.grad = torch.tensor([1.0])
        opt.step()
        self.assertTrue(torch.allclose(p1.data, torch.tensor([0.9])))
        self.assertTrue(torch.allclose(p2.data, torch.tensor([0.8])))


if __name__ == "__main__":
    unittest.main()
