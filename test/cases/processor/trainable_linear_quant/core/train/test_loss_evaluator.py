#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

import torch
from torch import nn

from msmodelslim.processor.trainable_linear_quant.core.train.loss_evaluator import (
    BlockLossEvaluator,
    iter_tlq_accumulate_slots,
    resolve_device_type,
    to_device,
)
from msmodelslim.processor.trainable_linear_quant.data.block_data import (
    DefaultTLQBlockDataInterface,
)


class ScaleBlock(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class TestIterTlqAccumulateSlots(unittest.TestCase):
    def _sample(self, value: float):
        return ((torch.tensor([[value]]),), {})

    def _teacher(self, value: float):
        return torch.tensor([[value * 10.0]])

    def test_single_index_per_slot(self):
        datas = [self._sample(1.0), self._sample(2.0)]
        teachers = [self._teacher(1.0), self._teacher(2.0)]
        indices = torch.tensor([1, 0])
        slots = list(
            iter_tlq_accumulate_slots(
                datas,
                teachers,
                indices,
                batch_size=1,
                gradient_accumulate_steps=2,
            )
        )
        self.assertEqual(len(slots), 2)
        self.assertIs(slots[0][0][1], teachers[1])
        self.assertIs(slots[1][0][1], teachers[0])

    def test_multi_index_slot_aligns_each_teacher(self):
        datas = [self._sample(float(i)) for i in range(4)]
        teachers = [self._teacher(float(i)) for i in range(4)]
        indices = torch.tensor([2, 3])
        slots = list(
            iter_tlq_accumulate_slots(
                datas,
                teachers,
                indices,
                batch_size=2,
                gradient_accumulate_steps=1,
            )
        )
        self.assertEqual(len(slots), 1)
        self.assertEqual(len(slots[0]), 2)
        self.assertIs(slots[0][0][1], teachers[2])
        self.assertIs(slots[0][1][1], teachers[3])


class TestBlockLossEvaluator(unittest.TestCase):
    def test_eval_and_backward_uses_matching_teacher(self):
        block_data = DefaultTLQBlockDataInterface()
        evaluator = BlockLossEvaluator(block_data, loss_scale=1.0, gradient_accumulate_steps=1)
        block = ScaleBlock(scale=2.0)

        datas = [((torch.tensor([[1.0]]),), {})]
        teachers = [torch.tensor([[2.0]])]
        indices = torch.tensor([0])

        loss = evaluator.eval_and_backward(block, datas, teachers, "cpu", indices)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_mismatched_teacher_produces_nonzero_loss(self):
        block_data = DefaultTLQBlockDataInterface()
        evaluator = BlockLossEvaluator(block_data, loss_scale=1.0, gradient_accumulate_steps=1)
        block = ScaleBlock(scale=2.0)

        datas = [((torch.tensor([[1.0]]),), {})]
        teachers = [torch.tensor([[10.0]])]
        indices = torch.tensor([0])

        loss = evaluator.eval_and_backward(block, datas, teachers, "cpu", indices)
        self.assertGreater(loss, 0.0)


class TestDeviceUtils(unittest.TestCase):
    def test_resolve_device_type_with_index(self):
        # UT 不依赖 NPU；用通用 device 字符串验证带 index 的解析
        self.assertEqual(resolve_device_type("cpu"), "cpu")
        self.assertEqual(resolve_device_type("cuda:0"), "cuda")

    def test_to_device_nested_dict(self):
        x = torch.randn(2)
        out = to_device({"a": x}, "cpu")
        self.assertTrue(torch.equal(out["a"], x))


if __name__ == "__main__":
    unittest.main()
