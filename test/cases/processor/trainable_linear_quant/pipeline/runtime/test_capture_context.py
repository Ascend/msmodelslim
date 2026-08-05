#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import (
    BlockTLQContext,
    capture_float_teacher,
    capture_quant_propagation,
)


class TinyBlock(nn.Module):
    def forward(self, x):
        return x * 2


class TestBlockCaptureFunctions(unittest.TestCase):
    def test_capture_float_teacher_moves_outputs_to_cpu(self):
        block = TinyBlock()
        x = torch.randn(2, 3)
        request = BatchProcessRequest(
            name="block0",
            module=block,
            datas=[((x,), {})],
        )
        outputs = capture_float_teacher(request)
        self.assertEqual(len(outputs), 1)
        self.assertFalse(outputs[0].is_cuda)
        self.assertTrue(torch.equal(outputs[0], x * 2))

    def test_capture_quant_propagation_keeps_device(self):
        block = TinyBlock()
        x = torch.randn(2, 3)
        request = BatchProcessRequest(
            name="block0",
            module=block,
            datas=[((x,), {})],
        )
        outputs = capture_quant_propagation(request)
        self.assertIs(request.outputs, outputs)
        self.assertEqual(len(outputs), 1)

    def test_capture_float_teacher_clears_request_outputs(self):
        block = TinyBlock()
        request = BatchProcessRequest(
            name="block0",
            module=block,
            datas=[((torch.randn(1, 2),), {})],
        )
        outputs = capture_float_teacher(request)
        self.assertIsNone(request.outputs)
        self.assertEqual(len(outputs), 1)


class TestBlockTLQContext(unittest.TestCase):
    def test_require_ops_raises_when_empty(self):
        ctx = BlockTLQContext(block_name="layer.0")
        with self.assertRaises(RuntimeError):
            ctx.require_ops()

    def test_require_ops_returns_installed_ops(self):
        ctx = BlockTLQContext(block_name="layer.0")
        op = MagicMock()
        ctx.ops.append(op)
        self.assertEqual(ctx.require_ops(), [op])


if __name__ == "__main__":
    unittest.main()
