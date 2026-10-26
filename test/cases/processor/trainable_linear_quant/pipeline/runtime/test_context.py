#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for BlockTLQContext: op requirement and post-finalize release."""

import unittest
from unittest.mock import MagicMock

import torch

from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import BlockTLQContext


class TestBlockTLQContext(unittest.TestCase):
    def test_defaults(self):
        ctx = BlockTLQContext(block_name="layer.0")

        self.assertEqual(ctx.block_name, "layer.0")
        self.assertEqual(ctx.device.type, "cpu")
        self.assertEqual(ctx.teacher_outputs, [])
        self.assertIsNone(ctx.train_result)
        self.assertEqual(ctx.wrappers_by_path, {})
        self.assertEqual(ctx.ops, [])

    def test_require_ops_raises_when_empty(self):
        ctx = BlockTLQContext(block_name="layer.0")

        with self.assertRaises(RuntimeError) as err:
            ctx.require_ops()

        self.assertIn("layer.0", str(err.exception))
        self.assertIn("has no TLQ ops installed", str(err.exception))

    def test_require_ops_returns_installed_ops(self):
        ctx = BlockTLQContext(block_name="layer.0")
        op = MagicMock()
        ctx.ops.append(op)

        self.assertEqual(ctx.require_ops(), [op])

    def test_release_clears_training_state(self):
        op = MagicMock()
        op.target_modules = {"block0.fc": MagicMock()}
        ctx = BlockTLQContext(block_name="block0")
        ctx.ops = [op]
        ctx.wrappers_by_path = {"block0.fc": MagicMock()}
        ctx.teacher_outputs = [torch.randn(2, 4)]
        ctx.train_result = MagicMock()

        ctx.release()

        op.release_cached_params.assert_called_once_with()
        self.assertEqual(op.target_modules, {})
        self.assertEqual(ctx.ops, [])
        self.assertEqual(ctx.wrappers_by_path, {})
        self.assertEqual(ctx.teacher_outputs, [])
        self.assertIsNone(ctx.train_result)


if __name__ == "__main__":
    unittest.main()
