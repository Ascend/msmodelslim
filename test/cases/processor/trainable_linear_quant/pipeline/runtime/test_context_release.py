#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import BlockTLQContext


def _int8_qconfig() -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


class TestBlockTLQContextRelease(unittest.TestCase):
    def test_release_clears_training_state(self):
        op = MagicMock()
        op.target_modules = {"block0.fc": MagicMock()}
        ctx = BlockTLQContext(block_name="block0")
        ctx.ops = [op]
        ctx.wrappers_by_path = {"block0.fc": MagicMock()}
        ctx.teacher_outputs = [torch.randn(2, 4)]

        ctx.release()

        op.release_cached_params.assert_called_once()
        self.assertEqual(op.target_modules, {})
        self.assertEqual(ctx.ops, [])
        self.assertEqual(ctx.wrappers_by_path, {})
        self.assertEqual(ctx.teacher_outputs, [])
