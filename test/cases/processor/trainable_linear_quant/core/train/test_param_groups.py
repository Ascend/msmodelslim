#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.config import (
    BlockTrainConfig,
)
from msmodelslim.processor.trainable_linear_quant.core.train import (
    TrainableLinearQuantBlockTrainer,
)
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


class TestBuildOptimizerParamGroups(unittest.TestCase):
    def test_deduplicates_shared_parameters(self):
        shared = torch.nn.Parameter(torch.ones(1))
        op1 = MagicMock()
        op1.config.lr = None
        op1.train_params = {"a": shared}
        op2 = MagicMock()
        op2.config.lr = None
        op2.train_params = {"b": shared}
        config = BlockTrainConfig(iters=10, lr=0.1)
        trainer = TrainableLinearQuantBlockTrainer(config, MagicMock())
        groups = trainer._build_optimizer_param_groups([op1, op2])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["params"]), 1)

    def test_scales_per_op_learning_rate(self):
        p = torch.nn.Parameter(torch.ones(1))
        op = MagicMock()
        op.config.lr = 0.2
        op.train_params = {"w": p}
        config = BlockTrainConfig(iters=10, lr=0.1)
        trainer = TrainableLinearQuantBlockTrainer(config, MagicMock())
        groups = trainer._build_optimizer_param_groups([op])
        self.assertEqual(groups[0]["lr"], 0.2)


class TestTrainerAssembly(unittest.TestCase):
    def test_trainer_and_param_groups_from_block_ctx(self):
        train_config = BlockTrainConfig(iters=1)
        block_data = MagicMock()

        p = torch.nn.Parameter(torch.ones(1))
        op = MagicMock()
        op.config.lr = None
        op.train_params = {"w": p}
        ctx = BlockTLQContext(block_name="block0")
        ctx.ops = [op]

        ops = ctx.require_ops()
        trainer = TrainableLinearQuantBlockTrainer(train_config, block_data)
        groups = trainer._build_optimizer_param_groups(ops)
        self.assertEqual(len(groups), 1)
        self.assertIsInstance(trainer, TrainableLinearQuantBlockTrainer)
        self.assertEqual(ops, [op])
