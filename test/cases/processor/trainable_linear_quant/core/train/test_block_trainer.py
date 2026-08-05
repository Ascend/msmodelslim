#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.processor.trainable_linear_quant.config import BlockTrainConfig
from msmodelslim.processor.trainable_linear_quant.config.train_config import (
    EmaSelectBest,
    LastSelectBest,
    MinLossSelectBest,
)
from msmodelslim.processor.trainable_linear_quant.core.train import (
    TrainableLinearQuantBlockTrainer,
)
from msmodelslim.processor.trainable_linear_quant.core.train.loss_evaluator import (
    BlockLossEvaluator,
)
from msmodelslim.processor.trainable_linear_quant.data.block_data import (
    DefaultTLQBlockDataInterface,
)


class TinyBlock(nn.Module):
    def forward(self, x):
        return x


def _make_trainer(**overrides) -> TrainableLinearQuantBlockTrainer:
    cfg = BlockTrainConfig(**overrides)
    return TrainableLinearQuantBlockTrainer(cfg, DefaultTLQBlockDataInterface())


def _mock_op() -> MagicMock:
    op = MagicMock()
    op.save_best_params = MagicMock()
    op.train_params = {"value": nn.Parameter(torch.ones(1))}
    op.best_params = None
    op.op_id = "layer.round_tune"
    op.config.type = "round_tune"
    op.config.lr = None
    return op


class TestTrainableLinearQuantBlockTrainerInit(unittest.TestCase):
    def test_config_kwargs_set_advanced_fields(self):
        trainer = _make_trainer(
            lr=0.02,
            select_best=EmaSelectBest(ema_window_size=3),
        )
        self.assertEqual(trainer.config.select_best.ema_window_size, 3)
        self.assertEqual(trainer.config.lr, 0.02)

    def test_lr_default_from_config(self):
        trainer = _make_trainer()
        self.assertAlmostEqual(trainer.config.lr, 0.01)

    def test_lr_iters_zero_unchanged(self):
        trainer = _make_trainer(iters=0)
        self.assertEqual(trainer.config.lr, 0.01)


class TestTrainableLinearQuantBlockTrainerCheckpoint(unittest.TestCase):
    def setUp(self):
        self.block = TinyBlock()
        self.datas = [((torch.randn(2, 4),), {})]
        self.teachers = [torch.randn(2, 4)]
        self.op = _mock_op()
        self.block_name = "block0"

    def test_iters_zero_saves_init_params_without_optimizer(self):
        trainer = _make_trainer(iters=0)
        with patch.object(TrainableLinearQuantBlockTrainer, "get_optimizer") as mock_opt:
            result = trainer.train_block(
                block=self.block,
                all_datas=self.datas,
                float_output=self.teachers,
                device="cpu",
                tlq_ops=[self.op],
                block_name=self.block_name,
            )
            mock_opt.assert_not_called()
        self.op.save_best_params.assert_called_once()
        self.assertEqual(result.completed_iters, 0)

    @patch.object(BlockLossEvaluator, "eval_and_backward", return_value=1.0)
    def test_save_last_iter_saves_on_first_and_last_iter(self, _mock_eval):
        trainer = _make_trainer(
            iters=3,
            select_best=LastSelectBest(),
        )
        result = trainer.train_block(
            block=self.block,
            all_datas=self.datas,
            float_output=self.teachers,
            device="cpu",
            tlq_ops=[self.op],
            block_name=self.block_name,
        )
        self.assertEqual(self.op.save_best_params.call_count, 2)
        self.assertGreater(result.completed_iters, 0)
        self.assertEqual(result.best_iter, 2)

    @patch.object(BlockLossEvaluator, "eval_and_backward")
    def test_min_loss_mode_saves_when_loss_improves(self, mock_eval):
        mock_eval.side_effect = [2.0, 1.5, 1.0]
        trainer = _make_trainer(
            iters=3,
            select_best=MinLossSelectBest(),
            gradient_accumulate_steps=1,
        )
        result = trainer.train_block(
            block=self.block,
            all_datas=self.datas,
            float_output=self.teachers,
            device="cpu",
            tlq_ops=[self.op],
            block_name=self.block_name,
        )
        # iter0 always save + iter1 and iter2 improvements
        self.assertGreaterEqual(self.op.save_best_params.call_count, 2)
        self.assertEqual(result.completed_iters, 3)

    @patch.object(BlockLossEvaluator, "eval_and_backward", return_value=0.5)
    def test_early_stop_patience_early_stops(self, _mock_eval):
        trainer = _make_trainer(
            iters=20,
            select_best=MinLossSelectBest(early_stop_patience=2),
        )
        with patch.object(trainer, "step") as mock_step:
            result = trainer.train_block(
                block=self.block,
                all_datas=self.datas,
                float_output=self.teachers,
                device="cpu",
                tlq_ops=[self.op],
                block_name=self.block_name,
            )
            # iter0 save only; no improvement -> stop after gap at iter 2 (3 steps: 0,1,2)
            self.assertLessEqual(mock_step.call_count, 3)
            self.assertLess(result.completed_iters, 20)


class TestBuildOptimizerParamGroupsExtended(unittest.TestCase):
    def test_per_op_lr_in_param_group(self):
        p = nn.Parameter(torch.ones(1))
        op = MagicMock()
        op.config.lr = 0.02
        op.train_params = {"w": p}
        config = BlockTrainConfig(iters=10, lr=0.1)
        trainer = TrainableLinearQuantBlockTrainer(config, DefaultTLQBlockDataInterface())
        groups = trainer._build_optimizer_param_groups([op])
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(groups[0]["lr"], 0.02)

    def test_default_group_omits_lr(self):
        p = nn.Parameter(torch.ones(1))
        op = MagicMock()
        op.config.lr = None
        op.train_params = {"w": p}
        config = BlockTrainConfig(iters=10, lr=0.1)
        trainer = TrainableLinearQuantBlockTrainer(config, DefaultTLQBlockDataInterface())
        groups = trainer._build_optimizer_param_groups([op])
        self.assertNotIn("lr", groups[0])


if __name__ == "__main__":
    unittest.main()
