#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest.mock import MagicMock

from msmodelslim.processor.trainable_linear_quant.config import BlockTrainConfig
from msmodelslim.processor.trainable_linear_quant.config.train_config import (
    EmaSelectBest,
    LastSelectBest,
    MinLossSelectBest,
)
from msmodelslim.processor.trainable_linear_quant.core.train.checkpoint_selector import (
    CheckpointSelector,
    EmaCheckpointSelector,
    LastIterSelector,
    MinLossCheckpointSelector,
    save_best_params,
)


class TestCheckpointSelectorFromConfig(unittest.TestCase):
    def test_last_iter_mode(self):
        cfg = BlockTrainConfig(select_best=LastSelectBest())
        sel = CheckpointSelector.from_config(cfg)
        self.assertIsInstance(sel, LastIterSelector)

    def test_ema_mode(self):
        cfg = BlockTrainConfig(select_best=EmaSelectBest())
        sel = CheckpointSelector.from_config(cfg)
        self.assertIsInstance(sel, EmaCheckpointSelector)

    def test_min_loss_mode(self):
        cfg = BlockTrainConfig(select_best=MinLossSelectBest())
        sel = CheckpointSelector.from_config(cfg)
        self.assertIsInstance(sel, MinLossCheckpointSelector)


class TestEmaCheckpointSelector(unittest.TestCase):
    def setUp(self):
        self.selector = EmaCheckpointSelector(
            iters=10,
            ema_beta=0.7,
            ema_window_size=3,
            early_stop_patience=-1,
        )

    def test_iter_zero_always_saves(self):
        should_save, should_stop = self.selector.select(0, 2.0)
        self.assertTrue(should_save)
        self.assertFalse(should_stop)
        self.assertEqual(self.selector.last_saved_iter, 0)

    def test_improving_avg_loss_triggers_save(self):
        self.selector.select(0, 2.0)
        should_save, _ = self.selector.select(1, 1.0)
        self.assertTrue(should_save)
        self.assertEqual(self.selector.last_saved_iter, 1)

    def test_flat_loss_does_not_save_after_iter_zero(self):
        self.selector.select(0, 1.0)
        should_save, _ = self.selector.select(1, 1.0)
        self.assertFalse(should_save)


class TestMinLossCheckpointSelector(unittest.TestCase):
    def setUp(self):
        self.selector = MinLossCheckpointSelector(
            iters=5,
            early_stop_patience=2,
        )

    def test_saves_when_mean_loss_improves(self):
        self.selector.select(0, 2.0)
        should_save, _ = self.selector.select(1, 1.0)
        self.assertTrue(should_save)
        self.assertEqual(self.selector.last_saved_iter, 1)

    def test_early_stop_when_no_improvement(self):
        self.selector.select(0, 1.0)
        should_save, _ = self.selector.select(1, 1.0)
        self.assertFalse(should_save)
        _, should_stop = self.selector.select(2, 1.0)
        self.assertTrue(should_stop)


class TestLastIterSelector(unittest.TestCase):
    def test_saves_on_iter_zero_and_before_final_step(self):
        sel = LastIterSelector(iters=3)
        should_save, _ = sel.select(0, 1.0)
        self.assertTrue(should_save)
        should_save, _ = sel.select(1, 1.0)
        self.assertFalse(should_save)
        should_save, _ = sel.select(2, 0.8)
        self.assertTrue(should_save)
        self.assertEqual(sel.last_saved_iter, 2)

    def test_get_result_uses_last_iter(self):
        sel = LastIterSelector(iters=4)
        sel.select(0, 1.5)
        sel.select(3, 0.8)
        result = sel.get_result(last_mean_loss=0.8)
        self.assertEqual(result.best_iter, 3)
        self.assertEqual(result.best_loss, 0.8)


class TestSaveBestParams(unittest.TestCase):
    def test_calls_each_op(self):
        op1 = MagicMock()
        op2 = MagicMock()
        save_best_params([op1, op2])
        op1.save_best_params.assert_called_once()
        op2.save_best_params.assert_called_once()


if __name__ == "__main__":
    unittest.main()
