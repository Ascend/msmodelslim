#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest

from msmodelslim.processor.trainable_linear_quant.config import BlockTrainConfig
from msmodelslim.processor.trainable_linear_quant.config.train_config import (
    EmaSelectBest,
    LastSelectBest,
    MinLossSelectBest,
)


class TestBlockTrainConfigResolved:
    def test_lr_default(self):
        cfg = BlockTrainConfig()
        assert cfg.lr == pytest.approx(0.01)

    def test_lr_explicit(self):
        cfg = BlockTrainConfig(lr=0.02)
        assert cfg.lr == 0.02

    def test_lr_iters_zero_unchanged(self):
        cfg = BlockTrainConfig(iters=0)
        assert cfg.lr == pytest.approx(0.01)

    def test_negative_iters_raises(self):
        with pytest.raises(Exception):
            BlockTrainConfig(iters=-1)

    def test_advanced_defaults(self):
        cfg = BlockTrainConfig()
        assert isinstance(cfg.select_best, EmaSelectBest)
        assert cfg.select_best.mode == "ema"
        assert cfg.select_best.ema_beta == 0.7
        assert cfg.select_best.ema_window_size == 5
        assert cfg.select_best.early_stop_patience == -1
        assert cfg.train_seed == 42
        assert "loss_scale" not in BlockTrainConfig.model_fields

    def test_select_best_by_mode_classes(self):
        cfg = BlockTrainConfig(select_best=MinLossSelectBest(early_stop_patience=3))
        assert isinstance(cfg.select_best, MinLossSelectBest)
        assert cfg.select_best.early_stop_patience == 3

        cfg_last = BlockTrainConfig(select_best=LastSelectBest())
        assert isinstance(cfg_last.select_best, LastSelectBest)
        assert not hasattr(cfg_last.select_best, "early_stop_patience")

    def test_last_rejects_early_stop_field(self):
        with pytest.raises(Exception):
            BlockTrainConfig(select_best={"mode": "last", "early_stop_patience": 2})

    def test_min_loss_rejects_ema_fields(self):
        with pytest.raises(Exception):
            BlockTrainConfig(select_best={"mode": "min_loss", "ema_beta": 0.8})

    def test_invalid_lr_raises(self):
        with pytest.raises(Exception):
            BlockTrainConfig(lr=0)

    def test_invalid_ema_beta_raises(self):
        with pytest.raises(Exception):
            BlockTrainConfig(select_best={"mode": "ema", "ema_beta": 1.5})
