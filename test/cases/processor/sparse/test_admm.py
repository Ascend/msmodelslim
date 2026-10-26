#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.processor.sparse.admm 模块的单元测试
"""

import numpy as np
import pytest
import torch
from torch import nn

from msmodelslim.processor.sparse.admm import AdmmPruner, quantize_clip, quantize_l2, split_half


class TestSplitHalf:
    def test_returns_mantissa_and_exponent_for_known_values(self):
        para = torch.tensor([1.0, 2.0, 0.5, 0.0], dtype=torch.float16)

        mant, exp = split_half(para)

        assert mant.tolist() == [1.0, 1.0, 1.0, 0.0]
        assert exp.tolist() == [0, 1, -1, 0]

    def test_accepts_numpy_array(self):
        para = np.array([1.0, 2.0, 0.5], dtype=np.float16)

        _, exp = split_half(para)

        assert exp.tolist() == [0, 1, -1]

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            split_half("not-a-tensor")


class TestQuantizeClip:
    def test_rounds_mantissa_on_power_of_two_grid(self):
        para = torch.tensor([[0.1, 1.0, 3.0, -2.0]], dtype=torch.float16)

        out = quantize_clip(para, threshold=2)

        assert out.tolist() == [[0.09375, 1.0, 3.0, -2.0]]


class TestQuantizeL2:
    def test_empty_keep_mask_matches_clip_result(self):
        para = torch.tensor([[0.1, 1.0, 3.0, -2.0]], dtype=torch.float16)
        keep_mask = torch.zeros_like(para, dtype=torch.bool)

        out = quantize_l2(keep_mask, para, threshold=2)

        assert out.tolist() == quantize_clip(para, threshold=2).tolist()

    def test_full_keep_mask_preserves_original_values(self):
        para = torch.tensor([[0.1]], dtype=torch.float16)
        keep_mask = torch.ones_like(para, dtype=torch.bool)

        out = quantize_l2(keep_mask, para, threshold=2)

        assert out.tolist() == para.tolist()


class TestAdmmPruner:
    def test_init_reads_layer_shape_and_zeroes_stats(self):
        pruner = AdmmPruner(nn.Linear(4, 3, bias=False))

        assert pruner.rows == 3
        assert pruner.columns == 4
        assert tuple(pruner.hessian.shape) == (4, 4)
        assert pruner.nsamples == 0
        assert tuple(pruner.scaler_row.shape) == (4,)

    def test_add_batch_accumulates_hessian_and_sample_count(self):
        pruner = AdmmPruner(nn.Linear(4, 3, bias=False))

        pruner.add_batch(torch.ones(1, 4))
        pruner.add_batch(torch.ones(1, 4))

        assert pruner.nsamples == 2
        assert torch.equal(pruner.hessian, torch.full((4, 4), 2.0))

    def test_free_releases_hessian(self):
        pruner = AdmmPruner(nn.Linear(2, 2, bias=False))

        pruner.free()

        assert pruner.hessian is None

    def test_fasterprune_rewrites_weight_in_place_and_releases_hessian(self):
        torch.manual_seed(0)
        layer = nn.Linear(8, 8, bias=False)
        pruner = AdmmPruner(layer)
        pruner.add_batch(torch.randn(16, 8))

        pruner.fasterprune(sparse_ratio=0.5)

        assert tuple(layer.weight.shape) == (8, 8)
        assert layer.weight.dtype == torch.float32
        assert pruner.hessian is None
