#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest
import torch

from msmodelslim.processor.paretoq.core import (
    WEIGHT_CLIP_INIT_AMAX,
    WEIGHT_CLIP_INIT_MEAN2_SQRT_QP,
    fake_quant_weight,
    init_weight_clip_from_weight,
    paretoq_weight_qp,
)


def test_paretoq_weight_qp_returns_one_when_binary_or_fp16():
    assert paretoq_weight_qp(1) == 1
    assert paretoq_weight_qp(0) == 1
    assert paretoq_weight_qp(16) == 1


def test_paretoq_weight_qp_matches_lsq_upper_when_w4():
    assert paretoq_weight_qp(4) == 7


def test_fake_quant_weight_returns_same_tensor_when_w_bits_ge_16():
    weight = torch.randn(4, 8)
    alpha = torch.ones(4, 1)
    out = fake_quant_weight(weight, alpha, w_bits=16, weight_layerwise=False)
    assert out is weight


def test_fake_quant_weight_keeps_shape_when_w4_lsq():
    weight = torch.randn(3, 5)
    alpha = init_weight_clip_from_weight(weight, w_bits=4, init_method=WEIGHT_CLIP_INIT_AMAX)
    out = fake_quant_weight(weight, alpha, w_bits=4, weight_layerwise=False)
    assert out.shape == weight.shape
    assert out.dtype == weight.dtype


def test_fake_quant_weight_keeps_shape_when_w2_seq():
    weight = torch.randn(2, 4)
    alpha = init_weight_clip_from_weight(weight, w_bits=2, init_method=WEIGHT_CLIP_INIT_AMAX)
    out = fake_quant_weight(weight, alpha, w_bits=2, weight_layerwise=False)
    assert out.shape == weight.shape


def test_fake_quant_weight_raises_when_w_bits_unsupported():
    weight = torch.randn(2, 2)
    alpha = torch.ones(2, 1)
    with pytest.raises(NotImplementedError, match="w_bits=8"):
        fake_quant_weight(weight, alpha, w_bits=8, weight_layerwise=False)


def test_init_weight_clip_from_weight_raises_when_method_unknown():
    weight = torch.randn(2, 3)
    with pytest.raises(ValueError, match="Unknown weight_clip"):
        init_weight_clip_from_weight(weight, w_bits=4, init_method="bad")


@pytest.mark.parametrize("method", [WEIGHT_CLIP_INIT_AMAX, WEIGHT_CLIP_INIT_MEAN2_SQRT_QP])
def test_init_weight_clip_from_weight_returns_per_row_when_2d(method):
    weight = torch.randn(5, 7)
    clip = init_weight_clip_from_weight(weight, w_bits=4, init_method=method)
    assert clip.shape == (5, 1)
    assert torch.all(clip > 0)
