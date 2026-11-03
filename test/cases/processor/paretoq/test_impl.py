#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import pytest
import torch
from torch import nn

from msmodelslim.app.train.qat.qat_linear_algorithm import QATLinearAlgorithm
from msmodelslim.processor.paretoq.config import ParetoQConfig
from msmodelslim.processor.paretoq.convert import (
    ParetoQToW4A4DynamicProcessor,
    ParetoQToW4A8DynamicProcessor,
)
from msmodelslim.processor.paretoq.impl import ParetoQFakeQuantLinear, ParetoQImpl


def test_ParetoQFakeQuantLinear_forward_returns_2d_when_fp16_passthrough():
    module = ParetoQFakeQuantLinear(4, 3, bias=False, w_bits=16)
    x = torch.randn(2, 4)
    y = module(x)
    assert y.shape == (2, 3)


def test_ParetoQFakeQuantLinear_forward_applies_fake_quant_when_w4():
    weight = torch.randn(3, 4)
    alpha = torch.ones(3, 1)
    module = ParetoQFakeQuantLinear(4, 3, bias=False, w_bits=4, weight_clip_val=alpha)
    with torch.no_grad():
        module.weight.copy_(weight)
    x = torch.randn(2, 4)
    y = module(x)
    assert y.shape == (2, 3)


def test_ParetoQFakeQuantLinear_from_qat_tensors_raises_when_w_bits_fp16():
    with pytest.raises(ValueError, match="w_bits < 16"):
        ParetoQFakeQuantLinear.from_qat_tensors(
            torch.randn(2, 3),
            torch.ones(2, 1),
            w_bits=16,
        )


def test_ParetoQFakeQuantLinear_from_qat_tensors_builds_shell_when_w4():
    weight = torch.randn(2, 3)
    alpha = torch.ones(2, 1)
    module = ParetoQFakeQuantLinear.from_qat_tensors(weight, alpha, w_bits=4, deploy_act="a8")
    assert module.w_bits == 4
    assert torch.allclose(module.weight, weight)
    assert torch.allclose(module.weight_clip_val, alpha)


def test_ParetoQFakeQuantLinear_deploy_raises_when_fp16_shell():
    module = ParetoQFakeQuantLinear(2, 2, w_bits=16)
    with pytest.raises(ValueError, match="Cannot deploy"):
        module.deploy()


def test_ParetoQImpl_is_qat_linear_algorithm_when_defined():
    assert issubclass(ParetoQImpl, QATLinearAlgorithm)


def test_ParetoQImpl_qparam_names_returns_weight_clip_val():
    impl = ParetoQImpl(ParetoQConfig(w_bits=4), model=nn.Sequential(nn.Linear(2, 2)))
    assert impl.qparam_names() == ("weight_clip_val",)


def test_ParetoQImpl_get_convert_processors_returns_deploy_edges_when_called():
    names = {type(p).__name__ for p in ParetoQImpl.get_convert_processors()}
    assert names == {
        ParetoQToW4A8DynamicProcessor.__name__,
        ParetoQToW4A4DynamicProcessor.__name__,
    }


def test_ParetoQImpl_replace_linears_swaps_matching_modules():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    impl = ParetoQImpl(ParetoQConfig(w_bits=4, include=["*"], exclude=[]), model=model)
    shells = impl.replace_linears(model)
    assert len(shells) == 2
    assert all(isinstance(s, ParetoQFakeQuantLinear) for s in shells)


def test_ParetoQImpl_init_weight_clip_val_sets_positive_alpha_when_shells_given():
    model = nn.Sequential(nn.Linear(4, 4))
    impl = ParetoQImpl(ParetoQConfig(w_bits=4, include=["*"], exclude=[]), model=model)
    shells = impl.replace_linears(model)
    impl.init_weight_clip_val(shells)
    assert all(torch.all(s.weight_clip_val > 0) for s in shells)


def test_ParetoQImpl_replace_linears_skips_excluded_modules():
    model = nn.ModuleDict({"proj": nn.Linear(2, 2), "lm_head": nn.Linear(2, 3)})
    impl = ParetoQImpl(
        ParetoQConfig(w_bits=4, include=["*"], exclude=["lm_head"]),
        model=model,
    )
    shells = impl.replace_linears(model)
    assert len(shells) == 1
    assert isinstance(model["proj"], ParetoQFakeQuantLinear)
    assert isinstance(model["lm_head"], nn.Linear)
