# -*- coding: UTF-8 -*-

import pytest
import torch
import torch.nn.functional as F

from msmodelslim.core.quantization_algorithm.paretoq.config import DeployAct, ParetoQQuantSpec
from msmodelslim.core.quantization_algorithm.paretoq.core import fake_quant_weight
from msmodelslim.core.quantization_algorithm.paretoq.impl import ParetoQFakeQuantLinear
from msmodelslim.core.quantization_schema import (
    W4A4DynamicPerChannelFakeQuantLinear,
    W4A8DynamicFakeQuantLinear,
)


def _spec(w_bits=4, deploy_act=DeployAct.A8, weight_layerwise=False):
    return ParetoQQuantSpec(
        w_bits=w_bits,
        weight_layerwise=weight_layerwise,
        deploy_act=deploy_act,
    )


def _shell(w_bits=4, deploy_act=DeployAct.A8, weight_layerwise=False, bias=True):
    torch.manual_seed(0)
    module = ParetoQFakeQuantLinear(
        4,
        3,
        bias=bias,
        quant_spec=_spec(w_bits=w_bits, deploy_act=deploy_act, weight_layerwise=weight_layerwise),
    )
    with torch.no_grad():
        module.weight.normal_(0, 0.2)
        if module.bias is not None:
            module.bias.normal_()
        if hasattr(module, "weight_clip_val"):
            module.weight_clip_val.copy_(module.weight.abs().amax(dim=1, keepdim=True).clamp_min(0.1) / 7.0)
    return module


def test_ParetoQFakeQuantLinear_requires_quant_spec_when_constructed():
    with pytest.raises(TypeError, match="quant_spec"):
        ParetoQFakeQuantLinear(4, 3, bias=False)  # pylint: disable=missing-kwoa


def test_ParetoQFakeQuantLinear_casts_alpha_to_weight_dtype_when_provided():
    alpha = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    module = ParetoQFakeQuantLinear(4, 3, bias=False, quant_spec=_spec(), weight_clip_val=alpha)

    assert module.weight_clip_val.dtype == module.weight.dtype


def test_ParetoQFakeQuantLinear_clones_alpha_when_provided():
    alpha = torch.tensor([[1.0], [2.0], [3.0]])
    module = ParetoQFakeQuantLinear(4, 3, bias=False, quant_spec=_spec(), weight_clip_val=alpha)

    alpha.add_(100)

    torch.testing.assert_close(module.weight_clip_val, torch.tensor([[1.0], [2.0], [3.0]]))


def test_ParetoQFakeQuantLinear_forward_matches_manual_fake_quant_when_w4():
    module = _shell(w_bits=4)
    x = torch.randn(2, 4)
    expected_weight = fake_quant_weight(module.weight, module.weight_clip_val, 4, False).to(x.dtype)

    torch.testing.assert_close(module(x), F.linear(x, expected_weight, module.bias))


def test_ParetoQFakeQuantLinear_passes_weight_layerwise_to_fake_quant_when_enabled(monkeypatch):
    module = _shell(w_bits=4, weight_layerwise=True)
    calls = []

    def _fake_quant_weight(weight, alpha, w_bits, weight_layerwise):
        calls.append(weight_layerwise)
        return weight

    monkeypatch.setattr("msmodelslim.core.quantization_algorithm.paretoq.impl.fake_quant_weight", _fake_quant_weight)
    module(torch.randn(1, 4))

    assert calls == [True]


def test_ParetoQFakeQuantLinear_from_qat_tensors_builds_linear_shape_when_weight_is_2d():
    module = ParetoQFakeQuantLinear.from_qat_tensors(
        torch.randn(3, 4),
        torch.ones(3, 1),
        quant_spec=_spec(w_bits=4),
    )

    assert (module.in_features, module.out_features) == (4, 3)


def test_ParetoQFakeQuantLinear_from_qat_tensors_copies_weight_when_called():
    weight = torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8], [0.9, -1.0, 1.1, -1.2]])

    module = ParetoQFakeQuantLinear.from_qat_tensors(weight, torch.ones(3, 1), quant_spec=_spec(w_bits=4))

    torch.testing.assert_close(module.weight, weight)


def test_ParetoQFakeQuantLinear_from_qat_tensors_copies_alpha_when_called():
    alpha = torch.tensor([[0.1], [0.2], [0.3]])

    module = ParetoQFakeQuantLinear.from_qat_tensors(torch.ones(3, 4), alpha, quant_spec=_spec(w_bits=4))

    torch.testing.assert_close(module.weight_clip_val, alpha)


def test_ParetoQFakeQuantLinear_from_qat_tensors_restores_bias_when_present():
    bias = torch.tensor([0.5, -0.5, 0.25])

    module = ParetoQFakeQuantLinear.from_qat_tensors(
        torch.ones(3, 4),
        torch.ones(3, 1),
        bias,
        quant_spec=_spec(w_bits=4),
    )

    torch.testing.assert_close(module.bias, bias)


def test_ParetoQFakeQuantLinear_from_qat_tensors_omits_bias_when_absent():
    module = ParetoQFakeQuantLinear.from_qat_tensors(
        torch.ones(3, 4),
        torch.ones(3, 1),
        quant_spec=_spec(w_bits=4),
    )
    assert module.bias is None


def test_ParetoQFakeQuantLinear_from_qat_tensors_raises_when_weight_is_not_2d():
    with pytest.raises(ValueError, match="2D weight"):
        ParetoQFakeQuantLinear.from_qat_tensors(torch.randn(3), torch.ones(1, 1), quant_spec=_spec(w_bits=4))


@pytest.mark.parametrize(
    ("deploy_act", "expected_type"),
    [
        (DeployAct.A8, W4A8DynamicFakeQuantLinear),
        (DeployAct.A4, W4A4DynamicPerChannelFakeQuantLinear),
    ],
)
def test_ParetoQFakeQuantLinear_deploy_returns_target_ir_when_act_mode_changes(deploy_act, expected_type):
    assert isinstance(_shell(w_bits=4, deploy_act=deploy_act).deploy(), expected_type)


def test_ParetoQFakeQuantLinear_deploy_exports_alpha_as_scale_when_a8():
    module = _shell(w_bits=4, deploy_act=DeployAct.A8)
    deployed = module.deploy()

    torch.testing.assert_close(deployed.weight_scale, module.weight_clip_val.squeeze(-1))


def test_ParetoQFakeQuantLinear_deploy_preserves_bias_when_present():
    module = _shell(w_bits=4, deploy_act=DeployAct.A8, bias=True)
    deployed = module.deploy()

    torch.testing.assert_close(deployed.bias, module.bias)


def test_ParetoQFakeQuantLinear_deploy_emits_int32_zero_offset_when_a4():
    deployed = _shell(w_bits=4, deploy_act=DeployAct.A4).deploy()

    assert deployed.weight_offset.dtype == torch.int32


def test_ParetoQFakeQuantLinear_deploy_preserves_training_state_when_called():
    module = _shell(w_bits=4, deploy_act=DeployAct.A8)
    weight_before = module.weight.detach().clone()
    alpha_before = module.weight_clip_val.detach().clone()

    module.deploy()

    assert torch.equal(module.weight, weight_before)
    assert torch.equal(module.weight_clip_val, alpha_before)


def test_ParetoQFakeQuantLinear_deploy_clamps_weight_to_int4_range_when_value_exceeds_alpha():
    module = ParetoQFakeQuantLinear(
        7, 1, bias=False, quant_spec=_spec(deploy_act=DeployAct.A8), weight_clip_val=torch.ones(1, 1)
    )
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[-100.0, -7.2, -6.8, 0.6, 6.8, 7.2, 100.0]]))

    deployed = module.deploy()

    assert deployed.weight.tolist() == [[-8, -7, -7, 1, 7, 7, 7]]


def test_ParetoQFakeQuantLinear_deploy_keeps_weight_finite_when_alpha_is_zero():
    module = ParetoQFakeQuantLinear(
        3, 1, bias=False, quant_spec=_spec(deploy_act=DeployAct.A8), weight_clip_val=torch.zeros(1, 1)
    )
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0.0, 1.0, -1.0]]))

    deployed = module.deploy()

    assert torch.isfinite(deployed.weight).all()
