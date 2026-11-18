# -*- coding: UTF-8 -*-

import pytest
import torch
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig, ModuleRule
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.core.quantization_algorithm.paretoq.config import DeployAct, ParetoQConfig, ParetoQQuantSpec
from msmodelslim.core.quantization_algorithm.paretoq.convert import (
    ParetoQToW4A4DynamicProcessor,
    ParetoQToW4A8DynamicProcessor,
)
from msmodelslim.core.quantization_algorithm.paretoq.core import init_weight_clip_from_weight
from msmodelslim.core.quantization_algorithm.paretoq.impl import ParetoQFakeQuantLinear, ParetoQImpl
from msmodelslim.core.quantization_schema import (
    W4A4DynamicPerChannelFakeQuantLinear,
    W4A8DynamicFakeQuantLinear,
    W8A8StaticFakeQuantLinear,
    int4_per_channel_sym,
    int4_per_token_sym,
    int8_per_channel_sym,
    int8_per_tensor_asym,
    int8_per_token_sym,
)
from msmodelslim.core.quantization_schema.qal import QDType, QParam, QStorage
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


class _Holder(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.proj = module


def _cfg(weight=None, act=None, include=None, exclude=None):
    return ParetoQConfig(
        weight={"dtype": "int4"} if weight is None else weight,
        act={"dtype": "int8"} if act is None else act,
        include=["*"] if include is None else include,
        exclude=[] if exclude is None else exclude,
    )


def _w4a8(q, scale):
    return W4A8DynamicFakeQuantLinear(
        QParam(scheme=int8_per_token_sym),
        QParam(scheme=int4_per_channel_sym, ext={"scale": torch.as_tensor(scale, dtype=torch.float32)}),
        QStorage(dtype=QDType.INT4, value=torch.as_tensor(q, dtype=torch.int8)),
        None,
    )


def _w4a4(q, scale):
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
    return W4A4DynamicPerChannelFakeQuantLinear(
        QParam(scheme=int4_per_token_sym),
        QParam(
            scheme=int4_per_channel_sym,
            ext={"scale": scale_tensor, "offset": torch.zeros_like(scale_tensor, dtype=torch.int32)},
        ),
        QStorage(dtype=QDType.INT4, value=torch.as_tensor(q, dtype=torch.int8)),
        None,
    )


def _w8a8():
    return W8A8StaticFakeQuantLinear(
        QParam(scheme=int8_per_tensor_asym, ext={"scale": torch.tensor(1.0), "offset": torch.tensor(0.0)}),
        QParam(scheme=int8_per_channel_sym, ext={"scale": torch.ones(2)}),
        QStorage(dtype=QDType.INT8, value=torch.zeros(2, 2, dtype=torch.int8)),
        None,
    )


def _context(*rules):
    return ConvertContext(
        ConvertConfig(
            model_path="input",
            save_path="output",
            module_rules=list(rules),
        )
    )


def _rule(match, defaults=None):
    return ModuleRule(match=match, defaults={} if defaults is None else defaults)


def _virtual(full_name="blocks.0.proj", *, weight=None, alpha=None, bias=None):
    module = ModelFreeLinear(full_name=full_name)
    if weight is not None:
        module.weight = nn.Parameter(weight)
    if alpha is not None:
        module.weight_clip_val = nn.Parameter(alpha)
    if bias is not None:
        module.bias = nn.Parameter(bias)
    return module


def test_ParetoQImpl_qparam_names_returns_weight_clip_val_when_called():
    impl = ParetoQImpl(_cfg(), model=nn.Linear(2, 2))
    assert impl.qparam_names() == ("weight_clip_val",)


def test_ParetoQImpl_get_convert_processors_returns_all_deploy_edges_when_called():
    processors = ParetoQImpl.get_convert_processors()
    assert {type(processor) for processor in processors} == {
        ParetoQToW4A8DynamicProcessor,
        ParetoQToW4A4DynamicProcessor,
    }


def test_ParetoQImpl_replace_linears_returns_matching_module_names_when_called():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    impl = ParetoQImpl(_cfg(), model=model)

    shells = impl.replace_linears(model)

    assert [name for name, _ in shells] == ["0", "2"]


def test_ParetoQImpl_replace_linears_copies_weight_when_replacing():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3))
    expected = model[0].weight.detach().clone()

    ParetoQImpl(_cfg(), model=model).replace_linears(model)

    torch.testing.assert_close(model[0].weight, expected)


def test_ParetoQImpl_replace_linears_copies_bias_when_replacing():
    model = nn.Sequential(nn.Linear(4, 3))
    expected = torch.arange(3, dtype=torch.float32)
    with torch.no_grad():
        model[0].bias.copy_(expected)

    ParetoQImpl(_cfg(), model=model).replace_linears(model)

    torch.testing.assert_close(model[0].bias, expected)


def test_ParetoQImpl_replace_linears_skips_excluded_module_when_excluded():
    model = nn.ModuleDict({"proj": nn.Linear(2, 2), "lm_head": nn.Linear(2, 3)})
    impl = ParetoQImpl(_cfg(exclude=["lm_head"]), model=model)

    shells = impl.replace_linears(model)

    assert [name for name, _ in shells] == ["proj"]
    assert isinstance(model["lm_head"], nn.Linear)


def test_ParetoQImpl_replace_linears_skips_linear_subclass_when_exact_type_required():
    class _LinearSubclass(nn.Linear):
        pass

    model = nn.ModuleDict({"base": nn.Linear(2, 2), "derived": _LinearSubclass(2, 2)})

    shells = ParetoQImpl(_cfg(), model=model).replace_linears(model)

    assert [name for name, _ in shells] == ["base"]
    assert isinstance(model["derived"], _LinearSubclass)


def test_ParetoQImpl_replace_linears_returns_empty_when_model_has_no_linear():
    model = nn.Sequential(nn.ReLU())
    assert not ParetoQImpl(_cfg(), model=model).replace_linears(model)


def test_ParetoQImpl_replace_linears_initializes_alpha_with_amax_when_configured():
    model = nn.Sequential(nn.Linear(4, 3))
    expected = init_weight_clip_from_weight(model[0].weight, 4, "amax")

    ParetoQImpl(_cfg(), model=model).replace_linears(model)

    torch.testing.assert_close(model[0].weight_clip_val, expected)


def test_ParetoQImpl_replace_linears_initializes_alpha_with_mean2_sqrt_qp_when_configured():
    model = nn.Sequential(nn.Linear(4, 3))
    expected = init_weight_clip_from_weight(model[0].weight, 4, "mean2_sqrt_qp")

    ParetoQImpl(_cfg(weight={"dtype": "int4", "qparam_init": "mean2_sqrt_qp"}), model=model).replace_linears(model)

    torch.testing.assert_close(model[0].weight_clip_val, expected)


def test_ParetoQImpl_replace_linears_keeps_alpha_positive_when_weight_is_zero():
    model = nn.Sequential(nn.Linear(4, 3))
    with torch.no_grad():
        model[0].weight.zero_()

    ParetoQImpl(_cfg(), model=model).replace_linears(model)

    assert torch.all(model[0].weight_clip_val > 0)


def test_ParetoQImpl_replace_linears_returns_empty_when_model_has_no_supported_source():
    model = nn.Sequential(nn.ReLU())
    assert not ParetoQImpl(_cfg(), model=model).replace_linears(model)


def test_ParetoQImpl_replace_linears_initializes_alpha_when_float_linear_replaced():
    model = nn.Sequential(nn.Linear(4, 3))
    expected = init_weight_clip_from_weight(model[0].weight, 4, "amax")

    shells = ParetoQImpl(_cfg(), model=model).replace_linears(model)

    assert len(shells) == 1
    torch.testing.assert_close(model[0].weight_clip_val, expected)


def test_ParetoQImpl_replace_linears_builds_meta_shell_when_float_linear_is_meta():
    model = _Holder(nn.Linear(3, 2, device="meta"))

    replaced = ParetoQImpl(_cfg(), model=model).replace_linears(model)

    assert [name for name, _ in replaced] == ["proj"]
    assert isinstance(model.proj, ParetoQFakeQuantLinear)
    assert model.proj.weight.is_meta
    assert model.proj.weight_clip_val.is_meta


def test_ParetoQImpl_replace_linears_keeps_nested_module_path_when_replacing():
    class _Nested(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Module()])
            self.blocks[0].proj = module

    model = _Nested(nn.Linear(3, 2))

    replaced = ParetoQImpl(_cfg(include=["blocks.*.proj"]), model=model).replace_linears(model)

    assert [name for name, _ in replaced] == ["blocks.0.proj"]


def test_ParetoQImpl_replace_linears_raises_when_w8a8_ir_is_present():
    model = _Holder(_w8a8())

    with pytest.raises(UnsupportedError, match="W8A8StaticFakeQuantLinear"):
        ParetoQImpl(_cfg(), model=model).replace_linears(model)


@pytest.mark.parametrize(
    ("ir_factory", "ir_name"),
    [
        (lambda: _w4a8([[-2, 1], [3, -1]], [0.1, 0.2]), "W4A8DynamicFakeQuantLinear"),
        (lambda: _w4a4([[-2, 1], [3, -1]], [0.1, 0.2]), "W4A4DynamicPerChannelFakeQuantLinear"),
    ],
)
def test_ParetoQImpl_replace_linears_raises_when_unsupported_ir_is_present(ir_factory, ir_name):
    model = _Holder(ir_factory())

    with pytest.raises(UnsupportedError, match=ir_name):
        ParetoQImpl(_cfg(), model=model).replace_linears(model)


def test_ParetoQImpl_replace_linears_returns_empty_when_called_twice():
    model = _Holder(nn.Linear(3, 2))
    impl = ParetoQImpl(_cfg(), model=model)
    impl.replace_linears(model)

    assert not impl.replace_linears(model)


def test_ParetoQImpl_materialize_from_virtual_uses_default_spec_when_context_is_none():
    shell = ParetoQImpl.materialize_from_virtual(_virtual(weight=torch.ones(2, 3), alpha=torch.ones(2, 1)))
    assert shell.quant_spec == ParetoQQuantSpec(w_bits=4, weight_layerwise=False, deploy_act=DeployAct.A8)


def test_ParetoQImpl_materialize_from_virtual_returns_existing_shell_when_shell_is_given():
    shell = ParetoQFakeQuantLinear(
        3,
        2,
        quant_spec=ParetoQQuantSpec(w_bits=4, weight_layerwise=False, deploy_act=DeployAct.A8),
    )
    assert ParetoQImpl.materialize_from_virtual(shell) is shell


def test_ParetoQImpl_materialize_from_virtual_raises_when_module_type_is_unsupported():
    with pytest.raises(SchemaValidateError, match="ModelFreeLinear or ParetoQFakeQuantLinear"):
        ParetoQImpl.materialize_from_virtual(nn.Linear(3, 2))


def test_ParetoQImpl_materialize_from_virtual_raises_when_weight_is_missing():
    with pytest.raises(SchemaValidateError, match="requires a 2D weight"):
        ParetoQImpl.materialize_from_virtual(_virtual(alpha=torch.ones(2, 1)))


def test_ParetoQImpl_materialize_from_virtual_raises_when_weight_is_not_2d():
    with pytest.raises(SchemaValidateError, match="requires a 2D weight"):
        ParetoQImpl.materialize_from_virtual(_virtual(weight=torch.ones(3), alpha=torch.ones(1, 1)))


def test_ParetoQImpl_materialize_from_virtual_raises_when_alpha_is_missing():
    with pytest.raises(SchemaValidateError, match="requires weight_clip_val"):
        ParetoQImpl.materialize_from_virtual(_virtual(weight=torch.ones(2, 3)))


@pytest.mark.parametrize(
    "defaults",
    [
        {"w_bits": 8},
        {"deploy_act": "a2"},
        {"w_bits": "not-an-int"},
    ],
)
def test_ParetoQImpl_materialize_from_virtual_raises_when_metadata_is_invalid(defaults):
    module = _virtual(weight=torch.ones(2, 3), alpha=torch.ones(2, 1))

    with pytest.raises(SchemaValidateError, match="Invalid ParetoQ metadata"):
        ParetoQImpl.materialize_from_virtual(module, _context(_rule("blocks.*", defaults)))


def test_ParetoQImpl_materialize_from_virtual_prefers_exact_rule_when_wildcard_also_matches():
    module = _virtual(weight=torch.ones(2, 3), alpha=torch.ones(2, 1))
    context = _context(
        _rule("blocks.*", {"w_bits": 2, "weight_layerwise": True, "deploy_act": "a8"}),
        _rule("blocks.0.proj", {"w_bits": 3, "weight_layerwise": False, "deploy_act": "a8"}),
    )

    shell = ParetoQImpl.materialize_from_virtual(module, context)

    assert shell.quant_spec == ParetoQQuantSpec(w_bits=3, weight_layerwise=False, deploy_act=DeployAct.A8)


def test_ParetoQImpl_materialize_from_virtual_uses_wildcard_rule_when_exact_rule_is_absent():
    module = _virtual(weight=torch.ones(2, 3), alpha=torch.ones(2, 1))
    context = _context(_rule("blocks.*", {"w_bits": 2, "weight_layerwise": True, "deploy_act": "a4"}))

    shell = ParetoQImpl.materialize_from_virtual(module, context)

    assert shell.quant_spec == ParetoQQuantSpec(w_bits=2, weight_layerwise=True, deploy_act=DeployAct.A4)


def test_ParetoQImpl_materialize_from_virtual_uses_later_wildcard_when_first_rule_has_no_defaults():
    module = _virtual(weight=torch.ones(2, 3), alpha=torch.ones(2, 1))
    context = _context(
        _rule("blocks.0.proj"),
        _rule("blocks.*", {"w_bits": 2, "deploy_act": "a8"}),
    )

    shell = ParetoQImpl.materialize_from_virtual(module, context)

    assert shell.w_bits == 2


def test_ParetoQImpl_materialize_from_virtual_restores_weight_when_shell_is_created():
    weight = torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]])

    shell = ParetoQImpl.materialize_from_virtual(_virtual(weight=weight, alpha=torch.ones(2, 1)))

    torch.testing.assert_close(shell.weight, weight)


def test_ParetoQImpl_materialize_from_virtual_restores_trained_alpha_when_shell_is_created():
    alpha = torch.tensor([[0.1], [0.2]])

    shell = ParetoQImpl.materialize_from_virtual(_virtual(weight=torch.ones(2, 3), alpha=alpha))

    torch.testing.assert_close(shell.weight_clip_val, alpha)


def test_ParetoQImpl_materialize_from_virtual_restores_bias_when_shell_is_created():
    bias = torch.tensor([1.0, -1.0])

    shell = ParetoQImpl.materialize_from_virtual(_virtual(weight=torch.ones(2, 3), alpha=torch.ones(2, 1), bias=bias))

    torch.testing.assert_close(shell.bias, bias)


def test_ParetoQImpl_transform_for_convert_uses_trained_alpha_when_deploying():
    module = _virtual(weight=torch.ones(2, 3) * 0.1, alpha=torch.tensor([[0.1], [0.2]]))
    context = _context(_rule("blocks.*", {"w_bits": 4, "deploy_act": "a8"}))

    deployed = ParetoQImpl.transform_for_convert(module, context)

    assert isinstance(deployed, W4A8DynamicFakeQuantLinear)
    torch.testing.assert_close(deployed.weight_scale, torch.tensor([0.1, 0.2]))
