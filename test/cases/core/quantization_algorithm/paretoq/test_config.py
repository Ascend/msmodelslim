#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest

from msmodelslim.core.quantization_algorithm.paretoq.config import (
    DeployAct,
    ParetoQActConfig,
    ParetoQConfig,
    ParetoQQuantSpec,
    ParetoQWeightConfig,
)
from msmodelslim.utils.exception import SchemaValidateError


def _cfg(weight=None, act=None, **kwargs):
    return ParetoQConfig(
        weight={"dtype": "int4"} if weight is None else weight,
        act={"dtype": "int8"} if act is None else act,
        **kwargs,
    )


def test_ParetoQConfig_defaults_nested_quantization_when_blocks_empty():
    cfg = _cfg(weight={}, act={})
    assert cfg.weight.w_bits == 4
    assert cfg.weight.scope == "per_channel"
    assert cfg.weight.qparam_init == "amax"
    assert cfg.act.dtype == "int8"
    assert cfg.act.scope == "per_token"
    assert cfg.act.qparam_init == "amax"
    assert cfg.act.deploy_act == "a8"


@pytest.mark.parametrize("missing", ["weight", "act"])
def test_ParetoQConfig_rejects_missing_quantization_block_when_block_missing(missing):
    kwargs = {"type": "paretoq"}
    if missing != "weight":
        kwargs["weight"] = {"dtype": "int4"}
    if missing != "act":
        kwargs["act"] = {"dtype": "int8"}

    with pytest.raises(SchemaValidateError, match=missing):
        ParetoQConfig(**kwargs)


def test_ParetoQConfig_rejects_unknown_algorithm_type_when_type_unknown():
    with pytest.raises(SchemaValidateError, match="type"):
        ParetoQConfig(type="other", weight={"dtype": "int4"}, act={"dtype": "int8"})


def test_ParetoQConfig_parses_documented_yaml_when_yaml_given():
    cfg = ParetoQConfig(
        type="paretoq",
        weight={
            "dtype": "int4",
            "scope": "per_channel",
            "symmetric": True,
            "qparam_init": "amax",
        },
        act={
            "dtype": "int8",
            "scope": "per_token",
            "symmetric": True,
            "qparam_init": "amax",
        },
        include=["model.layers.*.self_attn.q_proj"],
        exclude=["lm_head"],
    )

    assert cfg.include == ["model.layers.*.self_attn.q_proj"]
    assert cfg.exclude == ["lm_head"]
    assert cfg.weight.w_bits == 4
    assert cfg.weight.scope == "per_channel"
    assert cfg.weight.qparam_init == "amax"
    assert cfg.act.dtype == "int8"
    assert cfg.act.scope == "per_token"
    assert cfg.quant_spec.w_bits == 4
    assert cfg.quant_spec.deploy_act is DeployAct.A8


@pytest.mark.parametrize("w_bits", range(5))
def test_ParetoQConfig_maps_weight_dtype_to_w_bits_when_dtype_given(w_bits):
    assert _cfg(weight={"dtype": f"int{w_bits}"}).weight.w_bits == w_bits


def test_ParetoQConfig_rejects_weight_dtype_when_unsupported():
    with pytest.raises(SchemaValidateError, match="dtype"):
        _cfg(weight={"dtype": "int8"})


def test_ParetoQConfig_rejects_weight_symmetric_when_false():
    with pytest.raises(SchemaValidateError, match="symmetric"):
        _cfg(weight={"dtype": "int4", "symmetric": False})


def test_ParetoQConfig_rejects_weight_qparam_init_when_unsupported():
    with pytest.raises(SchemaValidateError, match="qparam_init"):
        _cfg(weight={"dtype": "int4", "qparam_init": "minmax"})


def test_ParetoQConfig_accepts_layer_wise_scope_when_scope_is_layer_wise():
    assert _cfg(weight={"dtype": "int4", "scope": "layer_wise"}).weight.scope == "layer_wise"


def test_ParetoQConfig_rejects_layerwise_alias_when_alias_used():
    with pytest.raises(SchemaValidateError, match="scope"):
        _cfg(weight={"dtype": "int4", "scope": "layerwise"})


@pytest.mark.parametrize(
    ("dtype", "deploy_act", "scope"),
    [
        ("int4", "a4", "per_token"),
        ("int8", "a8", "per_token"),
    ],
)
def test_ParetoQConfig_maps_act_dtype_to_deploy_act_when_dtype_given(dtype, deploy_act, scope):
    cfg = _cfg(act={"dtype": dtype})
    assert cfg.act.deploy_act == deploy_act
    assert cfg.act.scope == scope


def test_ParetoQConfig_quant_spec_combines_weight_and_act_when_both_given():
    cfg = _cfg(
        weight={"dtype": "int4", "scope": "layer_wise"},
        act={"dtype": "int8"},
    )
    spec = cfg.quant_spec
    assert spec.w_bits == 4
    assert spec.weight_layerwise is True
    assert spec.deploy_act is DeployAct.A8


@pytest.mark.parametrize("dtype", ["bf16", "float"])
def test_ParetoQConfig_rejects_act_dtype_when_unsupported(dtype):
    with pytest.raises(SchemaValidateError, match="dtype"):
        _cfg(act={"dtype": dtype})


@pytest.mark.parametrize(
    ("dtype", "scope"),
    [
        ("int4", "per_tensor"),
        ("int8", "per_tensor"),
    ],
)
def test_ParetoQConfig_rejects_act_scope_when_each_incompatible_dtype(dtype, scope):
    with pytest.raises(SchemaValidateError, match="act.scope"):
        _cfg(act={"dtype": dtype, "scope": scope})


@pytest.mark.parametrize(
    ("dtype", "scope", "deploy_act"),
    [
        ("int4", "per_token", "a4"),
        ("int8", "per_token", "a8"),
    ],
)
def test_ParetoQConfig_accepts_explicit_act_scope_when_compatible(dtype, scope, deploy_act):
    cfg = _cfg(act={"dtype": dtype, "scope": scope})
    assert cfg.act.scope == scope
    assert cfg.act.deploy_act == deploy_act


def test_ParetoQConfig_rejects_act_symmetric_when_false():
    with pytest.raises(SchemaValidateError, match="symmetric"):
        _cfg(act={"dtype": "int8", "symmetric": False})


def test_ParetoQConfig_rejects_act_qparam_init_when_unsupported():
    with pytest.raises(SchemaValidateError, match="qparam_init"):
        _cfg(act={"dtype": "int8", "qparam_init": "minmax"})


@pytest.mark.parametrize(
    ("legacy_field", "value"),
    [
        ("w_bits", 4),
        ("a_bits", 8),
        ("weight_scope", "per_channel"),
        ("weight_clip_init", "amax"),
        ("deploy_act", "a8"),
    ],
)
def test_ParetoQConfig_rejects_legacy_flat_fields_when_legacy_field_given(legacy_field, value):
    with pytest.raises(SchemaValidateError):
        _cfg(**{legacy_field: value})


def test_ParetoQConfig_rejects_removed_optional_fields_when_simulate_act_quant_present():
    with pytest.raises(SchemaValidateError):
        ParetoQConfig(
            weight={"dtype": "int4"},
            act={"dtype": "int8"},
            simulate_act_quant=False,
        )


def test_ParetoQConfig_rejects_unknown_nested_weight_field_when_method_present():
    with pytest.raises(SchemaValidateError):
        _cfg(weight={"dtype": "int4", "method": "minmax"})


def test_ParetoQConfig_rejects_unknown_nested_act_field_when_method_present():
    with pytest.raises(SchemaValidateError):
        _cfg(act={"dtype": "int8", "method": "minmax"})


@pytest.mark.parametrize("w_bits", [-1, 5, 7])
def test_ParetoQQuantSpec_rejects_unsupported_w_bits_when_w_bits_unsupported(w_bits):
    with pytest.raises(ValueError, match="Unsupported ParetoQ weight bits"):
        ParetoQQuantSpec(w_bits=w_bits, weight_layerwise=False, deploy_act=DeployAct.A8)


def test_ParetoQQuantSpec_rejects_unsupported_deploy_act_when_deploy_act_unsupported():
    with pytest.raises(ValueError, match="Unsupported ParetoQ deploy_act"):
        ParetoQQuantSpec(w_bits=4, weight_layerwise=False, deploy_act="a2")


def test_ParetoQQuantSpec_parses_deploy_act_string_when_string_given():
    spec = ParetoQQuantSpec(w_bits=4, weight_layerwise=False, deploy_act="a8")
    assert spec.deploy_act is DeployAct.A8


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (ParetoQConfig, "weight"),
        (ParetoQConfig, "act"),
        (ParetoQWeightConfig, "dtype"),
        (ParetoQWeightConfig, "scope"),
        (ParetoQWeightConfig, "symmetric"),
        (ParetoQWeightConfig, "qparam_init"),
        (ParetoQActConfig, "dtype"),
        (ParetoQActConfig, "scope"),
        (ParetoQActConfig, "symmetric"),
        (ParetoQActConfig, "qparam_init"),
    ],
)
def test_ParetoQConfig_fields_have_descriptions_when_inspected(model, field):
    assert (model.model_fields[field].description or "").strip()
