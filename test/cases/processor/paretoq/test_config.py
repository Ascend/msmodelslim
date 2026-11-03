#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest

from msmodelslim.processor.paretoq.config import (
    SUPPORTED_A_BITS,
    SUPPORTED_W_BITS,
    ParetoQConfig,
)
from msmodelslim.utils.exception import SchemaValidateError


def test_ParetoQConfig_defaults_type_when_minimal():
    assert ParetoQConfig().type == "paretoq"


def test_ParetoQConfig_defaults_w_bits_when_minimal():
    assert ParetoQConfig().w_bits == 4


def test_ParetoQConfig_defaults_a_bits_when_minimal():
    assert ParetoQConfig().a_bits == 8


def test_ParetoQConfig_defaults_deploy_act_when_minimal():
    assert ParetoQConfig().deploy_act == "a8"


def test_ParetoQConfig_defaults_weight_scope_when_minimal():
    assert ParetoQConfig().weight_scope == "per_channel"


@pytest.mark.parametrize("w_bits", [-1, 5, 8, 16])
def test_ParetoQConfig_rejects_w_bits_when_unsupported(w_bits):
    with pytest.raises(SchemaValidateError, match="w_bits"):
        ParetoQConfig(w_bits=w_bits)


@pytest.mark.parametrize("a_bits", [0, 2, 16])
def test_ParetoQConfig_rejects_a_bits_when_unsupported(a_bits):
    with pytest.raises(SchemaValidateError, match="a_bits"):
        ParetoQConfig(a_bits=a_bits)


def test_ParetoQConfig_rejects_deploy_act_when_unknown():
    with pytest.raises(SchemaValidateError, match="deploy_act"):
        ParetoQConfig(deploy_act="a16")


def test_ParetoQConfig_maps_legacy_deploy_act_to_a_bits_when_a4():
    assert ParetoQConfig(deploy_act="a4").a_bits == 4


def test_ParetoQConfig_maps_legacy_deploy_act_tag_when_a4():
    assert ParetoQConfig(deploy_act="a4").deploy_act == "a4"


def test_ParetoQConfig_rejects_conflicting_a_bits_and_deploy_act():
    with pytest.raises(SchemaValidateError, match="Conflicting"):
        ParetoQConfig(a_bits=8, deploy_act="a4")


@pytest.mark.parametrize("w_bits", sorted(SUPPORTED_W_BITS))
def test_ParetoQConfig_accepts_w_bits_when_supported(w_bits):
    assert ParetoQConfig(w_bits=w_bits).w_bits == w_bits


@pytest.mark.parametrize("a_bits", sorted(SUPPORTED_A_BITS))
def test_ParetoQConfig_accepts_a_bits_when_supported(a_bits):
    assert ParetoQConfig(a_bits=a_bits).a_bits == a_bits


@pytest.mark.parametrize(
    ("a_bits", "deploy_act"),
    [(4, "a4"), (8, "a8")],
)
def test_ParetoQConfig_derives_deploy_act_when_a_bits_set(a_bits, deploy_act):
    assert ParetoQConfig(a_bits=a_bits).deploy_act == deploy_act
