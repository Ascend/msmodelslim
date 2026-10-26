#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import pytest

from msmodelslim.common.low_rank_decompose.export import Decompose


def _make_decompose_with_config_backend(monkeypatch, config):
    model = object()
    decompose = Decompose(model)
    captured = {}

    def fake_backend(*, network, hidden_channels, excludes, divisor):
        captured["network"] = network
        captured["hidden_channels"] = hidden_channels
        captured["excludes"] = excludes
        captured["divisor"] = divisor
        return config

    monkeypatch.setattr(decompose, "get_decomposed_config_backend", fake_backend)
    return decompose, model, captured


def test_init_rejects_non_json_config_file():
    with pytest.raises(ValueError):
        _ = Decompose(object(), config_file="config.yaml")


@pytest.mark.parametrize("channel_fixed", [0, -3, "16", 1.5])
def test_from_fixed_rejects_invalid_channel_fixed(channel_fixed):
    decompose = Decompose(object())

    with pytest.raises(ValueError):
        _ = decompose.from_fixed(channel_fixed)


def test_from_fixed_delegates_to_backend_and_returns_self(monkeypatch):
    decompose, model, captured = _make_decompose_with_config_backend(
        monkeypatch, {"layer0.fc": {"hidden_channels": 16}}
    )

    result = decompose.from_fixed(16, divisor=8)

    assert result is decompose
    assert captured == {"network": model, "hidden_channels": 16, "excludes": None, "divisor": 8}
    assert decompose.decompose_config == {"layer0.fc": {"hidden_channels": 16}}


def test_from_fixed_raises_when_backend_returns_empty_config(monkeypatch):
    decompose, _, _ = _make_decompose_with_config_backend(monkeypatch, {})

    with pytest.raises(ValueError):
        _ = decompose.from_fixed(16)


@pytest.mark.parametrize("channel_ratio", [0.0, 1.0, -0.2, "0.5"])
def test_from_ratio_rejects_invalid_channel_ratio(channel_ratio):
    decompose = Decompose(object())

    with pytest.raises(ValueError):
        _ = decompose.from_ratio(channel_ratio)


def test_from_ratio_delegates_to_backend_and_returns_self(monkeypatch):
    decompose, _, captured = _make_decompose_with_config_backend(monkeypatch, {"layer0.fc": {"hidden_channels": 16}})

    result = decompose.from_ratio(0.5, divisor=1)

    assert result is decompose
    assert captured["hidden_channels"] == 0.5
    assert captured["divisor"] == 1


@pytest.mark.parametrize("channel_dict", [{}, "not-a-dict", []])
def test_from_dict_rejects_invalid_channel_dict(channel_dict):
    decompose = Decompose(object())

    with pytest.raises(ValueError):
        _ = decompose.from_dict(channel_dict)


def test_from_dict_compiles_regex_keys_and_delegates_to_backend(monkeypatch):
    decompose, _, captured = _make_decompose_with_config_backend(monkeypatch, {"compiled": {"hidden_channels": 16}})

    result = decompose.from_dict({".*output.fc": 16})

    assert result is decompose
    compiled_keys = list(captured["hidden_channels"].keys())
    assert len(compiled_keys) == 1
    assert compiled_keys[0].pattern == ".*output.fc"
    assert captured["hidden_channels"][compiled_keys[0]] == 16


@pytest.mark.parametrize("kwargs", [{"divisor": 0}, {"excludes": "layer0"}])
def test_get_decomposed_config_rejects_invalid_options(monkeypatch, kwargs):
    decompose = Decompose(object())
    monkeypatch.setattr(decompose, "get_decomposed_config_backend", lambda **_: {"x": 1})

    with pytest.raises(ValueError):
        _ = decompose.from_fixed(16, **kwargs)


def test_decompose_network_raises_before_config_initialized():
    decompose = Decompose(object())

    with pytest.raises(ValueError):
        _ = decompose.decompose_network()


def test_decompose_network_delegates_to_backend_and_returns_result(monkeypatch):
    decompose, model, _ = _make_decompose_with_config_backend(monkeypatch, {"layer0.fc": {"hidden_channels": 16}})
    _ = decompose.from_fixed(16)
    result_marker = object()
    captured = {}

    def fake_decompose_backend(*, network, decompose_config, do_decompose_weight, datasets, max_iter):
        captured.update(
            network=network,
            decompose_config=decompose_config,
            do_decompose_weight=do_decompose_weight,
            datasets=datasets,
            max_iter=max_iter,
        )
        return result_marker

    monkeypatch.setattr(decompose, "decompose_network_backend", fake_decompose_backend)

    result = decompose.decompose_network(do_decompose_weight=False, max_iter=5)

    assert result is result_marker
    assert captured == {
        "network": model,
        "decompose_config": {"layer0.fc": {"hidden_channels": 16}},
        "do_decompose_weight": False,
        "datasets": None,
        "max_iter": 5,
    }


def test_decompose_network_rejects_non_bool_flag(monkeypatch):
    decompose, _, _ = _make_decompose_with_config_backend(monkeypatch, {"layer0.fc": {"hidden_channels": 16}})
    _ = decompose.from_fixed(16)

    with pytest.raises(TypeError):
        _ = decompose.decompose_network(do_decompose_weight="yes")


def test_decompose_network_rejects_non_int_max_iter(monkeypatch):
    decompose, _, _ = _make_decompose_with_config_backend(monkeypatch, {"layer0.fc": {"hidden_channels": 16}})
    _ = decompose.from_fixed(16)

    with pytest.raises(TypeError):
        _ = decompose.decompose_network(max_iter="all")
