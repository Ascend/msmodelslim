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
from torch import nn

from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import ForwardMode
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_quant_manager import FlatQuantLayerManager
from msmodelslim.processor.flat_quant.flat_quant_utils.structure_pair import StructurePair
from msmodelslim.utils.exception import SchemaValidateError


class _RecordingStructurePair(StructurePair, register=False):
    _name = "recording"

    def __init__(self, config, sources, targets, prefix_name, model, **kwargs):
        super().__init__(sources, targets, prefix_name, model)
        self.config = config
        self.extra_config = kwargs
        self.wrap_called = 0
        self.rollback_called_with = None
        self.mode_changed = None

    def wrap_linear(self):
        self.wrap_called += 1

    def rollback_trans(self, pair_name=""):
        self.rollback_called_with = pair_name

    def change_mode(self, mod):
        self.mode_changed = mod


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


def _build_manager(monkeypatch):
    monkeypatch.setattr(StructurePair, "support_structure_pairs", [_RecordingStructurePair])
    manager = FlatQuantLayerManager(_Model())
    manager.register_layer_pairs(
        [{"source": "linear", "targets": ["linear"], "pair_class": _RecordingStructurePair}],
        layer_name="",
    )
    return manager


def test_flat_quant_layer_manager_registers_pair_when_source_matches(monkeypatch):
    manager = _build_manager(monkeypatch)
    manager.wrap_linear(device="cpu")
    pair = manager._layer_pairs_list[0]
    assert pair.wrap_called == 1
    assert pair.source_modules == "linear"


def test_flat_quant_layer_manager_calls_wrap_when_wrap_linear_invoked(monkeypatch):
    manager = _build_manager(monkeypatch)
    pair = manager._layer_pairs_list[0]
    manager.wrap_linear(device="cpu")
    assert pair.wrap_called == 1


def test_flat_quant_layer_manager_passes_pair_name_when_rollback_invoked(monkeypatch):
    manager = _build_manager(monkeypatch)
    pair = manager._layer_pairs_list[0]
    manager.rollback_trans("linear")
    assert pair.rollback_called_with == "linear"


def test_flat_quant_layer_manager_passes_mode_when_change_mode_invoked(monkeypatch):
    manager = _build_manager(monkeypatch)
    pair = manager._layer_pairs_list[0]
    manager.change_mode(ForwardMode.CALIB)
    assert pair.mode_changed is ForwardMode.CALIB


def test_flat_quant_layer_manager_dedups_pair_when_same_pair_registered_twice():
    manager = FlatQuantLayerManager(nn.Linear(4, 4))
    pair = _RecordingStructurePair(None, "linear", ["linear"], "", manager.module)

    manager.register_structure_pair(pair)
    manager.register_structure_pair(pair)

    assert manager._structure_pair_map["_RecordingStructurePair"] == [pair]


def test_flat_quant_layer_manager_raises_schema_validate_error_when_pair_type_is_invalid():
    manager = FlatQuantLayerManager(nn.Linear(4, 4))

    with pytest.raises(SchemaValidateError, match="pair must be an instance of StructurePair"):
        manager.register_structure_pair("not a pair")  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
