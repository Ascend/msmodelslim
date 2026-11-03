#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from msmodelslim.format.checkpoint_format.checkpoint_saver_format import (
    CheckpointSaverFormat,
)
from msmodelslim.format.checkpoint_format.constants import STRUCTURE_FILE, WEIGHT_FILE
from msmodelslim.format.interface import ExportContext
from msmodelslim.processor.paretoq.impl import ParetoQFakeQuantLinear


class _JsonWriter:
    def __init__(self):
        self.payload = None

    def dump(self, data, *, indent=2):
        self.payload = data


class _JsonFactory:
    def __init__(self, writer):
        self._writer = writer

    def create_json_writer(self, save_directory, file_name):
        assert file_name == STRUCTURE_FILE
        return self._writer


class _StWriter:
    def __init__(self):
        self.written = {}
        self.closed = False

    def write(self, key, value):
        self.written[key] = value

    def close(self):
        self.closed = True


class _StFactory:
    def __init__(self, writer):
        self._writer = writer

    def create_safetensors_writer(self, part_file_size, save_directory, save_prefix):
        return self._writer


def test_CheckpointSaverFormat_finalize_export_raises_when_state_dict_missing(tmp_path):
    ctx = ExportContext(save_directory=tmp_path, state_dict={})
    fmt = CheckpointSaverFormat(
        ctx,
        _StFactory(_StWriter()),
        _JsonFactory(_JsonWriter()),
        qparam_names=["weight_clip_val"],
    )
    fmt.prepare_export()
    with pytest.raises(RuntimeError, match="state_dict"):
        fmt.finalize_export(MagicMock())


def test_CheckpointSaverFormat_on_paretoq_linear_records_meta_when_called():
    ctx = ExportContext(save_directory=Path("."), state_dict={"w": torch.ones(1)})
    fmt = CheckpointSaverFormat(
        ctx,
        _StFactory(_StWriter()),
        _JsonFactory(_JsonWriter()),
        qparam_names=["weight_clip_val"],
    )
    module = ParetoQFakeQuantLinear(2, 2, w_bits=4, deploy_act="a8")
    fmt.on_paretoq_linear("model.layers.0._fsdp_wrapped_module.mlp", module)
    meta = fmt._modules_meta["model.layers.0.mlp"]
    assert meta["w_bits"] == 4
    assert meta["deploy_act"] == "a8"
    assert meta["class"] == "ParetoQFakeQuantLinear"


def test_CheckpointSaverFormat_finalize_export_writes_json_and_tensors_when_ready(
    tmp_path,
):
    st = _StWriter()
    js = _JsonWriter()
    weight = torch.randn(2, 2)
    ctx = ExportContext(
        save_directory=tmp_path,
        state_dict={"layer.weight": weight},
        source_model_path=None,
    )
    fmt = CheckpointSaverFormat(
        ctx,
        _StFactory(st),
        _JsonFactory(js),
        qparam_names=["weight_clip_val"],
        algorithm=[{"type": "paretoq"}],
    )
    fmt.prepare_export()
    fmt.on_paretoq_linear("layer", ParetoQFakeQuantLinear(2, 2, w_bits=4))
    fmt.finalize_export(MagicMock())

    assert "layer.weight" in st.written
    assert getattr(st, "closed", False) is True
    assert js.payload["format"]
    assert js.payload["modules"]["layer"]["w_bits"] == 4
    assert js.payload["tensors"]["weight_file"] == WEIGHT_FILE
    assert js.payload["qparam_names"] == ["weight_clip_val"]
