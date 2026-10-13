# -*- coding: UTF-8 -*-
"""Adapter hooks for modelslim_v0 W8A8S (pipeline protocol + save postprocess)."""

from __future__ import annotations

import json
from pathlib import Path

from msmodelslim.core.quant_service.modelslim_v0.pipeline_interface import PipelineInterface
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV0
from msmodelslim.model.qwen3_5_moe.model_adapter import Qwen3_5ModelAdapter


def test_adapter_implements_v0_pipeline_and_keeps_v1():
    from msmodelslim.core.infer_engine.interface import FakeQuantInferenceInterface
    from msmodelslim.model.interface_hub import (
        AscendV1SaveInterface,
        FA3QuantAdapterInterface,
        LayerWiseOffloadOptionalInterface,
        ModelSlimPipelineInterfaceV1,
    )

    assert issubclass(Qwen3_5ModelAdapter, ModelSlimPipelineInterfaceV0)
    assert issubclass(Qwen3_5ModelAdapter, PipelineInterface)
    assert Qwen3_5ModelAdapter.uses_image_calib(None) is True
    assert PipelineInterface.uses_image_calib(None) is False
    assert PipelineInterface.modelslim_v0_save_postprocess(None, "unused") is None
    assert callable(getattr(Qwen3_5ModelAdapter, "load_model"))
    assert callable(getattr(Qwen3_5ModelAdapter, "handle_dataset_by_batch"))
    assert callable(getattr(Qwen3_5ModelAdapter, "modelslim_v0_save_postprocess"))

    assert issubclass(Qwen3_5ModelAdapter, ModelSlimPipelineInterfaceV1)
    assert issubclass(Qwen3_5ModelAdapter, FA3QuantAdapterInterface)
    assert issubclass(Qwen3_5ModelAdapter, AscendV1SaveInterface)
    assert issubclass(Qwen3_5ModelAdapter, LayerWiseOffloadOptionalInterface)
    assert issubclass(Qwen3_5ModelAdapter, FakeQuantInferenceInterface)
    assert callable(getattr(Qwen3_5ModelAdapter, "inject_fa3_placeholders"))
    assert callable(getattr(Qwen3_5ModelAdapter, "ascendv1_save_module_preprocess"))


def test_adapter_save_postprocess_sets_torch_dtype(tmp_path: Path):
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    (save_dir / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}), encoding="utf-8")
    Qwen3_5ModelAdapter.modelslim_v0_save_postprocess(None, str(save_dir))
    saved = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["torch_dtype"] == "float16"


def test_adapter_save_postprocess_warns_when_config_missing(tmp_path: Path, monkeypatch):
    save_dir = tmp_path / "empty_out"
    save_dir.mkdir()
    warnings = []

    class _Logger:
        def warning(self, msg, *args):
            warnings.append(msg % args if args else msg)

        def info(self, msg, *args):
            return

    monkeypatch.setattr(
        "msmodelslim.model.qwen3_5_moe.model_adapter.get_logger",
        _Logger,
    )
    Qwen3_5ModelAdapter.modelslim_v0_save_postprocess(None, str(save_dir))
    assert not (save_dir / "config.json").exists()
    assert any("missing" in message for message in warnings)
