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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.generated_runner import KEY_DATA_LOADER
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.model.deepseek_v3.model_adapter import DeepSeekV3ModelAdapter
from msmodelslim.processor.lossless.processor import (
    LosslessCheckProcessor,
    LosslessCheckProcessorConfig,
)
from msmodelslim.processor.quarot.offline_quarot.quarot import QuaRotProcessorConfig
from msmodelslim.utils.cache.memory import clear_cache


ROTATION = torch.tensor(
    [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)


class _LazyStreamingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))
            self.o_proj.weight.copy_(torch.eye(4))

    def forward(self, hidden_states, **_kwargs):
        return (self.o_proj(self.q_proj(hidden_states)),)


class _LazyStreamingMTPBlock(_LazyStreamingBlock):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 4)
        self.enorm = nn.LayerNorm(4)
        self.hnorm = nn.LayerNorm(4)
        self.eh_proj = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            values = torch.arange(128, dtype=torch.float32).reshape(32, 4) / 128
            self.embed_tokens.weight.copy_(values)


class _LazyStreamingModel(nn.Module):
    def __init__(self, num_hidden_layers=2):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=32,
            hidden_size=4,
            num_hidden_layers=num_hidden_layers,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 4)
        self.model.layers = nn.ModuleList([_LazyStreamingBlock()])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            values = torch.arange(128, dtype=torch.float32).reshape(32, 4) / 128
            self.model.embed_tokens.weight.copy_(values)
            self.lm_head.weight.copy_(torch.eye(4))

    def forward(self, input_ids, **_kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def _lazy_get_ln_fuse_map(_adapter):
    return {}, {}


def _lazy_get_bake_names(_adapter):
    return [], []


def _lazy_get_rotate_map(adapter, block_size):
    del block_size
    left_rot = {}
    right_rot = {"lm_head": ROTATION}
    for layer_index in range(adapter.config.num_hidden_layers):
        right_rot[f"model.layers.{layer_index}.q_proj"] = ROTATION
        left_rot[f"model.layers.{layer_index}.o_proj"] = ROTATION
    final_layer = adapter.config.num_hidden_layers - 1
    right_rot[f"model.layers.{final_layer}.embed_tokens"] = ROTATION
    right_rot[f"model.layers.{final_layer}.eh_proj"] = torch.block_diag(ROTATION, ROTATION)
    left_rot[f"model.layers.{final_layer}.eh_proj"] = ROTATION
    return [adapter.RotatePair(left_rot={}, right_rot={"model.embed_tokens": ROTATION})], [
        adapter.RotatePair(left_rot=left_rot, right_rot=right_rot)
    ]


def _lazy_handle_dataset(_adapter, dataset, device=DeviceType.CPU):
    del device
    return [
        {
            "input_ids": torch.tensor([sample]),
            "attention_mask": torch.ones((1, len(sample)), dtype=torch.long),
        }
        for sample in dataset
    ]


def _load_lazy_layer(model, layer_index):
    is_mtp_layer = layer_index == model.config.num_hidden_layers - 1
    while len(model.model.layers) <= layer_index:
        block_type = _LazyStreamingMTPBlock if is_mtp_layer else _LazyStreamingBlock
        model.model.layers.append(block_type())
    if is_mtp_layer and not hasattr(model.model.layers[layer_index], "eh_proj"):
        model.model.layers[layer_index] = _LazyStreamingMTPBlock()
    return model.model.layers[layer_index]


def _lazy_generate_decoder_layer(_adapter, model):
    for layer_index in range(model.config.num_hidden_layers):
        yield f"model.layers.{layer_index}", _load_lazy_layer(model, layer_index)


def _lazy_mtp_preprocess(_adapter, model, mtp_decoder, inputs, args, kwargs):
    del inputs
    hidden_states = args[0]
    logits = model.lm_head(model.model.norm(hidden_states))
    token_ids = logits[:, -1, :].argmax(dim=-1)
    hidden_states = hidden_states + mtp_decoder.embed_tokens(token_ids).unsqueeze(1)
    return (hidden_states,), kwargs


def _lazy_enable_kv_cache(_adapter, model, need_kv_cache):
    del model, need_kv_cache


def _build_lazy_streaming_adapter(model):
    adapter = object.__new__(DeepSeekV3ModelAdapter)
    adapter.model = model
    adapter.config = model.config
    methods = {
        "get_ln_fuse_map": _lazy_get_ln_fuse_map,
        "get_bake_names": _lazy_get_bake_names,
        "get_rotate_map": _lazy_get_rotate_map,
        "handle_dataset": _lazy_handle_dataset,
        "generate_decoder_layer": _lazy_generate_decoder_layer,
        "mtp_preprocess": _lazy_mtp_preprocess,
        "enable_kv_cache": _lazy_enable_kv_cache,
    }
    for name, method in methods.items():
        setattr(adapter, name, MethodType(method, adapter))
    return adapter


class _RecordingLayerWiseRunner(LayerWiseRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = []

    def build_process_unit(self, *args, **kwargs):
        self.units = super().build_process_unit(*args, **kwargs)
        return self.units


def _run_layer_wise(num_hidden_layers):
    clear_cache(KEY_DATA_LOADER)
    try:
        model = _LazyStreamingModel(num_hidden_layers=num_hidden_layers)
        adapter = _build_lazy_streaming_adapter(model)
        runner = _RecordingLayerWiseRunner(adapter=adapter, offload_device="meta")
        runner.add_processor(
            LosslessCheckProcessorConfig(
                configs=[QuaRotProcessorConfig(block_size=-1, export_extra_info=False)],
                metrics=[{"name": "numerical_equivalence"}],
            )
        )
        runner.run(
            model=model,
            calib_data=[[1, 3, 5, 7], [2, 4, 6, 8]],
            device=DeviceType.CPU,
        )
        processor = next(unit.processor for unit in runner.units if isinstance(unit.processor, LosslessCheckProcessor))
        return model, adapter, processor
    finally:
        clear_cache(KEY_DATA_LOADER)


@pytest.mark.parametrize("num_hidden_layers", [1, 2])
def test_layer_wise_runner_covers_global_targets_and_offloads(
    num_hidden_layers,
):
    model, _adapter, processor = _run_layer_wise(num_hidden_layers)

    checked = set(processor.result["checked_targets"])
    assert processor.result["passed"] is True
    assert not processor.result["skipped"]
    assert processor.result["skip_reason"] is None
    assert not processor.result["missing_targets"]
    assert not processor.result["skipped_targets"]
    assert "model.layers.0.q_proj" in checked
    assert f"model.layers.{num_hidden_layers - 1}.o_proj" in checked
    assert "lm_head" in checked
    assert "model.embed_tokens" in checked
    assert f"model.layers.{num_hidden_layers - 1}.embed_tokens" in checked
    assert f"model.layers.{num_hidden_layers - 1}.eh_proj" in checked
    assert len(model.model.layers) == num_hidden_layers
    assert all(next(layer.parameters()).device.type == "meta" for layer in model.model.layers)
