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

from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest, DataUnit
from msmodelslim.core.runner.generated_runner import GeneratedProcessUnit, GeneratedRunner
from msmodelslim.model.deepseek_v3.model_adapter import DeepSeekV3ModelAdapter
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.processor.lossless.processor import (
    LosslessCheckProcessor,
    LosslessCheckProcessorConfig,
)
from msmodelslim.processor.quarot.offline_quarot.quarot import QuaRotProcessorConfig


ROTATION = torch.tensor(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
)


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, hidden_size=4, vocab_size=16, mtp=False, dormant=False):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        if mtp:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.enorm = nn.Identity()
            self.hnorm = nn.Identity()
            self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        if dormant:
            self.unused_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **_kwargs):
        return (hidden_states + self.o_proj(self.q_proj(hidden_states)),)


class _TinyDeepSeek(nn.Module):
    def __init__(self, dormant=False, hidden_size=4, vocab_size=16):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(hidden_size, vocab_size),
                DeepseekV3DecoderLayer(hidden_size, vocab_size, mtp=True, dormant=dormant),
            ]
        )
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, **_kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return self.lm_head(self.model.norm(hidden_states))


def _generate_tiny_deepseek_decoder_layer(_adapter, model):
    return [(f"model.layers.{index}", layer) for index, layer in enumerate(model.model.layers)]


def _get_empty_ln_fuse_map(_adapter):
    return {}, {}


def _get_empty_bake_names(_adapter):
    return [], []


def _get_tiny_deepseek_rotate_map(adapter, block_size):
    del block_size
    right = {"lm_head": ROTATION}
    left = {}
    for index in range(adapter.config.num_hidden_layers):
        right[f"model.layers.{index}.q_proj"] = ROTATION
        left[f"model.layers.{index}.o_proj"] = ROTATION
    final = adapter.config.num_hidden_layers - 1
    right[f"model.layers.{final}.embed_tokens"] = ROTATION
    right[f"model.layers.{final}.eh_proj"] = torch.block_diag(ROTATION, ROTATION)
    left[f"model.layers.{final}.eh_proj"] = ROTATION
    pre_run = adapter.RotatePair(left_rot={}, right_rot={"model.embed_tokens": ROTATION})
    return [pre_run], [adapter.RotatePair(left_rot=left, right_rot=right)]


def _preprocess_tiny_deepseek_mtp(_adapter, model, mtp_decoder, inputs, args, kwargs):
    pre_hidden_states = args[0]
    logits = model.lm_head(model.model.norm(pre_hidden_states)).float()
    input_ids = inputs["input_ids"].clone()
    input_ids[:, -1] = logits[:, -1, :].argmax(dim=-1)
    input_embeds = mtp_decoder.enorm(mtp_decoder.embed_tokens(input_ids))
    hidden_states = mtp_decoder.hnorm(pre_hidden_states)
    hidden_states = mtp_decoder.eh_proj(torch.cat([input_embeds, hidden_states], dim=-1))
    return (hidden_states,), kwargs


def _build_tiny_deepseek_adapter(model):
    adapter = object.__new__(DeepSeekV3ModelAdapter)
    adapter.config = SimpleNamespace(num_hidden_layers=len(model.model.layers), hidden_size=4)
    methods = {
        "generate_decoder_layer": _generate_tiny_deepseek_decoder_layer,
        "get_ln_fuse_map": _get_empty_ln_fuse_map,
        "get_bake_names": _get_empty_bake_names,
        "get_rotate_map": _get_tiny_deepseek_rotate_map,
        "mtp_preprocess": _preprocess_tiny_deepseek_mtp,
    }
    for name, method in methods.items():
        setattr(adapter, name, MethodType(method, adapter))
    return adapter


def _build(model, adapter):
    return LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(
            configs=[QuaRotProcessorConfig(block_size=-1, export_extra_info=False)],
            metrics=[{"name": "numerical_equivalence"}],
        ),
        adapter,
    )


def _run(processor, adapter, model):
    processor.pre_run()

    generator = adapter.generate_model_forward(
        model,
        {
            "input_ids": torch.tensor([[2, 4, 0, 6]]),
            "attention_mask": torch.ones(1, 4),
        },
    )
    response = None
    while True:
        try:
            request = generator.send(response)
        except StopIteration:
            break
        batch = BatchProcessRequest(request.name, request.module, [(request.args, request.kwargs)], None)
        processor.process(batch)
        response = batch.outputs[0]
    processor.post_run()


class _RecordingPassThroughProcessor(AutoSessionProcessor):
    def __init__(self, model):
        super().__init__(model)
        self.request_names = []
        self.final_output = None

    def process(self, request):
        self.request_names.append(request.name)
        super().process(request)
        if request.name == "model.layers.1" and request.outputs:
            output = request.outputs[-1]
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.final_output = output.detach().clone()


def _run_generated_schedule(model, adapter, processors, monkeypatch):
    import msmodelslim.core.runner.generated_runner as generated_runner_module

    batch = {
        "input_ids": torch.tensor([[2, 4, 0, 6]]),
        "attention_mask": torch.ones(1, 4),
    }
    monkeypatch.setattr(generated_runner_module, "get_input_datas", lambda _adapter, data: data)
    data_recorder = DataUnit(None, None)
    units = [GeneratedProcessUnit(model, processor, adapter, [batch], data_recorder) for processor in processors]
    GeneratedRunner(adapter).generated_schedule(units, data_recorder)
    return data_recorder


def _capture_final_layer_output(processor):
    process = processor.process
    captured_output = {}

    def capture(request):
        process(request)
        if request.name == "model.layers.1" and request.outputs:
            output = request.outputs[-1]
            if isinstance(output, (tuple, list)):
                output = output[0]
            captured_output["value"] = output.detach().clone()

    processor.process = capture
    return captured_output


@pytest.mark.parametrize("seed", [11, 17, 23])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_deepseek_tiny_checks_global_and_mtp_targets_through_adapter_requests(dtype, seed):
    torch.manual_seed(seed)
    model = _TinyDeepSeek().to(dtype=dtype)
    adapter = _build_tiny_deepseek_adapter(model)
    processor = _build(model, adapter)

    _run(processor, adapter, model)

    checked = {check["structure"] for check in processor.result["checks"]}
    assert processor.result["passed"] is True
    assert not processor.result["skipped"]
    assert not processor.result["skipped_targets"]
    assert "model.embed_tokens" in checked
    assert "lm_head" in checked
    assert "model.layers.1.embed_tokens" in checked
    assert "model.layers.1.eh_proj" in checked
    assert "model.layers.1.q_proj" in checked
    assert any(check["reference_dtype"] == str(dtype) for check in processor.result["checks"])
    assert any(check["request"] == "model.layers.1" for check in processor.result["checks"])


def test_deepseek_unrouted_linear_is_reported_as_skipped_not_passed():
    torch.manual_seed(11)
    model = _TinyDeepSeek(dormant=True)
    adapter = _build_tiny_deepseek_adapter(model)
    processor = _build(model, adapter)

    _run(processor, adapter, model)

    assert processor.result["passed"] is None
    assert "model.layers.1.unused_proj" in processor.result["skipped_targets"]
    assert not processor.result["anomalies"]


def test_deepseek_lossless_requests_restore_adapter_mode_and_preserve_normal_flow():
    torch.manual_seed(31)
    model = _TinyDeepSeek()
    adapter = _build_tiny_deepseek_adapter(model)
    adapter._lossless_check_expose_global_requests = False
    processor = _build(model, adapter)

    _run(processor, adapter, model)

    assert adapter._lossless_check_expose_global_requests is False
    assert not getattr(adapter, "_lossless_check_forward_hooks", {})
    assert not getattr(adapter, "_lossless_check_mtp_forward_wrappers", {})

    generator = adapter.generate_model_forward(
        model,
        {
            "input_ids": torch.tensor([[2, 4, 0, 6]]),
            "attention_mask": torch.ones(1, 4),
        },
    )
    first_request = generator.send(None)
    generator.close()
    assert first_request.name == "model.layers.0"


def test_deepseek_quarot_abort_restores_adapter_mode_without_applying_pending_rotations():
    torch.manual_seed(37)
    model = _TinyDeepSeek()
    adapter = _build_tiny_deepseek_adapter(model)
    adapter._lossless_check_expose_global_requests = False
    processor = _build(model, adapter)
    original_embedding = model.model.embed_tokens.weight.detach().clone()

    processor.pre_run()

    assert adapter._lossless_check_expose_global_requests is True
    assert processor.processor.rotate_commands
    processor.processor.abort_lossless_check()

    assert adapter._lossless_check_expose_global_requests is False
    assert not getattr(adapter, "_lossless_check_forward_hooks", {})
    assert not getattr(adapter, "_lossless_check_mtp_forward_wrappers", {})
    # Aborting discards deferred work; it does not run the pending transformations.
    assert torch.equal(model.model.embed_tokens.weight, original_embedding)


def test_deepseek_lossless_and_normal_units_share_request_stream_in_generated_schedule(monkeypatch):
    torch.manual_seed(41)
    initial_model = _TinyDeepSeek()
    baseline_model = deepcopy(initial_model)
    model = deepcopy(initial_model)
    baseline_adapter = _build_tiny_deepseek_adapter(baseline_model)
    baseline_lossless_processor = _build(baseline_model, baseline_adapter)
    baseline_output = _capture_final_layer_output(baseline_lossless_processor)
    _run_generated_schedule(baseline_model, baseline_adapter, [baseline_lossless_processor], monkeypatch)

    adapter = _build_tiny_deepseek_adapter(model)
    lossless_processor = _build(model, adapter)
    normal_processor = _RecordingPassThroughProcessor(model)
    _run_generated_schedule(model, adapter, [lossless_processor, normal_processor], monkeypatch)

    assert normal_processor.request_names == [
        "model.embed_tokens",
        "model.layers.0",
        "lm_head",
        "model.layers.1",
    ]
    assert baseline_lossless_processor.result["passed"] is True
    assert lossless_processor.result["passed"] is True
    assert normal_processor.final_output is not None
    assert baseline_output.get("value") is not None
    torch.testing.assert_close(normal_processor.final_output, baseline_output["value"])


def test_deepseek_two_lossless_units_restore_shared_adapter_mode_after_normal_post_run():
    torch.manual_seed(43)
    model = _TinyDeepSeek()
    adapter = _build_tiny_deepseek_adapter(model)
    adapter._lossless_check_expose_global_requests = False
    first = _build(model, adapter)
    second = _build(model, adapter)

    first.pre_run()
    second.pre_run()
    assert adapter._lossless_check_expose_global_requests is True

    first.post_run()
    assert adapter._lossless_check_expose_global_requests is True
    second.post_run()

    assert adapter._lossless_check_expose_global_requests is False
    assert not getattr(adapter, "_lossless_check_request_mode_state", {})
    assert not getattr(adapter, "_lossless_check_forward_hooks", {})
    assert not getattr(adapter, "_lossless_check_mtp_forward_wrappers", {})


def test_deepseek_two_lossless_units_restore_shared_adapter_mode_after_exception(monkeypatch):
    torch.manual_seed(47)
    model = _TinyDeepSeek()
    adapter = _build_tiny_deepseek_adapter(model)
    adapter._lossless_check_expose_global_requests = False
    first = _build(model, adapter)
    second = _build(model, adapter)
    first.pre_run()
    second.pre_run()

    def fail_preprocess(_request):
        raise RuntimeError("injected child processor failure")

    monkeypatch.setattr(first.processor, "preprocess", fail_preprocess)
    request = BatchProcessRequest(
        "model.layers.0",
        model.model.layers[0],
        [((torch.ones(1, 2, 4),), {})],
        None,
    )
    with pytest.raises(RuntimeError, match="injected child processor failure"):
        first.process(request)

    assert first.result["passed"] is False
    assert adapter._lossless_check_expose_global_requests is True
    second.processor.abort_lossless_check()

    assert adapter._lossless_check_expose_global_requests is False
    assert not getattr(adapter, "_lossless_check_request_mode_state", {})


def test_deepseek_two_lossless_units_share_adapter_in_generated_schedule(monkeypatch):
    torch.manual_seed(53)
    model = _TinyDeepSeek()
    adapter = _build_tiny_deepseek_adapter(model)
    first = _build(model, adapter)
    second = _build(model, adapter)

    _run_generated_schedule(model, adapter, [first, second], monkeypatch)

    assert first.result["passed"] is True
    assert second.result["passed"] is True
    assert not hasattr(adapter, "_lossless_check_expose_global_requests")
    assert not getattr(adapter, "_lossless_check_request_mode_state", {})


@pytest.mark.parametrize("cleanup_raises", [False, True], ids=["cleanup-succeeds", "cleanup-raises"])
def test_deepseek_generated_schedule_aborts_all_lossless_units_after_sibling_failure(monkeypatch, cleanup_raises):
    torch.manual_seed(59)
    model = _TinyDeepSeek()
    adapter = _build_tiny_deepseek_adapter(model)
    adapter._lossless_check_expose_global_requests = False
    first = _build(model, adapter)
    second = _build(model, adapter)
    failing_sibling = _RecordingPassThroughProcessor(model)
    process = failing_sibling.process

    if cleanup_raises:
        abort = first.processor.abort_lossless_check

        def abort_then_fail():
            abort()
            raise ValueError("injected child cleanup failure")

        monkeypatch.setattr(first.processor, "abort_lossless_check", abort_then_fail)

    def fail_process(request):
        if request.name == "model.layers.1":
            assert adapter._lossless_check_expose_global_requests is True
            assert getattr(adapter, "_lossless_check_forward_hooks", {})
            assert getattr(adapter, "_lossless_check_mtp_forward_wrappers", {})
            raise RuntimeError("injected sibling processor failure")
        process(request)

    monkeypatch.setattr(failing_sibling, "process", fail_process)

    with pytest.raises(RuntimeError, match="injected sibling processor failure"):
        _run_generated_schedule(model, adapter, [failing_sibling, first, second], monkeypatch)

    assert first.result["passed"] is not True
    assert second.result["passed"] is not True
    assert adapter._lossless_check_expose_global_requests is False
    assert not getattr(adapter, "_lossless_check_request_mode_state", {})
    assert not getattr(adapter, "_lossless_check_forward_hooks", {})
    assert not getattr(adapter, "_lossless_check_mtp_forward_wrappers", {})
