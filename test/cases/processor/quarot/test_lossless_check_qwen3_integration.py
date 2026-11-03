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

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from msmodelslim.core.base.protocol import BatchProcessRequest, DataUnit
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.generated_runner import KEY_DATA_LOADER, GeneratedRunner
from msmodelslim.model.qwen3.model_adapter import Qwen3ModelAdapter
from msmodelslim.processor.lossless.processor import (
    LosslessCheckProcessor,
    LosslessCheckProcessorConfig,
)
from msmodelslim.processor.quarot.offline_quarot.quarot import QuaRotProcessorConfig
from msmodelslim.utils.cache.memory import clear_cache
from msmodelslim.utils.exception import (
    InvalidDatasetError,
    UnexpectedError,
    UnsupportedError,
)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return normalized * self.weight


class _TinyAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        # q/k calls are retained so the self-check observes the complete adapter map.
        self.q_proj(hidden_states)
        self.k_proj(hidden_states)
        return self.o_proj(self.v_proj(hidden_states))


class _TinyMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size=4, intermediate_size=8):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden_size)
        self.self_attn = _TinyAttention(hidden_size)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size)
        self.mlp = _TinyMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, **_kwargs):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return (hidden_states,)


class _TinyQwen3(nn.Module):
    def __init__(self, num_hidden_layers=2, hidden_size=4, vocab_size=16):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([Qwen3DecoderLayer(hidden_size) for _ in range(num_hidden_layers)])
        self.model.norm = Qwen3RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, **_kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return self.lm_head(self.model.norm(hidden_states))


def _build_tiny_qwen3_adapter(model):
    adapter = object.__new__(Qwen3ModelAdapter)
    adapter.config = SimpleNamespace(
        num_hidden_layers=len(model.model.layers),
        hidden_size=model.model.embed_tokens.embedding_dim,
        head_dim=model.model.embed_tokens.embedding_dim,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    return adapter


def _record_dataset(adapter, dataset, device=DeviceType.NPU):
    adapter.dataset_calls.append((tuple(tuple(sample) for sample in dataset), device))
    return [{"input_ids": torch.tensor([sample], dtype=torch.long)} for sample in dataset]


def _record_kv_cache(adapter, _model, need_kv_cache):
    adapter.kv_cache_calls.append(need_kv_cache)


def _build_dataset_qwen3_adapter(model):
    adapter = _build_tiny_qwen3_adapter(model)
    adapter.dataset_calls = []
    adapter.kv_cache_calls = []
    adapter.handle_dataset = MethodType(_record_dataset, adapter)
    adapter.enable_kv_cache = MethodType(_record_kv_cache, adapter)
    return adapter


class _RecordingGeneratedRunner(GeneratedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = []

    def build_process_unit(self, *args, **kwargs):
        self.units = super().build_process_unit(*args, **kwargs)
        return self.units


def _build(model, adapter):
    return LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(
            configs=[QuaRotProcessorConfig(block_size=-1, export_extra_info=False)],
            metrics=[{"name": "numerical_equivalence"}],
        ),
        adapter,
    )


def _run(processor, adapter, model, *, remove_command=None):
    processor.pre_run()
    quarot = processor.processor
    if remove_command is not None:
        quarot.rotate_commands = [command for command in quarot.rotate_commands if command.target != remove_command]

    generator = adapter.generate_model_forward(
        model,
        {"input_ids": torch.tensor([[1, 3, 5, 7]])},
    )
    response = None
    while True:
        try:
            request = generator.send(response)
        except StopIteration:
            break
        batch = BatchProcessRequest(
            request.name,
            request.module,
            [(request.args, request.kwargs)],
            None,
        )
        processor.process(batch)
        response = batch.outputs[0]
    processor.post_run()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen3_tiny_checks_existing_adapter_requests_and_reports_unexposed_global_targets(dtype):
    torch.manual_seed(7)
    model = _TinyQwen3().to(dtype=dtype)
    adapter = _build_tiny_qwen3_adapter(model)
    processor = _build(model, adapter)

    _run(processor, adapter, model)

    checked = {check["structure"] for check in processor.result["checks"]}
    assert processor.result["passed"] is None
    assert processor.result["skipped"]
    assert processor.result["skip_reason"] == "target_not_observed"
    assert processor.result["skipped_targets"] == [
        "lm_head",
        "model.embed_tokens",
    ]
    assert "model.layers.0.self_attn.q_proj" in checked
    assert "model.layers.1.mlp.down_proj" in checked
    assert "lm_head" not in checked
    assert any(check["reference_dtype"] == str(dtype) for check in processor.result["checks"])


def test_qwen3_tiny_deleted_real_right_command_fails_at_q_proj_output():
    torch.manual_seed(7)
    model = _TinyQwen3()
    adapter = _build_tiny_qwen3_adapter(model)
    processor = _build(model, adapter)

    with pytest.raises(UnexpectedError, match="model.layers.0.self_attn.q_proj"):
        _run(
            processor,
            adapter,
            model,
            remove_command="model.layers.0.self_attn.q_proj",
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["structure"] == "model.layers.0.self_attn.q_proj"
    assert anomaly["side"] == "output"
    assert anomaly["reason"] == "metric_threshold_exceeded"


def test_lossless_check_rejects_online_quarot_before_processing_data():
    model = _TinyQwen3()
    adapter = _build_tiny_qwen3_adapter(model)
    processor = LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(
            configs=[
                QuaRotProcessorConfig(
                    online=True,
                    block_size=-1,
                    export_extra_info=False,
                )
            ],
            metrics=[{"name": "numerical_equivalence"}],
        ),
        adapter,
    )

    with pytest.raises(UnsupportedError, match="only supports offline QuaRot"):
        processor.pre_run()


def test_qwen3_two_samples_run_through_standard_dataset_and_generated_runner(
    monkeypatch,
):
    clear_cache(KEY_DATA_LOADER)
    try:
        monkeypatch.delattr(torch, "npu", raising=False)
        torch.manual_seed(7)
        model = _TinyQwen3()
        adapter = _build_dataset_qwen3_adapter(model)
        runner = _RecordingGeneratedRunner(adapter)
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

        processor = runner.units[0].processor
        q_proj_checks = [
            check
            for check in processor.result["checks"]
            if check["structure"] == "model.layers.0.self_attn.q_proj" and check["side"] == "input"
        ]
        assert processor.result["passed"] is None
        assert processor.result["skipped_targets"] == [
            "lm_head",
            "model.embed_tokens",
        ]
        assert {check["invocation"] for check in q_proj_checks} == {0, 1}
        assert adapter.dataset_calls == [(((1, 3, 5, 7), (2, 4, 6, 8)), DeviceType.CPU)]
        assert adapter.kv_cache_calls == [False]
    finally:
        clear_cache(KEY_DATA_LOADER)


def test_generated_process_unit_rejects_missing_dataset_for_lossless_check():
    model = _TinyQwen3()
    adapter = _build_dataset_qwen3_adapter(model)
    runner = GeneratedRunner(adapter)
    units = runner.build_process_unit(
        [
            LosslessCheckProcessorConfig(
                configs=[QuaRotProcessorConfig(block_size=-1, export_extra_info=False)],
                metrics=[{"name": "numerical_equivalence"}],
            )
        ],
        model,
        adapter,
        DataUnit(None, None),
        calib_data=None,
    )

    with pytest.raises(InvalidDatasetError, match="Calib data is needed"):
        units[0].init_generators()
