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

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import ContextManager, LocalDictContext
from msmodelslim.model.deepseek_v3.model_adapter import DeepSeekV3ModelAdapter
from msmodelslim.processor.lossless.processor import (
    LosslessCheckProcessor,
    LosslessCheckProcessorConfig,
)
from msmodelslim.processor.quarot.offline_quarot.quarot import QuaRotProcessorConfig
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import (
    RotSide,
    RotateCommand,
)
from msmodelslim.utils.exception import UnexpectedError


HIDDEN_SIZE = 32
Q_LORA_RANK = 32
KV_LORA_RANK = 32
QK_NOPE_HEAD_DIM = 16
QK_ROPE_HEAD_DIM = 16
V_HEAD_DIM = 32
VOCAB_SIZE = 64


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.75, 1.25, hidden_size))
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return normalized.to(hidden_states.dtype) * self.weight


class _DeepSeekAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_a_proj = nn.Linear(HIDDEN_SIZE, Q_LORA_RANK, bias=False)
        self.q_a_layernorm = _RMSNorm(Q_LORA_RANK)
        self.q_b_proj = nn.Linear(Q_LORA_RANK, HIDDEN_SIZE, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            HIDDEN_SIZE,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            bias=False,
        )
        self.kv_a_layernorm = _RMSNorm(KV_LORA_RANK)
        self.kv_b_proj = nn.Linear(
            KV_LORA_RANK,
            QK_NOPE_HEAD_DIM + V_HEAD_DIM,
            bias=False,
        )
        self.o_proj = nn.Linear(V_HEAD_DIM, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states):
        query = self.q_a_proj(hidden_states)
        self.q_b_proj(self.q_a_layernorm(query))

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, _key_pe = torch.split(
            compressed_kv,
            [KV_LORA_RANK, QK_ROPE_HEAD_DIM],
            dim=-1,
        )
        key_value = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        _key_nope, value = torch.split(
            key_value,
            [QK_NOPE_HEAD_DIM, V_HEAD_DIM],
            dim=-1,
        )
        return self.o_proj(value)


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 2, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 2, bias=False)
        self.down_proj = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _DenseMLP(_Expert):
    pass


class _MoEMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN_SIZE, 2, bias=False)
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        self.shared_experts = _Expert()

    def forward(self, hidden_states, route_id):
        self.gate(hidden_states)
        expert_index = int(route_id.reshape(-1)[0].item())
        return self.experts[expert_index](hidden_states) + self.shared_experts(hidden_states)


class _SharedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _RMSNorm(HIDDEN_SIZE)
        self.head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, hidden_states):
        return self.head(self.norm(hidden_states))


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, *, use_moe, mtp=False):
        super().__init__()
        self.input_layernorm = _RMSNorm(HIDDEN_SIZE)
        self.self_attn = _DeepSeekAttention()
        self.post_attention_layernorm = _RMSNorm(HIDDEN_SIZE)
        self.mlp = _MoEMLP() if use_moe else _DenseMLP()
        if mtp:
            self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
            self.enorm = _RMSNorm(HIDDEN_SIZE)
            self.hnorm = _RMSNorm(HIDDEN_SIZE)
            self.eh_proj = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE, bias=False)
            self.shared_head = _SharedHead()

    def forward(self, hidden_states, route_id, **_kwargs):
        attention_output = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + attention_output
        mlp_input = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, _MoEMLP):
            mlp_output = self.mlp(mlp_input, route_id)
        else:
            mlp_output = self.mlp(mlp_input)
        hidden_states = hidden_states + mlp_output
        if hasattr(self, "shared_head"):
            self.shared_head(hidden_states)
        return (hidden_states,)


class _ProductionMapTinyDeepSeek(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.model.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(use_moe=False),
                DeepseekV3DecoderLayer(use_moe=True),
                DeepseekV3DecoderLayer(use_moe=True, mtp=True),
            ]
        )
        self.model.norm = _RMSNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, input_ids, route_id, **_kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, route_id=route_id)[0]
        return self.lm_head(self.model.norm(hidden_states))


def _generate_production_map_decoder_layer(_adapter, model):
    return [(f"model.layers.{index}", layer) for index, layer in enumerate(model.model.layers)]


def _preprocess_production_map_mtp(_adapter, model, mtp_decoder, inputs, args, kwargs):
    previous_hidden_states = args[0]
    logits = model.lm_head(model.model.norm(previous_hidden_states)).float()
    input_ids = inputs["input_ids"].clone()
    input_ids[:, -1] = logits[:, -1, :].argmax(dim=-1)
    input_embeds = mtp_decoder.enorm(mtp_decoder.embed_tokens(input_ids))
    hidden_states = mtp_decoder.hnorm(previous_hidden_states)
    hidden_states = mtp_decoder.eh_proj(torch.cat([input_embeds, hidden_states], dim=-1))
    return (hidden_states,), kwargs


def _build_production_map_adapter(config):
    adapter = object.__new__(DeepSeekV3ModelAdapter)
    adapter.config = config
    adapter.generate_decoder_layer = MethodType(_generate_production_map_decoder_layer, adapter)
    object.__setattr__(adapter, "mtp_preprocess", MethodType(_preprocess_production_map_mtp, adapter))
    return adapter


def _build_config():
    return SimpleNamespace(
        num_hidden_layers=3,
        hidden_size=HIDDEN_SIZE,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        first_k_dense_replace=1,
        n_routed_experts=2,
        n_shared_experts=1,
    )


def _build(seed=31):
    torch.manual_seed(seed)
    config = _build_config()
    model = _ProductionMapTinyDeepSeek(config)
    adapter = _build_production_map_adapter(config)
    processor = LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(
            configs=[QuaRotProcessorConfig(block_size=-1, export_extra_info=False)],
            metrics=[{"name": "numerical_equivalence"}],
        ),
        adapter,
    )
    return model, adapter, processor


def _run_lockstep(
    processor,
    adapter,
    model,
    route_ids=(0, 1),
    *,
    remove_target=None,
    remove_side=None,
    remove_pre_run_target=None,
    remove_fused_target=None,
    append_missing_target=None,
):
    processor.pre_run()
    if remove_pre_run_target is not None:
        quarot = processor.processor
        removed = [command for command in quarot._deferred_pre_run_commands if command.target == remove_pre_run_target]
        assert len(removed) == 1
        quarot._deferred_pre_run_commands = [
            command for command in quarot._deferred_pre_run_commands if command.target != remove_pre_run_target
        ]
    if remove_target is not None:
        quarot = processor.processor
        removed = [
            command
            for command in quarot.rotate_commands
            if command.target == remove_target and command.side == remove_side
        ]
        assert len(removed) == 1
        quarot.rotate_commands = [
            command
            for command in quarot.rotate_commands
            if not (command.target == remove_target and command.side == remove_side)
        ]
    if remove_fused_target is not None:
        quarot = processor.processor
        updated_map = {}
        removed = 0
        for source, targets in quarot.fused_map.items():
            if isinstance(targets, (list, tuple)):
                filtered = [target for target in targets if target != remove_fused_target]
                removed += len(targets) - len(filtered)
                if filtered:
                    updated_map[source] = type(targets)(filtered)
            elif targets == remove_fused_target:
                removed += 1
            else:
                updated_map[source] = targets
        assert removed == 1
        quarot.fused_map = updated_map
    if append_missing_target is not None:
        quarot = processor.processor
        quarot.rotate_commands.append(
            RotateCommand(
                target=append_missing_target,
                rot=torch.eye(HIDDEN_SIZE),
                side=RotSide.RIGHT,
            )
        )
    generators = [
        adapter.generate_model_forward(
            model,
            {
                "input_ids": torch.tensor([[2, 4, 0, 8]]),
                "attention_mask": torch.ones(1, 4),
                "route_id": torch.tensor([route_id]),
            },
        )
        for route_id in route_ids
    ]
    responses = [None] * len(generators)

    while True:
        requests = []
        stopped = []
        for generator, response in zip(generators, responses):
            try:
                requests.append(generator.send(response))
                stopped.append(False)
            except StopIteration:
                stopped.append(True)
        if all(stopped):
            break
        assert not any(stopped)
        assert len({request.name for request in requests}) == 1
        assert len({id(request.module) for request in requests}) == 1

        batch = BatchProcessRequest(
            requests[0].name,
            requests[0].module,
            [(request.args, request.kwargs) for request in requests],
            None,
        )
        processor.process(batch)
        responses = list(batch.outputs)

    processor.post_run()


@pytest.mark.parametrize("seed", [31, 37, 41])
def test_production_deepseek_map_full_configuration_covers_dense_moe_mtp_and_both_experts(
    seed,
):
    model, adapter, processor = _build(seed)

    _run_lockstep(processor, adapter, model)

    checked = set(processor.result["checked_targets"])
    assert processor.result["passed"] is True
    assert not processor.result["skipped"]
    assert not processor.result["skipped_targets"]
    assert not processor.result["missing_targets"]
    assert "model.embed_tokens" in checked
    assert "lm_head" in checked
    assert "model.layers.2.embed_tokens" in checked
    assert "model.layers.2.eh_proj" in checked
    assert "model.layers.0.mlp.gate_proj" in checked
    assert "model.layers.1.mlp.gate" in checked
    assert "model.layers.1.mlp.experts.0.gate_proj" in checked
    assert "model.layers.1.mlp.experts.1.gate_proj" in checked
    assert "model.layers.1.mlp.shared_experts.down_proj" in checked
    assert "model.layers.2.shared_head.head" in checked


@pytest.mark.parametrize(
    ("target", "side", "expected_structure", "expected_side", "expected_reason"),
    [
        (
            "lm_head",
            RotSide.RIGHT,
            "lm_head",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.2.eh_proj",
            RotSide.RIGHT,
            "model.layers.2.eh_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.2.eh_proj",
            RotSide.LEFT,
            "model.layers.2.eh_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.q_a_proj",
            RotSide.RIGHT,
            "model.layers.0.self_attn.q_a_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.q_a_proj",
            RotSide.LEFT,
            "model.layers.0.self_attn.q_a_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.q_b_proj",
            RotSide.RIGHT,
            "model.layers.0.self_attn.q_b_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
            RotSide.RIGHT,
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
            RotSide.LEFT,
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.kv_b_proj",
            RotSide.RIGHT,
            "model.layers.0.self_attn.kv_b_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.kv_b_proj",
            RotSide.LEFT,
            "model.layers.0.self_attn.kv_b_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.o_proj",
            RotSide.RIGHT,
            "model.layers.0.self_attn.o_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.self_attn.o_proj",
            RotSide.LEFT,
            "model.layers.0.self_attn.o_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.mlp.gate_proj",
            RotSide.RIGHT,
            "model.layers.0.mlp.gate_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.mlp.up_proj",
            RotSide.RIGHT,
            "model.layers.0.mlp.up_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.0.mlp.down_proj",
            RotSide.LEFT,
            "model.layers.0.mlp.down_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.1.mlp.gate",
            RotSide.RIGHT,
            "model.layers.1.mlp.gate",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.1.mlp.experts.0.gate_proj",
            RotSide.RIGHT,
            "model.layers.1.mlp.experts.0.gate_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.1.mlp.experts.0.down_proj",
            RotSide.LEFT,
            "model.layers.1.mlp.experts.0.down_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.1.mlp.experts.1.up_proj",
            RotSide.RIGHT,
            "model.layers.1.mlp.experts.1.up_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.1.mlp.shared_experts.up_proj",
            RotSide.RIGHT,
            "model.layers.1.mlp.shared_experts.up_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.1.mlp.shared_experts.down_proj",
            RotSide.LEFT,
            "model.layers.1.mlp.shared_experts.down_proj",
            "output",
            "metric_threshold_exceeded",
        ),
        (
            "model.layers.2.shared_head.head",
            RotSide.RIGHT,
            "model.layers.2.shared_head.head",
            "output",
            "metric_threshold_exceeded",
        ),
    ],
)
@pytest.mark.parametrize("seed", [31, 37, 41])
def test_deleted_command_reports_exact_first_causal_structure(
    target,
    side,
    expected_structure,
    expected_side,
    expected_reason,
    seed,
):
    model, adapter, processor = _build(seed)

    with pytest.raises(UnexpectedError):
        _run_lockstep(
            processor,
            adapter,
            model,
            remove_target=target,
            remove_side=side,
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["structure"] == expected_structure
    assert anomaly["side"] == expected_side
    assert anomaly["reason"] == expected_reason


@pytest.mark.parametrize("seed", [31, 37, 41])
def test_deleted_global_embedding_rotation_reports_embedding_output(seed):
    model, adapter, processor = _build(seed)

    with pytest.raises(UnexpectedError):
        _run_lockstep(
            processor,
            adapter,
            model,
            remove_pre_run_target="model.embed_tokens",
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["structure"] == "model.embed_tokens"
    assert anomaly["side"] == "output"
    assert anomaly["reason"] == "metric_threshold_exceeded"


@pytest.mark.parametrize(
    ("target", "expected_side"),
    [
        ("lm_head", "input"),
        ("model.layers.2.eh_proj", "input"),
        ("model.layers.0.self_attn.q_a_proj", "output"),
        ("model.layers.0.self_attn.q_b_proj", "input"),
        ("model.layers.0.self_attn.kv_b_proj", "input"),
        ("model.layers.0.mlp.gate_proj", "output"),
        ("model.layers.1.mlp.experts.0.gate_proj", "output"),
        ("model.layers.2.shared_head.head", "input"),
    ],
)
@pytest.mark.parametrize("seed", [31, 37, 41])
def test_deleted_norm_fusion_reports_exact_first_causal_structure(target, expected_side, seed):
    model, adapter, processor = _build(seed)

    with pytest.raises(UnexpectedError):
        _run_lockstep(
            processor,
            adapter,
            model,
            remove_fused_target=target,
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["structure"] == target
    assert anomaly["side"] == expected_side
    assert anomaly["reason"] == "metric_threshold_exceeded"


@pytest.mark.parametrize("seed", [31, 37, 41])
def test_single_route_reports_inactive_expert_as_skipped_not_passed(seed):
    model, adapter, processor = _build(seed)

    _run_lockstep(processor, adapter, model, route_ids=(0,))

    assert processor.result["passed"] is None
    assert "model.layers.1.mlp.experts.1.gate_proj" in processor.result["skipped_targets"]
    assert "model.layers.2.mlp.experts.1.up_proj" in processor.result["skipped_targets"]
    assert not processor.result["anomalies"]


@pytest.mark.parametrize("seed", [31, 37, 41])
def test_nonexistent_expert_target_is_structure_not_found_not_unexecuted(seed):
    model, adapter, processor = _build(seed)
    missing_target = "model.layers.1.mlp.experts.99.gate_proj"

    with pytest.raises(UnexpectedError):
        _run_lockstep(
            processor,
            adapter,
            model,
            append_missing_target=missing_target,
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["structure"] == missing_target
    assert anomaly["side"] == "coverage"
    assert anomaly["reason"] == "structure_not_found"
    assert processor.result["missing_targets"] == [missing_target]


def test_full_configuration_twenty_independent_rounds_release_runtime_state():
    expected_check_count = None
    for seed in range(101, 121):
        model, adapter, processor = _build(seed)
        context = LocalDictContext(enable_debug=True)

        with ContextManager(context):
            _run_lockstep(processor, adapter, model)

        result = context["lossless_check"].state["result"]
        assert result == processor.result
        assert result["passed"] is True
        assert not result["skipped"]
        assert not result["skipped_targets"]
        assert {"lm_head", "model.embed_tokens", "model.layers.2.eh_proj"}.issubset(set(result["checked_targets"]))
        assert result["anomalies"] == []
        assert result["missing_targets"] == []
        assert processor.active_capture_count == 0
        if expected_check_count is None:
            expected_check_count = len(result["checks"])
        assert len(result["checks"]) == expected_check_count
