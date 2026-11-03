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

import inspect
from dataclasses import dataclass, replace
from typing import Any, Literal

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.processor.lossless import processor as lossless_check_module
from msmodelslim.processor.lossless.processor import (
    LosslessCheckProcessor,
    LosslessCheckProcessorConfig,
)
from msmodelslim.processor.lossless.protocol import (
    LosslessCheckContext,
    LosslessCheckPlan,
    LosslessCheckTarget,
    StructureIORecord,
)
from msmodelslim.processor.quarot.offline_quarot.quarot import QuaRotProcessor, QuaRotProcessorConfig
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import QuaRotInterface
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


ROTATION = torch.tensor([[0.0, -1.0], [1.0, 0.0]])


def _map_matching_tensors(value: Any, transform):
    if isinstance(value, torch.Tensor):
        return transform(value) if value.is_floating_point() and value.shape[-1] == 2 else value
    if isinstance(value, tuple):
        return tuple(_map_matching_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_matching_tensors(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: _map_matching_tensors(item, transform) for key, item in value.items()}
    return value


class _TinyBlock(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(weight)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)
        self.layers = nn.ModuleList(
            [
                _TinyBlock(torch.tensor([[1.0, 2.0], [0.5, -1.0]])),
                _TinyBlock(torch.tensor([[0.25, 1.5], [-2.0, 0.75]])),
            ]
        )
        self.output_head = _TinyBlock(torch.tensor([[1.0, 0.25], [-0.5, 1.0]]))
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]))

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.output_head(hidden_states)


class _TestRotateProcessorConfig(AutoProcessorConfig):
    type: Literal["test_lossless_rotate"] = "test_lossless_rotate"
    rotate_embedding_in_pre_run: bool = False
    mutate_first_block_in_pre_run: bool = False


class _TestOutputScaleProcessorConfig(AutoProcessorConfig):
    type: Literal["test_lossless_output_scale"] = "test_lossless_output_scale"
    scale: float = 3.0


@dataclass(frozen=True)
class _TestRotateContext(LosslessCheckContext):
    pass


def _linear_targets(request):
    return tuple(
        LosslessCheckTarget(
            name=f"{request.name}.{name}" if request.name and name else request.name or name,
            path=f"{request.name}.{name}" if request.name and name else request.name or name,
            decoder=request.name,
            structure_type=module.__class__.__name__,
        )
        for name, module in request.module.named_modules()
        if isinstance(module, nn.Linear)
    )


@QABCRegistry.register(dispatch_key=_TestRotateProcessorConfig, abc_class=AutoSessionProcessor)
class _TestRotateProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.processed_targets = set()
        self._unsafe_module = None

    def prepare_lossless_check(self):
        return LosslessCheckPlan(expected_targets=())

    def pre_run(self):
        if self.config.rotate_embedding_in_pre_run:
            with torch.no_grad():
                self.model.embed_tokens.weight.copy_(self.model.embed_tokens.weight @ ROTATION)
        if self.config.mutate_first_block_in_pre_run:
            self._unsafe_module = self.model.layers[0]
            with torch.no_grad():
                self._unsafe_module.proj.weight.add_(1.0)

    def create_lossless_check_context(self, request, reference_datas):
        if request.module is self._unsafe_module:
            raise UnsupportedError("request module was modified before lossless reference forward")
        return _TestRotateContext(
            targets=_linear_targets(request),
            reference_datas=_map_matching_tensors(
                reference_datas,
                lambda tensor: tensor.matmul(ROTATION.T),
            ),
            covered_targets=(),
            unresolved_targets=(),
        )

    def align_lossless_check_io(self, context, target, reference, processed):
        del context, target
        aligned_processed = replace(
            processed,
            args=_map_matching_tensors(processed.args, lambda tensor: tensor.matmul(ROTATION.T)),
            kwargs=_map_matching_tensors(processed.kwargs, lambda tensor: tensor.matmul(ROTATION.T)),
            output=_map_matching_tensors(processed.output, lambda tensor: tensor.matmul(ROTATION.T)),
        )
        return reference, aligned_processed

    def process(self, request):
        for target in _linear_targets(request):
            if target.name in self.processed_targets:
                continue
            module = request.module.get_submodule(target.relative_path(request.name))
            with torch.no_grad():
                module.weight.copy_(ROTATION.T @ module.weight @ ROTATION)
            self.processed_targets.add(target.name)
        request.outputs = None


@QABCRegistry.register(dispatch_key=_TestOutputScaleProcessorConfig, abc_class=AutoSessionProcessor)
class _TestOutputScaleProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.processed_targets = set()

    def prepare_lossless_check(self):
        return LosslessCheckPlan(expected_targets=())

    def create_lossless_check_context(self, request, reference_datas):
        return LosslessCheckContext(
            targets=_linear_targets(request),
            reference_datas=reference_datas,
            covered_targets=(),
            unresolved_targets=(),
        )

    def align_lossless_check_io(self, context, target, reference, processed):
        del context, target
        return reference, replace(
            processed,
            output=_map_matching_tensors(
                processed.output,
                lambda tensor: tensor / self.config.scale,
            ),
        )

    def process(self, request):
        for target in _linear_targets(request):
            if target.name in self.processed_targets:
                continue
            module = request.module.get_submodule(target.relative_path(request.name))
            with torch.no_grad():
                module.weight.mul_(self.config.scale)
                if module.bias is not None:
                    module.bias.mul_(self.config.scale)
            self.processed_targets.add(target.name)


class _ForwardCapableAdapter:
    def generate_model_forward(self, _model, _inputs):
        yield


class _WeightedNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.5, 0.75]))

    def forward(self, hidden_states):
        return hidden_states * self.weight


class _FuseBakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.norm = _WeightedNorm()
        self.block.proj = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.block.proj.weight.copy_(torch.tensor([[1.0, 2.0], [-0.5, 1.25]]))
            self.block.proj.bias.copy_(torch.tensor([0.5, -1.0]))

    def forward(self, hidden_states):
        return self.block.proj(self.block.norm(hidden_states))


class _FuseBakeAdapter(QuaRotInterface):
    def get_ln_fuse_map(self):
        return {}, {"block.norm": "block.proj"}

    def get_bake_names(self):
        return [], ["block.proj"]

    def get_rotate_map(self, block_size):
        del block_size
        pair = self.RotatePair(
            left_rot={"block.proj": ROTATION},
            right_rot={"block.proj": ROTATION},
        )
        return [], [pair]


def _build_processor(model, adapter=None, **child_options):
    if adapter is None:
        adapter = _ForwardCapableAdapter()
    return LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(
            configs=[_TestRotateProcessorConfig(**child_options)],
            metrics=["numerical_equivalence"],
        ),
        adapter,
    )


def test_g0_1_two_forwards_preserve_processed_stream_and_release_captures():
    model = _TinyModel()
    reference_model = _TinyModel()
    processor = _build_processor(model)
    processor.pre_run()

    reference_hidden = torch.tensor([[0.75, -1.25]])
    processed_hidden = reference_hidden @ ROTATION
    for index, block in enumerate(model.layers):
        request = BatchProcessRequest(
            name=f"layers.{index}",
            module=block,
            datas=[((processed_hidden,), {})],
            outputs=None,
        )
        processor.process(request)
        processed_hidden = request.outputs[0]
        reference_hidden = reference_model.layers[index](reference_hidden)
        assert torch.allclose(processed_hidden, reference_hidden @ ROTATION)
        assert processor.active_capture_count == 0

    processor.post_run()
    assert processor.result["passed"]


def test_g0_2_pre_run_global_transform_is_aligned_but_request_mutation_is_rejected():
    model = _TinyModel()
    processor = _build_processor(model, rotate_embedding_in_pre_run=True)
    processor.pre_run()
    processed_hidden = model.embed_tokens(torch.tensor([[0, 1]]))
    request = BatchProcessRequest("layers.0", model.layers[0], [((processed_hidden,), {})], None)
    processor.process(request)
    assert processor.result["passed"]

    unsafe_model = _TinyModel()
    unsafe_processor = _build_processor(unsafe_model, mutate_first_block_in_pre_run=True)
    unsafe_processor.pre_run()
    unsafe_request = BatchProcessRequest("layers.0", unsafe_model.layers[0], [((torch.ones(1, 2),), {})], None)
    with pytest.raises(UnsupportedError, match="modified before lossless reference"):
        unsafe_processor.process(unsafe_request)


def test_g0_3_norm_bake_right_and_left_share_one_correct_comparison_space():
    model = _FuseBakeModel()
    processor = QuaRotProcessor(model, QuaRotProcessorConfig(), _FuseBakeAdapter())
    processor.prepare_lossless_check()
    processor.pre_run()
    request = BatchProcessRequest("block", model.block, [((torch.zeros(1, 2),), {})], None)
    context = processor.create_lossless_check_context(request, [((torch.zeros(1, 2),), {})])
    target = next(target for target in context.targets if target.name == "block.proj")

    base_input = torch.tensor([[0.75, -1.25]])
    reference_input = base_input * model.block.norm.weight.detach().clone()
    reference_output = model.block.proj(reference_input)
    reference = StructureIORecord(
        request_name="block",
        structure_name="block.proj",
        invocation=0,
        args=(reference_input,),
        kwargs={},
        output=reference_output,
    )

    processor.process(request)
    processed_input = base_input @ ROTATION
    processed_output = model.block.proj(processed_input)
    processed = replace(reference, args=(processed_input,), output=processed_output)
    assert not torch.allclose(reference.output, processed.output)

    aligned_reference, aligned_processed = processor.align_lossless_check_io(
        context,
        target,
        reference,
        processed,
    )
    assert torch.allclose(aligned_reference.args[0], aligned_processed.args[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(aligned_reference.output, aligned_processed.output, atol=1e-6, rtol=1e-6)


def test_two_protocol_processors_are_rejected_by_single_processor_contract():
    with pytest.raises(SchemaValidateError):
        LosslessCheckProcessorConfig(
            configs=[
                _TestRotateProcessorConfig(),
                _TestOutputScaleProcessorConfig(scale=3.0),
            ],
            metrics=["numerical_equivalence"],
        )


def test_quarot_theoretical_rotation_plan_is_frozen_independently():
    model = _FuseBakeModel()
    processor = QuaRotProcessor(model, QuaRotProcessorConfig(), _FuseBakeAdapter())
    processor.prepare_lossless_check()
    expected = processor._lossless_rotate_commands[0].rot.detach().clone()

    processor.rotate_commands[0].rot.zero_()

    assert torch.equal(processor._lossless_rotate_commands[0].rot, expected)
    assert not torch.equal(processor.rotate_commands[0].rot, expected)


def test_generic_container_has_no_model_specific_structure_or_forward_knowledge():
    source = inspect.getsource(lossless_check_module)

    for forbidden in ("Qwen", "DeepSeek", "MTP", "model.layers", "model.norm", "lm_head"):
        assert forbidden not in source
