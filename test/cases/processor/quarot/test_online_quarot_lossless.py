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

import torch
import pytest
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.quarot import OnlineRotationInputHookIR, OnlineRotationInfo, OnlineRotationWrapper, RotationType
from msmodelslim.processor.lossless.processor import LosslessCheckProcessor, LosslessCheckProcessorConfig
from msmodelslim.processor.lossless.protocol import (
    LosslessCheckContext,
    LosslessCheckableProcessor,
    StructureIORecord,
)
from msmodelslim.processor.quarot.online_quarot.online_quarot import (
    OnlineQuaRotProcessor,
    OnlineQuaRotProcessorConfig,
)
from msmodelslim.processor.quarot.online_quarot.online_quarot_interface import (
    OnlineQuaRotInterface,
    RotationConfig,
)


class _Adapter(OnlineQuaRotInterface):
    def __init__(self, rotation_configs):
        self.rotation_configs = rotation_configs

    def get_online_rotation_configs(self, model=None):
        return self.rotation_configs

    def generate_model_forward(self, model):
        return model


class _RotateModel(nn.Module):
    def __init__(self, rotate):
        super().__init__()
        self.rotate = rotate

    def forward(self, value):
        return self.rotate(value)


def _rotation_matrix():
    return torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )


def _request(model, value=None):
    value = torch.arange(4, dtype=torch.float32).reshape(1, 4) if value is None else value
    return BatchProcessRequest(name="", module=model, datas=[((value,), {})], outputs=None)


def _record(name, args, output):
    return StructureIORecord(
        request_name="",
        structure_name=name,
        invocation=0,
        args=args,
        kwargs={},
        output=output,
    )


def test_online_quarot_declares_protocol_and_freezes_theoretical_rotation():
    matrix = _rotation_matrix()
    model = nn.Module()
    model.replace = nn.Identity()
    config = OnlineQuaRotProcessorConfig()
    adapter = _Adapter(
        {
            "replace": RotationConfig(rotation_type="replace", rotation_matrix=matrix),
        }
    )
    processor = OnlineQuaRotProcessor(model, config, adapter)

    assert isinstance(processor, LosslessCheckableProcessor)
    plan = processor.prepare_lossless_check()
    assert plan.expected_targets == ("replace",)
    assert isinstance(processor._lossless_rotation_infos, dict)
    frozen_matrix = processor._lossless_rotation_infos["replace"].rotation_matrix.clone()
    assert isinstance(model.replace, nn.Identity)

    processor.pre_run()
    processor.rotation_infos["replace"].rotation_matrix.copy_(torch.eye(4))
    context = processor.create_lossless_check_context(_request(model), [((torch.zeros(1, 4),), {})])

    assert isinstance(context, LosslessCheckContext)
    assert context.rotation_plans[0].rotation_matrix.equal(frozen_matrix)
    assert context.targets[0].structure_type == "Identity"
    assert context.covered_targets == ("replace",)
    assert not context.unresolved_targets

    value = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    reference = _record("replace", (value,), value)
    corrupted_processed = _record("replace", (value,), value)
    _, aligned = processor.align_lossless_check_io(context, context.targets[0], reference, corrupted_processed)
    assert not torch.allclose(aligned.output, reference.output)


def test_online_quarot_aligns_each_rotation_type_in_its_own_io_space():
    matrix = _rotation_matrix()
    model = nn.Module()
    model.input = nn.Identity()
    model.output = nn.Identity()
    model.replace = nn.Identity()
    model.offline_right = nn.Linear(4, 4, bias=False)
    model.offline_left = nn.Linear(4, 4, bias=False)
    model.unrotated = nn.Linear(4, 4, bias=False)
    adapter = _Adapter(
        {
            "input": RotationConfig(rotation_type="input", rotation_matrix=matrix),
            "output": RotationConfig(rotation_type="output", rotation_matrix=matrix),
            "replace": RotationConfig(rotation_type="replace", rotation_matrix=matrix),
            "offline_right": RotationConfig(
                rotation_type="offline",
                rotation_matrix=matrix,
                rotation_side="right",
            ),
            "offline_left": RotationConfig(
                rotation_type="offline",
                rotation_matrix=matrix,
                rotation_side="left",
            ),
        }
    )
    processor = OnlineQuaRotProcessor(model, OnlineQuaRotProcessorConfig(), adapter)
    processor.prepare_lossless_check()
    processor.pre_run()
    context = processor.create_lossless_check_context(_request(model), None)
    targets = {target.name: target for target in context.targets}
    plans = {plan.name: plan for plan in context.rotation_plans}
    assert set(targets) == {
        "input",
        "output",
        "replace",
        "offline_right",
        "offline_left",
        "unrotated",
    }
    value = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    cases = {
        "input": ((value @ matrix.T,), value),
        "output": ((value,), value @ matrix.T),
        "replace": ((value,), value @ matrix.T),
        "offline_right": ((value @ matrix,), value),
        "offline_left": ((value,), value @ matrix),
    }
    for name, (processed_args, processed_output) in cases.items():
        reference = _record(name, (value,), value)
        processed = _record(name, processed_args, processed_output)
        aligned_reference, aligned_processed = processor.align_lossless_check_io(
            context,
            targets[name],
            reference,
            processed,
        )
        assert torch.allclose(aligned_processed.args[0], aligned_reference.args[0])
        assert torch.allclose(aligned_processed.output, aligned_reference.output)
        assert plans[name].rotation_type == RotationType(adapter.rotation_configs[name].rotation_type)

    reference = _record("unrotated", (value,), value)
    processed = _record("unrotated", (value,), value + 1)
    aligned_reference, aligned_processed = processor.align_lossless_check_io(
        context,
        targets["unrotated"],
        reference,
        processed,
    )
    assert aligned_reference is reference
    assert aligned_processed is processed


def test_online_quarot_alignment_uses_nonsymmetric_rotation_direction():
    matrix = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    model = nn.Module()
    model.output = nn.Identity()
    processor = OnlineQuaRotProcessor(
        model,
        OnlineQuaRotProcessorConfig(),
        _Adapter({"output": RotationConfig(rotation_type="output", rotation_matrix=matrix)}),
    )
    processor.prepare_lossless_check()
    processor.pre_run()
    context = processor.create_lossless_check_context(_request(model), None)
    target = context.targets[0]
    value = torch.tensor([[1.0, 2.0, 3.0, 5.0]])
    reference = _record("output", (value,), value)
    processed = _record("output", (value,), value @ matrix.T)

    aligned_reference, aligned_processed = processor.align_lossless_check_io(
        context,
        target,
        reference,
        processed,
    )

    assert torch.allclose(aligned_processed.output, aligned_reference.output)
    assert not torch.allclose(value @ matrix.T @ matrix.T, value)


def test_online_quarot_include_exclude_and_repeated_preprocess_are_idempotent():
    matrix = _rotation_matrix()
    model = nn.Module()
    model.rotate = nn.Identity()
    model.exclude = nn.Identity()
    model.unrotated = nn.Linear(4, 4, bias=False)
    adapter = _Adapter(
        {
            "rotate": RotationConfig(rotation_type="replace", rotation_matrix=matrix),
            "exclude": RotationConfig(rotation_type="replace", rotation_matrix=matrix),
        }
    )
    processor = OnlineQuaRotProcessor(
        model,
        OnlineQuaRotProcessorConfig(include=["rotate", "exclude"], exclude=["exclude"]),
        adapter,
    )

    assert processor.prepare_lossless_check().expected_targets == ("rotate",)
    processor.pre_run()
    request = _request(model)
    context = processor.create_lossless_check_context(request, None)
    assert {target.name for target in context.targets} == {"rotate", "unrotated"}
    processor.preprocess(request)
    first_wrapper = model.rotate
    processor._lossless_applied_targets.clear()
    processor.preprocess(request)

    assert isinstance(first_wrapper, OnlineRotationWrapper)
    assert model.rotate is first_wrapper
    assert isinstance(model.exclude, nn.Identity)
    processor.abort_lossless_check()
    assert processor._lossless_rotation_infos is None
    assert not processor._lossless_applied_targets


def test_online_quarot_rejects_existing_wrapper_with_different_matrix_in_context():
    model = nn.Module()
    old_info = OnlineRotationInfo(torch.eye(4), RotationType.REPLACE, "rotate")
    model.rotate = OnlineRotationWrapper(nn.Identity(), "rotate", old_info)
    requested_matrix = _rotation_matrix()
    processor = OnlineQuaRotProcessor(
        model,
        OnlineQuaRotProcessorConfig(),
        _Adapter({"rotate": RotationConfig(rotation_type="replace", rotation_matrix=requested_matrix)}),
    )
    processor.prepare_lossless_check()
    processor.pre_run()

    with pytest.raises(ValueError, match="rotate"):
        processor.create_lossless_check_context(_request(model), None)


def test_online_quarot_rejects_existing_hook_with_different_matrix_in_preprocess():
    model = nn.Module()
    model.rotate = nn.Identity()
    old_info = OnlineRotationInfo(torch.eye(4), RotationType.INPUT, "rotate")
    hook = OnlineRotationInputHookIR("rotate", old_info)
    model.rotate.register_forward_pre_hook(hook)
    requested_matrix = _rotation_matrix()
    processor = OnlineQuaRotProcessor(
        model,
        OnlineQuaRotProcessorConfig(),
        _Adapter({"rotate": RotationConfig(rotation_type="input", rotation_matrix=requested_matrix)}),
    )
    processor.prepare_lossless_check()
    processor.pre_run()

    with pytest.raises(ValueError, match="rotate"):
        processor.preprocess(_request(model))


def test_online_quarot_rejects_existing_offline_rotation_with_different_side():
    matrix = _rotation_matrix()
    model = nn.Module()
    model.offline = nn.Linear(4, 4, bias=False)
    old_info = OnlineRotationInfo(matrix.clone(), RotationType.OFFLINE, "offline", rotation_side="right")
    model.offline.rotation_info = old_info
    model.offline.layer_name = "offline"
    processor = OnlineQuaRotProcessor(
        model,
        OnlineQuaRotProcessorConfig(),
        _Adapter(
            {
                "offline": RotationConfig(
                    rotation_type="offline",
                    rotation_matrix=matrix,
                    rotation_side="left",
                )
            }
        ),
    )
    processor.prepare_lossless_check()
    processor.pre_run()

    with pytest.raises(ValueError, match="offline"):
        processor.create_lossless_check_context(_request(model), None)


def test_online_quarot_lossless_container_fails_for_existing_wrapper_with_different_matrix():
    old_info = OnlineRotationInfo(torch.eye(4), RotationType.REPLACE, "rotate")
    model = _RotateModel(OnlineRotationWrapper(nn.Identity(), "rotate", old_info))
    config = LosslessCheckProcessorConfig(
        configs=[OnlineQuaRotProcessorConfig()],
        metrics=["numerical_equivalence"],
    )
    processor = LosslessCheckProcessor(
        model,
        config,
        _Adapter({"rotate": RotationConfig(rotation_type="replace", rotation_matrix=_rotation_matrix())}),
    )
    processor.pre_run()

    with pytest.raises(ValueError, match="rotate"):
        processor.process(_request(model))
    assert processor.result["passed"] is False
