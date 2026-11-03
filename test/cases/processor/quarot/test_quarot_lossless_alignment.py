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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.lossless.protocol import StructureIORecord
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import (
    RotSide,
    RotateCommand,
)
from msmodelslim.processor.quarot.offline_quarot.quarot_lossless import (
    align_quarot_lossless_check_io,
    create_quarot_lossless_check_context,
)
from msmodelslim.processor.quarot.offline_quarot.quarot_lossless import (
    _align_baked_tree,
)


R90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
REFLECT = torch.tensor([[1.0, 0.0], [0.0, -1.0]])


class _Scope(nn.Module):
    def __init__(self, in_features=2, out_features=2, with_norm=False):
        super().__init__()
        if with_norm:
            self.norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, hidden_states):
        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)
        return self.linear(hidden_states)


class _Model(nn.Module):
    def __init__(self, in_features=2, out_features=2, with_norm=False):
        super().__init__()
        self.block = _Scope(in_features, out_features, with_norm)

    def forward(self, hidden_states):
        return self.block(hidden_states)


def _context(
    *,
    model=None,
    theoretical_commands=(),
    fused_map=None,
    bake_names=(),
    global_rotation=None,
    reference_datas=None,
):
    model = model or _Model()
    processor = SimpleNamespace(
        model=model,
        _lossless_pre_run_commands=[],
        _lossless_rotate_commands=list(theoretical_commands),
        _lossless_pre_run_fused_map={},
        _lossless_fused_map=dict(fused_map or {}),
        _lossless_pre_run_bake_names=[],
        _lossless_bake_names=list(bake_names),
        _lossless_global_rotation=global_rotation,
        fused_map={},
        bake_names=[],
        rotate_commands=[],
    )
    request = BatchProcessRequest("block", model.block, [((torch.zeros(1, 2),), {})], None)
    if reference_datas is None:
        reference_datas = request.datas
    return create_quarot_lossless_check_context(processor, request, reference_datas)


def _record(args, output):
    return StructureIORecord("block", "block.linear", 0, (args,), {}, output)


def _align(
    context,
    reference_input,
    processed_input,
    reference_output=None,
    processed_output=None,
):
    target = next(target for target in context.targets if target.name == "block.linear")
    if reference_output is None:
        reference_output = torch.zeros_like(reference_input)
    if processed_output is None:
        processed_output = torch.zeros_like(reference_output)
    return align_quarot_lossless_check_io(
        context,
        target,
        _record(reference_input, reference_output),
        _record(processed_input, processed_output),
    )


def test_right_rotation_uses_transpose_for_non_symmetric_orthogonal_matrix():
    command = RotateCommand("block.linear", R90, RotSide.RIGHT)
    context = _context(theoretical_commands=[command])
    reference_input = torch.tensor([[1.5, -0.25]])
    processed_input = reference_input @ R90

    reference, processed = _align(context, reference_input, processed_input)

    assert torch.allclose(reference.args[0], processed.args[0])


def test_left_rotation_restores_structure_output():
    command = RotateCommand("block.linear", R90, RotSide.LEFT)
    context = _context(theoretical_commands=[command])
    reference_output = torch.tensor([[0.75, -2.0]])
    processed_output = reference_output @ R90

    reference, processed = _align(
        context,
        torch.ones(1, 2),
        torch.ones(1, 2),
        reference_output,
        processed_output,
    )

    assert torch.allclose(reference.output, processed.output)


def test_multiple_right_and_left_actions_are_reversed_in_strict_reverse_order():
    commands = [
        RotateCommand("block.linear", R90, RotSide.RIGHT),
        RotateCommand("block.linear", REFLECT, RotSide.RIGHT),
        RotateCommand("block.linear", REFLECT, RotSide.LEFT),
        RotateCommand("block.linear", R90, RotSide.LEFT),
    ]
    context = _context(theoretical_commands=commands)
    reference_input = torch.tensor([[2.0, -1.0]])
    reference_output = torch.tensor([[0.25, 3.0]])
    processed_input = reference_input @ R90 @ REFLECT
    processed_output = reference_output @ REFLECT @ R90

    reference, processed = _align(
        context,
        reference_input,
        processed_input,
        reference_output,
        processed_output,
    )

    assert torch.allclose(reference.args[0], processed.args[0])
    assert torch.allclose(reference.output, processed.output)


@pytest.mark.parametrize(
    ("rotation", "expanded"),
    [
        ([R90, REFLECT], torch.block_diag(R90, REFLECT)),
        (R90, torch.block_diag(R90, R90)),
    ],
)
def test_split_and_repeated_block_rotation_follow_formal_dimension_rules(rotation, expanded):
    command = RotateCommand("block.linear", rotation, RotSide.RIGHT)
    context = _context(model=_Model(4, 2), theoretical_commands=[command])
    reference_input = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    processed_input = reference_input @ expanded

    reference, processed = _align(context, reference_input, processed_input)

    assert torch.allclose(reference.args[0], processed.args[0])


def test_rectangular_semi_orthogonal_right_rotation_uses_processed_dimension():
    rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    command = RotateCommand("block.linear", rotation, RotSide.RIGHT)
    context = _context(model=_Model(3, 2), theoretical_commands=[command])
    reference_input = torch.tensor([[1.25, -0.75]])
    processed_input = reference_input @ rotation

    reference, processed = _align(context, reference_input, processed_input)

    assert processed.args[0].shape[-1] == 2
    assert torch.allclose(reference.args[0], processed.args[0])


def test_norm_fuse_restores_original_gamma_and_beta_input_space():
    model = _Model(with_norm=True)
    with torch.no_grad():
        model.block.norm.weight.copy_(torch.tensor([1.5, 0.75]))
        model.block.norm.bias.copy_(torch.tensor([0.25, -0.5]))
    context = _context(model=model, fused_map={"block.norm": "block.linear"})
    normalized_input = torch.tensor([[0.5, -1.25]])
    reference_input = normalized_input * model.block.norm.weight + model.block.norm.bias

    reference, processed = _align(context, reference_input, normalized_input)

    assert torch.allclose(reference.args[0], processed.args[0])


def test_bake_mean_aligns_both_outputs_to_zero_mean_common_space():
    context = _context(bake_names=["block.linear"])
    reference_output = torch.tensor([[1.0, 4.0]])
    processed_output = reference_output - reference_output.mean(dim=-1, keepdim=True)

    reference, processed = _align(
        context,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        reference_output,
        processed_output,
    )

    assert torch.allclose(reference.output, processed.output)


def test_global_embedding_rotation_restores_request_data():
    reference_hidden = torch.tensor([[0.5, -2.0]])
    processed_hidden = reference_hidden @ R90
    context = _context(
        global_rotation=R90,
        reference_datas=[((processed_hidden,), {})],
    )

    assert torch.allclose(context.reference_datas[0][0][0], reference_hidden)


def test_global_embedding_rotation_restores_nested_multimodal_inputs():
    reference_main = torch.tensor([[0.5, -2.0]])
    reference_deepstack = [
        torch.tensor([[1.5, 0.25]]),
        torch.tensor([[-0.75, 3.0]]),
    ]
    processed_datas = [
        (
            (
                reference_main @ R90,
                [hidden @ R90 for hidden in reference_deepstack],
            ),
            {},
        )
    ]
    context = _context(
        global_rotation=R90,
        reference_datas=processed_datas,
    )
    restored_main, restored_deepstack = context.reference_datas[0][0]

    assert torch.allclose(restored_main, reference_main)
    assert torch.allclose(restored_deepstack[0], reference_deepstack[0])
    assert torch.allclose(restored_deepstack[1], reference_deepstack[1])


def test_global_embedding_rotation_recurses_through_mapping_and_ignores_scalar():
    reference_hidden = torch.tensor([[0.5, -2.0]])
    scalar = torch.tensor(1.0)
    context = _context(
        global_rotation=R90,
        reference_datas={
            "nested": {"hidden_states": [reference_hidden @ R90]},
            "loss": scalar,
        },
    )
    restored = context.reference_datas

    assert torch.allclose(restored["nested"]["hidden_states"][0], reference_hidden)
    assert torch.equal(restored["loss"], scalar)


def test_theoretical_command_still_exposes_missing_execution_as_mismatch():
    command = RotateCommand("block.linear", R90, RotSide.RIGHT)
    context = _context(theoretical_commands=[command])
    reference_input = torch.tensor([[1.0, 2.0]])

    reference, processed = _align(context, reference_input, reference_input.clone())

    assert not torch.allclose(reference.args[0], processed.args[0])


def test_rotation_dimension_mismatch_is_not_silently_ignored():
    command = RotateCommand("block.linear", R90, RotSide.RIGHT)
    context = _context(theoretical_commands=[command])
    reference_input = torch.ones(1, 3)

    with pytest.raises(ValueError, match="hidden dim"):
        _align(context, reference_input, reference_input.clone())


@pytest.mark.parametrize(
    ("reference", "processed"),
    [
        ((torch.ones(1),), (torch.ones(1), torch.ones(1))),
        ([torch.ones(1)], [torch.ones(1), torch.ones(1)]),
        ({"left": torch.ones(1)}, {"right": torch.ones(1)}),
    ],
)
def test_baked_tree_structure_mismatch_is_not_truncated(reference, processed):
    with pytest.raises(ValueError, match="structure"):
        _align_baked_tree(reference, processed)
