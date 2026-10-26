#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

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

from msmodelslim.core.convert.types import IRKind, LossLevel, SourceIR, TensorRef, TensorRole


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        (IRKind.FLOAT, "FLOAT"),
        (IRKind.FP8_BLOCK, "FP8_BLOCK"),
        (IRKind.W8A8_MXFP8, "W8A8_MXFP8"),
        (TensorRole.MAIN_WEIGHT, "main_weight"),
        (TensorRole.GLOBAL_SCALE, "global_scale"),
        (LossLevel.LOSSLESS, "lossless"),
        (LossLevel.LOSSY, "lossy"),
    ],
)
def test_enum_members_have_expected_values(member, expected):
    assert member.value == expected


def test_tensor_ref_defaults_and_metadata_are_independent():
    first = TensorRef(logical_name="weight", key="layer.weight", shard="model-0001", dtype="bf16", shape=(2, 3))
    second = TensorRef(logical_name="bias", key="layer.bias", shard="model-0001", dtype="bf16", shape=(3,))

    assert first.role is None
    assert first.meta == {}
    first.meta["origin"] = "checkpoint"
    assert second.meta == {}


def test_source_ir_has_expected_defaults():
    source = SourceIR(kind=IRKind.FLOAT)

    assert source.source_format is None
    assert source.confidence == 1.0
    assert source.evidence == []
