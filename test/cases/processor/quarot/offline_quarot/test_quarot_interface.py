#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.processor.quarot.offline_quarot.quarot_interface 模块的单元测试
"""

import pytest

from msmodelslim.processor.quarot.common.quarot_utils import GLOBAL_DTYPE, QuaRotMode
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import (
    LAOSOnlineRotationInterface,
    QuaRotInterface,
    RotateCommand,
    RotatePair,
    RotSide,
    get_rotate_command,
)


class _LAOSAdapter(LAOSOnlineRotationInterface):
    def get_head_dim(self):
        return 8

    def get_num_attention_heads(self):
        return 4

    def get_layer_wise_ov_pair(self, decoder_module):
        return {}

    def get_layer_wise_up_down_pair(self, decoder_module):
        return {}


class TestRotSide:
    def test_enum_wire_values(self):
        assert RotSide.LEFT.value == "left"
        assert RotSide.RIGHT.value == "right"


class TestRotateDataclasses:
    def test_rotate_command_holds_fields(self):
        command = RotateCommand(target="a", rot="r", side=RotSide.LEFT)
        assert command.target == "a"
        assert command.rot == "r"
        assert command.side is RotSide.LEFT

    def test_rotate_pair_holds_left_and_right_maps(self):
        pair = RotatePair(left_rot={"a": 1}, right_rot={"b": 2})
        assert pair.left_rot == {"a": 1}
        assert pair.right_rot == {"b": 2}


class TestGetRotateCommand:
    def test_single_pair_orders_left_before_right(self):
        pair = RotatePair(left_rot={"a": 1}, right_rot={"b": 2})

        commands = get_rotate_command(pair)

        assert commands == [
            RotateCommand(target="a", rot=1, side=RotSide.LEFT),
            RotateCommand(target="b", rot=2, side=RotSide.RIGHT),
        ]

    def test_list_of_pairs_is_flattened_in_order(self):
        pairs = [RotatePair(left_rot={"a": 1}, right_rot={}), RotatePair(left_rot={}, right_rot={"c": 3})]

        commands = get_rotate_command(pairs)

        assert commands == [
            RotateCommand(target="a", rot=1, side=RotSide.LEFT),
            RotateCommand(target="c", rot=3, side=RotSide.RIGHT),
        ]


class TestQuaRotInterface:
    def test_class_exposes_shared_aliases(self):
        assert QuaRotInterface.RotatePair is RotatePair
        assert QuaRotInterface.QuaRotMode is QuaRotMode

    @pytest.mark.parametrize("name", ["get_ln_fuse_map", "get_bake_names", "get_rotate_map"])
    def test_rotation_methods_are_marked_abstract(self, name):
        assert getattr(QuaRotInterface, name).__isabstractmethod__ is True

    def test_default_map_methods_return_empty_containers(self):
        interface = QuaRotInterface()
        assert interface.get_ln_fuse_map() == ({}, {})
        assert interface.get_bake_names() == ([], [])
        assert interface.get_rotate_map(block_size=64) == ([], [])

    def test_get_rotate_command_builds_matrix_with_requested_size(self):
        rot = QuaRotInterface.get_rotate_command(QuaRotMode.HADAMARD, size=4)

        assert tuple(rot.shape) == (4, 4)
        assert rot.dtype == GLOBAL_DTYPE

    def test_get_rotate_command_honours_block_size(self):
        rot = QuaRotInterface.get_rotate_command(QuaRotMode.HADAMARD, size=4, block_size=2)

        assert tuple(rot.shape) == (4, 4)


class TestLAOSOnlineRotationInterface:
    @pytest.mark.parametrize(
        "name",
        ["get_head_dim", "get_num_attention_heads", "get_layer_wise_ov_pair", "get_layer_wise_up_down_pair"],
    )
    def test_methods_are_marked_abstract(self, name):
        assert getattr(LAOSOnlineRotationInterface, name).__isabstractmethod__ is True

    def test_concrete_subclass_implements_contract(self):
        adapter = _LAOSAdapter()
        assert adapter.get_head_dim() == 8
        assert adapter.get_num_attention_heads() == 4
        assert not adapter.get_layer_wise_ov_pair(None)
        assert not adapter.get_layer_wise_up_down_pair(None)
