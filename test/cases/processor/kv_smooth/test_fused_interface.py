#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.processor.kv_smooth.fused_interface 模块的单元测试
"""

# pylint: disable=abstract-class-instantiated

import pytest

from msmodelslim.processor.kv_smooth.fused_interface import (
    KVSmoothFusedInterface,
    KVSmoothFusedType,
    KVSmoothFusedUnit,
)
from msmodelslim.utils.exception import SchemaValidateError


class _CompleteAdapter(KVSmoothFusedInterface):
    def get_kvcache_smooth_fused_subgraph(self):
        return []

    def get_head_dim(self):
        return 8

    def get_num_key_value_groups(self):
        return 4

    def get_num_key_value_heads(self):
        return 4


class TestKVSmoothFusedType:
    def test_enum_members_have_expected_wire_values(self):
        assert KVSmoothFusedType.StateViaRopeToNorm.value == "state-rope-norm"
        assert KVSmoothFusedType.StateViaRopeToLinear.value == "state-rope-linear"

    def test_enum_lookup_by_value(self):
        assert KVSmoothFusedType("state-rope-norm") is KVSmoothFusedType.StateViaRopeToNorm
        assert KVSmoothFusedType("state-rope-linear") is KVSmoothFusedType.StateViaRopeToLinear


class TestKVSmoothFusedUnit:
    def _kwargs(self, **overrides):
        data = {
            "attention_name": "model.layers.0.self_attn",
            "layer_idx": 0,
            "fused_from_query_states_name": "q_proj",
            "fused_from_key_states_name": "k_proj",
            "fused_type": "state-rope-norm",
        }
        data.update(overrides)
        return data

    def test_valid_unit_coerces_string_fused_type_to_enum(self):
        unit = KVSmoothFusedUnit(**self._kwargs())
        assert unit.attention_name == "model.layers.0.self_attn"
        assert unit.layer_idx == 0
        assert unit.fused_type is KVSmoothFusedType.StateViaRopeToNorm

    def test_missing_required_field_is_rejected(self):
        data = self._kwargs()
        data.pop("fused_from_key_states_name")
        with pytest.raises(SchemaValidateError):
            KVSmoothFusedUnit(**data)

    def test_unknown_fused_type_is_rejected(self):
        with pytest.raises(SchemaValidateError):
            KVSmoothFusedUnit(**self._kwargs(fused_type="not-a-type"))


class TestKVSmoothFusedInterface:
    def test_base_interface_declares_four_abstract_methods(self):
        assert KVSmoothFusedInterface.__abstractmethods__ == {
            "get_kvcache_smooth_fused_subgraph",
            "get_head_dim",
            "get_num_key_value_groups",
            "get_num_key_value_heads",
        }

    def test_base_interface_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            KVSmoothFusedInterface()

    def test_incomplete_subclass_stays_abstract(self):
        class _Partial(KVSmoothFusedInterface):
            def get_head_dim(self):
                return 8

        with pytest.raises(TypeError):
            _Partial()

    def test_complete_subclass_is_instantiable(self):
        adapter = _CompleteAdapter()
        assert adapter.get_head_dim() == 8
        assert not adapter.get_kvcache_smooth_fused_subgraph()
