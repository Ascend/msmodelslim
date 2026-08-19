#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim.utils.config_map 的单元测试：覆盖 brace 展开 + 标准库 fnmatch
替代 wcmatch 后的匹配语义。
"""

from collections import OrderedDict

import pytest

from msmodelslim.utils.config_map import (
    ConfigMap,
    ConfigSet,
    _brace_fnmatch,
    _expand_braces,
)

Q_WEN3_BRACE = "model.layers.{1,2,3,4,5,6,7,8,30,31,32,43,44,45,46,52,60,61,62,63}.mlp.up_proj"


class TestExpandBraces:
    """测试 brace 展开实现"""

    def test_expand_braces_returns_alternatives_when_pattern_has_commas(self):
        assert _expand_braces("model.layers.{1,2,3}.mlp.up_proj") == (
            "model.layers.1.mlp.up_proj",
            "model.layers.2.mlp.up_proj",
            "model.layers.3.mlp.up_proj",
        )

    def test_expand_braces_returns_combinations_when_pattern_has_multiple_groups(self):
        assert _expand_braces("{a,b}.{1,2}") == ("a.1", "a.2", "b.1", "b.2")

    def test_expand_braces_returns_flattened_when_pattern_nested(self):
        assert _expand_braces("{a,{b,c}}") == ("a", "b", "c")

    def test_expand_braces_deduplicates_when_alternatives_repeat(self):
        assert _expand_braces("{a,a,b}") == ("a", "b")

    def test_expand_braces_returns_pattern_when_no_braces(self):
        assert _expand_braces("*.down_proj") == ("*.down_proj",)
        assert _expand_braces("") == ("",)

    def test_expand_braces_keeps_literal_when_single_item_brace(self):
        assert _expand_braces("{a}") == ("{a}",)

    def test_expand_braces_keeps_literal_when_brace_unclosed(self):
        assert _expand_braces("model.{1,2") == ("model.{1,2",)

    def test_expand_braces_keeps_literal_when_brace_escaped(self):
        assert _expand_braces(r"\{a,b}") == (r"\{a,b}",)


class TestBraceFnmatch:
    """测试 brace 展开 + fnmatch 的匹配语义"""

    def test_brace_fnmatch_matches_layer_when_name_in_qwen3_brace_pattern(self):
        assert _brace_fnmatch("model.layers.2.mlp.up_proj", Q_WEN3_BRACE)
        assert _brace_fnmatch("model.layers.61.mlp.up_proj", Q_WEN3_BRACE)
        assert not _brace_fnmatch("model.layers.9.mlp.up_proj", Q_WEN3_BRACE)
        assert not _brace_fnmatch("model.layers.2.mlp.down_proj", Q_WEN3_BRACE)

    def test_brace_fnmatch_matches_when_pattern_has_wildcard(self):
        assert _brace_fnmatch("model.layers.0.self_attn.q_proj", "*self_attn*")
        assert not _brace_fnmatch("model.layers.0.mlp", "*self_attn*")

    def test_brace_fnmatch_matches_when_pattern_has_question_mark_and_char_class(self):
        assert _brace_fnmatch("model.layers.0.self_attn.q_proj", "model.layers.?.self_attn.*")
        assert not _brace_fnmatch("model.layers.10.self_attn.q_proj", "model.layers.?.self_attn.*")
        assert _brace_fnmatch("model.layers.5.mlp.up_proj", "model.layers.[0-9].mlp.up_proj")
        assert not _brace_fnmatch("model.layers.15.mlp.up_proj", "model.layers.[0-9].mlp.up_proj")

    def test_brace_fnmatch_matches_when_name_empty(self):
        assert _brace_fnmatch("", "*")
        assert not _brace_fnmatch("", "a*")


class TestConfigSet:
    """测试 ConfigSet 的包含判断与 unmatched_keys"""

    def test_config_set_contains_returns_true_when_key_exact_match(self):
        config_set = ConfigSet(["model.layers.0.self_attn"])
        assert "model.layers.0.self_attn" in config_set
        assert "model.layers.1.self_attn" not in config_set

    def test_config_set_contains_returns_true_when_pattern_wildcard_match(self):
        config_set = ConfigSet(["*self_attn*", "*.down_proj"])
        assert "model.layers.0.self_attn.q_proj" in config_set
        assert "model.layers.0.mlp.down_proj" in config_set
        assert "model.layers.0.mlp.up_proj" not in config_set

    def test_config_set_contains_returns_true_when_pattern_brace_match(self):
        config_set = ConfigSet([Q_WEN3_BRACE])
        assert "model.layers.2.mlp.up_proj" in config_set
        assert "model.layers.9.mlp.up_proj" not in config_set

    def test_config_set_unmatched_keys_returns_unmatched_when_partial_patterns_matched(self):
        config_set = ConfigSet(["a.*", "b.*"])
        assert "a.x" in config_set
        assert config_set.unmatched_keys() == {"b.*"}

    def test_config_set_records_key_and_pattern_when_contains_matches(self):
        config_set = ConfigSet(["model.layers.0", "model.layers.{1,2,3}.mlp.up_proj", "b.*"])
        assert "model.layers.0" in config_set
        assert "model.layers.2.mlp.up_proj" in config_set
        # 直接命中记录 key 本身；模式命中记录模式原文（非展开结果）
        assert config_set.matched_patterns == {"model.layers.0", "model.layers.{1,2,3}.mlp.up_proj"}
        assert config_set.unmatched_keys() == {"b.*"}

    def test_config_set_unmatched_keys_empty_when_all_patterns_matched(self):
        config_set = ConfigSet(["a.*", "b.{1,2}"])
        assert "a.x" in config_set
        assert "b.2" in config_set
        assert config_set.unmatched_keys() == set()

    def test_config_set_deduplicates_patterns_when_duplicated_at_init(self):
        config_set = ConfigSet(["a", "a", "b"])
        assert len(config_set) == 2
        # 与 test_len_and_iter 相同的 C 层 set 继承怪癖，需用推导式
        assert {k for k in config_set} == {"a", "b"}  # pylint: disable=unnecessary-comprehension

    def test_config_set_contains_returns_false_when_case_differs(self):
        config_set = ConfigSet(["model.layers.0"])
        assert "model.layers.0" in config_set
        assert "Model.Layers.0" not in config_set

    def test_config_set_len_and_iter_work_when_initialized(self):
        config_set = ConfigSet(["a", "b", "a"])
        assert len(config_set) == 2
        # 注意：ConfigSet 继承 typing.Set（内置 set 别名），set(config_set) 走 C 层
        # 内部存储（为空），因此用推导式取迭代结果而非 set()
        assert {k for k in config_set} == {"a", "b"}  # pylint: disable=unnecessary-comprehension


class TestConfigMap:
    """测试 ConfigMap 的取值与包含判断"""

    def test_config_map_getitem_returns_exact_value_when_key_present(self):
        cfg = ConfigMap(OrderedDict([("model.layers.0", "exact"), ("model.layers.*", "wild")]))
        assert cfg["model.layers.0"] == "exact"
        assert cfg["model.layers.1"] == "wild"

    def test_config_map_getitem_matches_when_pattern_has_braces(self):
        cfg = ConfigMap(OrderedDict([("model.layers.{1,2,3}.mlp.up_proj", "target")]))
        assert cfg["model.layers.2.mlp.up_proj"] == "target"
        assert "model.layers.2.mlp.up_proj" in cfg
        assert "model.layers.4.mlp.up_proj" not in cfg

    def test_config_map_getitem_raises_key_error_when_no_match(self):
        cfg = ConfigMap(OrderedDict([("a.*", 1)]))
        with pytest.raises(KeyError):
            _ = cfg["b"]

    def test_config_map_getitem_returns_first_pattern_when_multiple_match(self):
        cfg = ConfigMap(
            OrderedDict(
                [
                    ("model.layers.{1,2}.mlp.up_proj", "brace"),
                    ("model.layers.*.mlp.up_proj", "wild"),
                ]
            )
        )
        assert cfg["model.layers.1.mlp.up_proj"] == "brace"  # 先命中的模式优先

    def test_config_map_contains_does_not_mutate_when_queried(self):
        cfg = ConfigMap(OrderedDict([("model.layers.{1,2}.mlp.up_proj", "v")]))
        assert len(cfg) == 1
        assert "model.layers.1.mlp.up_proj" in cfg
        assert len(cfg) == 1
        assert "model.layers.9.mlp.up_proj" not in cfg
        assert len(cfg) == 1

    def test_config_map_getitem_raises_and_contains_false_when_empty(self):
        cfg = ConfigMap(OrderedDict())
        assert len(cfg) == 0
        assert "anything" not in cfg
        with pytest.raises(KeyError):
            _ = cfg["anything"]

    def test_config_map_len_and_iter_work_when_initialized(self):
        cfg = ConfigMap(OrderedDict([("a", 1), ("b", 2)]))
        assert len(cfg) == 2
        assert set(cfg) == {"a", "b"}
        assert dict(cfg.items()) == {"a": 1, "b": 2}
