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

替换 pygtrie 后的依赖前缀冲突检测单元测试。
"""

from msmodelslim.utils.distributed.task_scheduler.backend.wave import _has_path_conflict


class TestHasPathConflict:
    """测试同路径 / 父子前缀冲突检测"""

    def test_has_path_conflict_returns_true_when_dep_exact_match(self):
        assert _has_path_conflict({"a.b.c"}, "a.b.c")

    def test_has_path_conflict_returns_true_when_dep_is_parent_of_registered(self):
        assert _has_path_conflict({"model.layers.0.self_attn.q_proj"}, "model.layers.0.self_attn")
        assert _has_path_conflict({"model.layers.0.self_attn.q_proj"}, "model.layers.0")

    def test_has_path_conflict_returns_true_when_registered_is_parent_of_dep(self):
        assert _has_path_conflict({"model.layers.0.self_attn"}, "model.layers.0.self_attn.q_proj")
        assert _has_path_conflict({"model.layers.0"}, "model.layers.0.self_attn.q_proj")

    def test_has_path_conflict_returns_false_when_paths_unrelated(self):
        assert not _has_path_conflict({"model.layers.0"}, "model.layers.1")
        assert not _has_path_conflict({"model.layers.0.self_attn"}, "model.layers.0.mlp")

    def test_has_path_conflict_returns_false_when_sibling_has_similar_prefix(self):
        assert not _has_path_conflict({"model.layers.1"}, "model.layers.10")
        assert not _has_path_conflict({"model.layers.1.mlp"}, "model.layers.10.mlp")

    def test_has_path_conflict_returns_false_when_dep_empty_or_dot_only(self):
        assert not _has_path_conflict({"model.layers.0"}, "")
        assert not _has_path_conflict({"model.layers.0"}, "..")

    def test_has_path_conflict_returns_true_when_dep_has_surrounding_dots(self):
        assert _has_path_conflict({"model.layers.0"}, ".model.layers.0.")

    def test_has_path_conflict_returns_true_when_any_registered_path_conflicts(self):
        assert _has_path_conflict({"m1", "m2.m3"}, "m2")
        assert not _has_path_conflict({"m1", "m2.m3"}, "m4")
