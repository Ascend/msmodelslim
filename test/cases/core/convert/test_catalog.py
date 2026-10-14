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

msmodelslim.core.convert.catalog 模块的单元测试
"""

from msmodelslim.core.convert.catalog import (
    DependencyMap,
    TensorCatalog,
    TensorEntry,
    build_dependency_map,
)


class TestTensorCatalog:
    """测试 TensorCatalog 类"""

    def test_from_raw_weight_map_create_entries_when_header_given(self):
        weight_map = {"a.weight": "s0", "b.weight": "s1"}
        headers = {"a.weight": ("bf16", (2, 3))}
        catalog = TensorCatalog.from_raw_weight_map(weight_map, headers)
        assert len(catalog) == 2
        assert catalog.get("a.weight").dtype == "bf16"
        assert catalog.get("a.weight").shape == (2, 3)
        assert catalog.get("b.weight").dtype == "UNKNOWN"

    def test_to_weight_map_return_shard_mapping_when_entries_exist(self):
        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="x", shard="shard0", dtype="bf16", shape=(1,)))
        assert catalog.to_weight_map() == {"x": "shard0"}


class TestDependencyMap:
    """测试 DependencyMap 类"""

    def test_inverse_load_map_group_keys_by_shard_when_owners_registered(self):
        dep = DependencyMap()
        dep.add_owner("a", "s0")
        dep.add_owner("b", "s0")
        dep.add_owner("c", "s1")
        result = dep.inverse_load_map(["a", "b", "c"])
        assert result == {"s0": ["a", "b"], "s1": ["c"]}

    def test_dependencies_of_return_fused_deps_when_added(self):
        dep = DependencyMap()
        dep.add_dependency("logical", "fused")
        assert dep.dependencies_of("logical") == {"fused"}


class TestBuildDependencyMap:
    """测试 build_dependency_map 函数"""

    def test_build_dependency_map_link_fused_from_when_catalog_has_meta(self):
        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="fused", shard="s0", dtype="bf16", shape=(2, 2)))
        catalog.add(
            TensorEntry(
                key="expert.0.gate",
                shard="s0",
                dtype="bf16",
                shape=(),
                meta={"fused_from": "fused"},
            ),
        )
        dep = build_dependency_map(catalog.to_weight_map(), catalog=catalog)
        assert "fused" in dep.dependencies_of("expert.0.gate")
