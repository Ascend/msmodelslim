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

msmodelslim.core.quant_service.modelslim_convert.impl.virtual_tree 模块的单元测试
"""

from unittest.mock import MagicMock

from torch import nn

from msmodelslim.core.convert.catalog import TensorCatalog, TensorEntry
from msmodelslim.core.convert.config import ConvertConfig, ModuleRule
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.impl.virtual_tree import (
    VirtualModelTreeBuilder,
    collect_bound_catalog_keys,
)
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear, PassthroughModule


class TestCollectBoundCatalogKeys:
    """测试 collect_bound_catalog_keys 函数"""

    def test_collect_bound_catalog_keys_return_bound_keys_when_tree_has_modules(self):
        catalog = TensorCatalog()
        for key, shape in [
            ("lm_head.weight", (1, 2)),
            ("mtp.fc.weight", (2, 2)),
            ("model.visual.merger.linear_fc1.weight", (3, 4)),
        ]:
            catalog.add(TensorEntry(key=key, shard="s0", dtype="bf16", shape=shape))
        root = nn.Module()
        merger = PassthroughModule(
            full_name="model.visual.merger.linear_fc1",
            tensor_bindings={
                "weight": TensorRef("weight", "model.visual.merger.linear_fc1.weight", "s0", "bf16", (3, 4)),
            },
            source_ir=SourceIR(kind=IRKind.FLOAT),
        )
        root.add_module("merger", merger)
        handled = collect_bound_catalog_keys(root)
        assert "model.visual.merger.linear_fc1.weight" in handled  # 校验已绑定 key
        remaining = [k for k in catalog.keys() if k not in handled]
        assert set(remaining) == {"lm_head.weight", "mtp.fc.weight"}  # 校验未绑定 key 保留


class TestVirtualModelTreeBuilder:
    """测试 VirtualModelTreeBuilder 类"""

    def test_build_create_linear_module_when_module_rule_matches(self):
        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="layers.0.q_proj.weight", shard="s0", dtype="bf16", shape=(4, 8)))
        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            module_rules=[
                ModuleRule(
                    match="layers.*.q_proj",
                    module_kind="linear",
                    source_format="bf16",
                    source_ir=IRKind.FLOAT,
                    tensor_map={"weight": "{module}.weight"},
                ),
            ],
        )
        context = ConvertContext(config=config)
        context.reader = MagicMock()

        tree = VirtualModelTreeBuilder().build(context, catalog)
        mod = tree.get_submodule("layers.0.q_proj")
        assert isinstance(mod, ModelFreeLinear)  # 校验 linear 规则创建 ModelFreeLinear
        assert "weight" in mod.tensor_bindings

    def test_build_infer_fp8_source_ir_when_auto_route_and_scale_inv_in_catalog(self):
        catalog = TensorCatalog()
        catalog.add(
            TensorEntry(key="layers.0.q_proj.weight", shard="s0", dtype="float8_e4m3fn", shape=(4, 8)),
        )
        catalog.add(
            TensorEntry(key="layers.0.q_proj.weight_scale_inv", shard="s0", dtype="bf16", shape=(4, 1)),
        )
        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            module_rules=[
                ModuleRule(
                    match="layers.*.q_proj",
                    module_kind="linear",
                    tensor_map={
                        "weight": "{module}.weight",
                        "weight_scale_inv": "{module}.weight_scale_inv",
                    },
                ),
            ],
            convert_rules=[],
        )
        context = ConvertContext(config=config)
        context.reader = MagicMock()

        tree = VirtualModelTreeBuilder().build(context, catalog)
        mod = tree.get_submodule("layers.0.q_proj")
        assert isinstance(mod, ModelFreeLinear)
        assert mod.source_ir.kind == IRKind.FP8_BLOCK  # 校验 FP8 推断
        assert "weight_scale_inv" in mod.tensor_bindings

    def test_build_attach_passthrough_for_unmatched_weights(self):
        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="lm_head.weight", shard="s0", dtype="bf16", shape=(2, 2)))
        config = ConvertConfig(model_path="/m", save_path="/o", module_rules=[])
        context = ConvertContext(config=config)
        context.reader = MagicMock()

        tree = VirtualModelTreeBuilder().build(context, catalog)
        mod = tree.get_submodule("lm_head")
        assert isinstance(mod, PassthroughModule)  # 校验未匹配权重走 passthrough

    def test_build_passthrough_keep_scale_inv_when_unmatched_fp8(self):
        """未匹配 linears 的 FP8 权重应与 weight_scale_inv 同组挂到 PassthroughModule。"""
        catalog = TensorCatalog()
        catalog.add(
            TensorEntry(key="mtp.embed_tokens.weight", shard="s0", dtype="float8_e4m3fn", shape=(4, 8)),
        )
        catalog.add(
            TensorEntry(key="mtp.embed_tokens.weight_scale_inv", shard="s0", dtype="bf16", shape=(1, 1)),
        )
        config = ConvertConfig(model_path="/m", save_path="/o", module_rules=[])
        context = ConvertContext(config=config)
        context.reader = MagicMock()

        tree = VirtualModelTreeBuilder().build(context, catalog)
        mod = tree.get_submodule("mtp.embed_tokens")
        assert isinstance(mod, PassthroughModule)
        assert "weight" in mod.tensor_bindings
        assert "weight_scale_inv" in mod.tensor_bindings
        assert mod.tensor_bindings["weight_scale_inv"].key == "mtp.embed_tokens.weight_scale_inv"

    def test_build_skip_norm_modules_when_convert_false(self):
        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="model.norm.weight", shard="s0", dtype="bf16", shape=(2,)))
        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            module_rules=[
                ModuleRule(
                    match="model.norm",
                    module_kind="norm",
                    convert=False,
                    tensor_map={"weight": "{module}.weight"},
                ),
            ],
        )
        context = ConvertContext(config=config)
        context.reader = MagicMock()
        tree = VirtualModelTreeBuilder().build(context, catalog)
        mod = tree.get_submodule("model.norm")
        assert isinstance(mod, PassthroughModule)  # 校验 convert=False 的 norm

    def test_build_preserve_all_merge_bindings_when_parent_has_children(self):
        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="model.layers.0.q_proj.weight", shard="s0", dtype="bf16", shape=(2, 2)))
        catalog.add(TensorEntry(key="lm_head.weight", shard="s0", dtype="bf16", shape=(2, 2)))
        config = ConvertConfig(
            model_path="/m",
            save_path="/o",
            module_rules=[
                ModuleRule(
                    match="model.layers.*.q_proj",
                    module_kind="linear",
                    source_format="bf16",
                    source_ir=IRKind.FLOAT,
                    tensor_map={"weight": "{module}.weight"},
                ),
            ],
        )
        context = ConvertContext(config=config)
        context.reader = MagicMock()
        tree = VirtualModelTreeBuilder().build(context, catalog)
        assert tree.get_submodule("model.layers.0.q_proj") is not None  # 校验子模块 linear
        assert tree.get_submodule("lm_head") is not None  # 校验余量 passthrough

    def test_build_attach_passthrough_for_unmatched_catalog_keys(self):
        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="model.embed_tokens.weight", shard="s0", dtype="bf16", shape=(2, 2)))
        config = ConvertConfig(model_path="/m", save_path="/o", module_rules=[])
        context = ConvertContext(config=config)
        context.reader = MagicMock()
        tree = VirtualModelTreeBuilder().build(context, catalog)
        assert isinstance(tree.get_submodule("model.embed_tokens"), PassthroughModule)
