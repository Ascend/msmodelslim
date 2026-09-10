# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim/utils/buffer/runtime.py 的单元测试。
"""

import pytest
import torch
from torch import nn

from msmodelslim.utils.buffer.runtime import (
    RuntimeBufferStore,
    _is_under_prefix,
    _name_in_scope,
    collect_nonpersistent_buffers,
)


class _ShellWithBuffers(nn.Module):
    """仿真推理 shell: 一个持久 buffer、若干非持久 buffer 与嵌套子模块。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("persistent", torch.ones(2))
        self.register_buffer("rope_freq", torch.arange(4, dtype=torch.float32), persistent=False)
        self.shared = nn.Module()
        self.shared.register_buffer("bias", torch.zeros(3), persistent=False)


class _MetaChild(nn.Module):
    """非持久 buffer 落在 meta 设备上的子模块。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.randn(4), persistent=False)
        self.to_empty(device="meta")


class _BufferModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pos", torch.zeros(1), persistent=False)


# ---------------------------------------------------------------------------
# collect_nonpersistent_buffers
# ---------------------------------------------------------------------------


class TestCollectNonpersistentBuffers:
    """对应 collect_nonpersistent_buffers 函数。"""

    def test_collect_returns_cpu_clone_when_buffer_holds_real_data(self):
        shell = _ShellWithBuffers()

        collected = collect_nonpersistent_buffers(shell)

        assert set(collected.keys()) == {"rope_freq", "shared.bias"}
        assert collected["rope_freq"].device.type == "cpu"
        assert collected["shared.bias"].device.type == "cpu"
        # 持久 buffer 不在收集范围
        assert "persistent" not in collected

    def test_collect_returns_empty_when_all_buffers_on_meta(self):
        child = _MetaChild()

        collected = collect_nonpersistent_buffers(child)

        assert not collected

    def test_collect_returns_detached_clone_when_mutated_afterwards(self):
        shell = _ShellWithBuffers()
        original = shell.rope_freq.clone()

        collected = collect_nonpersistent_buffers(shell)
        collected["rope_freq"].add_(1)

        assert torch.equal(shell.rope_freq, original)


# ---------------------------------------------------------------------------
# _is_under_prefix / _name_in_scope（内部工具，restore 依赖）
# ---------------------------------------------------------------------------


class TestNameScopeHelpers:
    """对应 _is_under_prefix 与 _name_in_scope。"""

    def test_is_under_prefix_returns_true_when_exact_or_descendant(self):
        prefixes = {"model.layers.0", "shared"}

        assert _is_under_prefix("model.layers.0", prefixes) is True
        assert _is_under_prefix("model.layers.0.attn", prefixes) is True
        assert _is_under_prefix("shared.emb", prefixes) is True

    def test_is_under_prefix_returns_false_when_sibling_or_unrelated(self):
        prefixes = {"model.layers.0"}

        assert _is_under_prefix("model.layers.1", prefixes) is False
        assert _is_under_prefix("model.layers00.attn", prefixes) is False

    def test_name_in_scope_filters_by_prefix_when_prefix_given(self):
        assert _name_in_scope("model.layers.0.weight", prefix="model.layers.0") is True
        assert _name_in_scope("model.layers.0", prefix="model.layers.0") is True
        assert _name_in_scope("model.layers.1.weight", prefix="model.layers.0") is False

    def test_name_in_scope_filters_by_skip_when_no_prefix(self):
        skip = {"model.layers.0"}

        assert _name_in_scope("model.layers.0.weight", skip_prefixes=skip) is False
        assert _name_in_scope("model.embed", skip_prefixes=skip) is True

    def test_name_in_scope_returns_true_when_no_scope_given(self):
        assert _name_in_scope("anything") is True


# ---------------------------------------------------------------------------
# RuntimeBufferStore
# ---------------------------------------------------------------------------


class TestRuntimeBufferStore:  # pylint: disable=attribute-defined-outside-init
    """对应 RuntimeBufferStore。"""

    def test_store_load_returns_snapshot_when_replaced(self):
        store = RuntimeBufferStore()
        tensors = {"rope": torch.ones(2)}

        store.store(tensors)

        assert store.load() is tensors

    def test_load_returns_empty_when_nothing_stored(self):
        store = RuntimeBufferStore()

        assert store.load() == {}

    def test_clear_drops_tensors_when_called(self):
        store = RuntimeBufferStore()
        store.store({"rope": torch.ones(2)})

        store.clear()

        assert store.load() == {}

    def test_store_overwrites_previous_when_called_twice(self):
        store = RuntimeBufferStore()
        store.store({"a": torch.ones(1)})

        store.store({"b": torch.ones(1)})

        assert set(store.load().keys()) == {"b"}

    def test_restore_registers_nonpersistent_when_target_is_meta(self):
        store = RuntimeBufferStore()
        model = nn.Module()
        model.register_buffer("inv_freq", torch.empty(4), persistent=False)
        model.to_empty(device="meta")
        assert model.inv_freq.device.type == "meta"
        store.store({"inv_freq": torch.arange(4, dtype=torch.float32)})

        restored = store.restore(model)

        assert restored == 1
        assert model.inv_freq.device.type == "cpu"
        assert "inv_freq" in model._non_persistent_buffers_set
        assert torch.equal(model.inv_freq, torch.arange(4, dtype=torch.float32))

    def test_restore_registers_when_target_is_real_cpu(self):
        store = RuntimeBufferStore()
        model = nn.Module()
        model.register_buffer("state", torch.zeros(3), persistent=False)
        store.store({"state": torch.tensor([1.0, 2.0, 3.0])})

        restored = store.restore(model)

        assert restored == 1
        assert torch.equal(model.state, torch.tensor([1.0, 2.0, 3.0]))
        assert "state" in model._non_persistent_buffers_set

    def test_restore_replaces_when_target_shape_drifted(self):
        """模拟长上下文缓存被前向按当前 seq_len 改写后再次 restore 的场景。"""
        store = RuntimeBufferStore()
        model = nn.Module()
        model.register_buffer("rope_table", torch.zeros(3), persistent=False)
        model.rope_table[:] = torch.tensor([9.0, 9.0, 9.0])  # 逃逸 .to(meta) 的真实小形状张量
        store.store({"rope_table": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])})  # 快照全量 5

        restored = store.restore(model)

        assert restored == 1
        assert torch.equal(model.rope_table, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert "rope_table" in model._non_persistent_buffers_set

    def test_restore_raises_key_error_when_target_is_unregistered_plain_tensor(self):
        """目标被模型前向改成普通张量属性(绕过 register_buffer)时,restore 不再接管,
        而是让 register_buffer 抛 KeyError,把建模侧的问题暴露出来。
        """
        store = RuntimeBufferStore()
        model = nn.Module()
        model.rope_cache = torch.zeros(3)  # 普通属性赋值,不在 _buffers 中
        store.store({"rope_cache": torch.tensor([1.0, 2.0, 3.0])})

        with pytest.raises(KeyError):
            store.restore(model)

    def test_restore_skips_submodule_when_same_name_is_module(self):
        """同名目标是子模块(非张量)时不应被接管。"""
        store = RuntimeBufferStore()
        model = nn.Module()
        model.rope = nn.Linear(2, 2)  # 与快照同名的子模块
        store.store({"rope": torch.tensor([1.0, 2.0])})

        assert store.restore(model) == 0
        assert isinstance(model.rope, nn.Linear)

    def test_restore_scopes_to_prefix_when_given(self):
        store = RuntimeBufferStore()
        model = _BufferModule()
        model.head = _BufferModule()
        store.store(
            {
                "pos": torch.tensor(1.0),
                "head.pos": torch.tensor(2.0),
            }
        )

        restored = store.restore(model, prefix="head")

        assert restored == 1
        assert model.head.pos.item() == 2.0
        assert model.pos.item() == 0.0  # 未被本次 restore 覆盖

    def test_restore_skips_prefixes_when_given(self):
        store = RuntimeBufferStore()
        model = _BufferModule()
        model.head = _BufferModule()
        store.store(
            {
                "pos": torch.tensor(1.0),
                "head.pos": torch.tensor(2.0),
            }
        )

        restored = store.restore(model, skip_prefixes={"head"})

        assert restored == 1
        assert model.pos.item() == 1.0
        assert model.head.pos.item() == 0.0  # 原值未被覆盖

    def test_restore_returns_zero_when_snapshot_empty(self):
        store = RuntimeBufferStore()
        model = _BufferModule()

        assert store.restore(model) == 0

    def test_restore_returns_zero_when_parent_submodule_missing(self):
        store = RuntimeBufferStore()
        model = _BufferModule()
        store.store({"ghost.pos": torch.tensor(1.0)})

        assert store.restore(model) == 0

    def test_restore_returns_zero_when_existing_attr_is_not_tensor(self):
        store = RuntimeBufferStore()
        model = _BufferModule()
        model.note = "i-am-not-a-tensor"
        store.store({"note": torch.tensor(1.0)})

        assert store.restore(model) == 0

    def test_restore_updates_root_level_key_when_on_root_module(self):
        store = RuntimeBufferStore()
        model = nn.Module()
        model.register_buffer("root_buf", torch.empty(2), persistent=False)
        store.store({"root_buf": torch.tensor([9.0, 8.0])})

        assert store.restore(model) == 1
        assert torch.equal(model.root_buf, torch.tensor([9.0, 8.0]))
