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

import sys
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.core.context.shared_dict_context.context import SharedDictContext, SharedNamespace

pytestmark_linux = pytest.mark.skipif(
    sys.platform != "linux",
    reason="SharedDictContext multi-process tests require Linux (AF_UNIX, SO_PEERCRED)",
)


class CustomInvalidType:
    def __init__(self, name: str = "invalid") -> None:
        self.name = name


def _worker_read_ns(shared_ctx, ev, ns_key, expected, tensor_keys):
    ev.wait(timeout=10)
    ns = shared_ctx[ns_key]
    for k, exp in expected.items():
        assert __import__("torch").allclose(ns.debug.get(k), exp) if k in tensor_keys else ns.debug.get(k) == exp
        assert (
            __import__("torch").allclose(ns.state.get(f"state_{k}"), exp)
            if k in tensor_keys
            else ns.state.get(f"state_{k}") == exp
        )


def _worker_invalid_type(shared_ctx, ev, ns_key, attr, queue):
    ev.wait(timeout=10)
    ns = shared_ctx[ns_key]
    try:
        getattr(ns, attr)["x"] = CustomInvalidType(attr)
        queue.put((attr, False, ""))
    except Exception as e:
        queue.put((attr, "Unsupported value type" in str(e), str(e)))


class TestSharedNamespace:
    """Tests for SharedNamespace."""

    @pytest.mark.skipif(sys.platform != "linux", reason="SharedDictContext requires Linux (PeercredManager)")
    def test_state_accepts_whitelist_types_when_written(self):
        """场景：写入白名单类型到 SharedNamespace.state。预期：读写一致。"""
        torch = pytest.importorskip("torch")
        ctx = SharedDictContext(enable_debug=True)
        try:
            ns = ctx["types"]
            assert isinstance(ns, SharedNamespace)
            t = torch.tensor([1.0, 2.0], device="cpu")
            ns.state["tensor"] = t
            ns.state["list"] = [1, "two"]
            assert torch.allclose(ns.state["tensor"], t)
            assert ns.state["list"] == [1, "two"]
        finally:
            ctx._manager.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="SharedDictContext requires Linux (PeercredManager)")
class TestSharedDictContext:
    """Tests for SharedDictContext namespace (single-process and multi-process)."""

    @pytest.fixture
    def shared_ctx(self):
        ctx = SharedDictContext(enable_debug=True)
        yield ctx
        ctx._manager.shutdown()

    def test_getitem_creates_namespace_when_key_missing(self, shared_ctx):
        """场景：首次访问 namespace key。预期：创建并可写入 state。"""
        ns = shared_ctx["test"]
        ns.state["string"] = "hello"
        ns.state["int"] = 42
        assert shared_ctx["test"].state["string"] == "hello"
        assert shared_ctx["test"].state["int"] == 42

    def test_namespace_accepts_whitelist_types_when_written(self, shared_ctx):
        """场景：写入白名单类型到 state。预期：读写一致。"""
        torch = pytest.importorskip("torch")
        ns = shared_ctx["types"]
        t = torch.tensor([1.0, 2.0], device="cpu")
        ns.state["tensor"] = t
        ns.state["list"] = [1, "two"]
        assert torch.allclose(ns.state["tensor"], t)
        assert ns.state["list"] == [1, "two"]

    @pytestmark_linux
    def test_setitem_raises_error_when_non_whitelist_type_in_child_process(self):
        """场景：子进程写入非白名单类型。预期：state/debug 均拒绝。"""
        mp = pytest.importorskip("torch.multiprocessing")
        ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
        parent_ctx = SharedDictContext(enable_debug=True)
        _ = parent_ctx["invalid_ns"]
        ev, q = ctx.Event(), ctx.Queue()
        p1 = ctx.Process(target=_worker_invalid_type, args=(parent_ctx, ev, "invalid_ns", "state", q))
        p2 = ctx.Process(target=_worker_invalid_type, args=(parent_ctx, ev, "invalid_ns", "debug", q))
        p1.start()
        p2.start()
        ev.set()
        p1.join(timeout=30)
        p2.join(timeout=30)
        assert p1.exitcode == 0 and p2.exitcode == 0
        res = {}
        for _ in range(2):
            attr, rejected, msg = q.get(timeout=5)
            res[attr] = (attr, rejected, msg)
        for attr in ("state", "debug"):
            assert res[attr][1] is True, res[attr][2]
        parent_ctx._manager.shutdown()

    @pytestmark_linux
    def test_namespace_stores_all_supported_types_when_written(self):
        """场景：写入基础类型、集合与张量。预期：读取值一致。"""
        torch = pytest.importorskip("torch")
        ctx = SharedDictContext(enable_debug=True)
        try:
            ns = ctx["test"]
            ns.state["string"] = "hello"
            ns.state["int"] = 42
            ns.state["float"] = 3.14
            ns.state["bool"] = True
            ns.state["bytes"] = b"data"
            ns.state["none"] = None
            ns.state["dict"] = {"a": 1, "c": [1, 2, 3]}
            ns.state["list"] = [1, "two", 3.0, True]
            ns.state["tuple"] = (1, 2, 3)
            t = torch.tensor([1, 2, 3], device="cpu")
            ns.state["tensor"] = t
            nested = {"config": {"lr": 0.001, "layers": [64, 128]}, "weights": torch.randn(10, 10, device="cpu")}
            ns.state["nested"] = nested
            assert ns.state["string"] == "hello" and ns.state["int"] == 42 and ns.state["float"] == 3.14
            assert ns.state["bool"] is True and ns.state["bytes"] == b"data" and ns.state["none"] is None
            assert ns.state["dict"] == {"a": 1, "c": [1, 2, 3]} and ns.state["list"] == [1, "two", 3.0, True]
            assert ns.state["tuple"] == (1, 2, 3) and torch.allclose(ns.state["tensor"], t)
            r = ns.state["nested"]
            assert r["config"]["lr"] == 0.001 and torch.allclose(r["weights"], nested["weights"])
        finally:
            ctx._manager.shutdown()

    def test_getitem_creates_shared_namespace_when_manager_mocked(self):
        """场景：mock manager 下首次访问 key。预期：创建 SharedNamespace。"""
        mock_mgr = MagicMock()
        mock_mgr.dict.return_value = {}
        mock_mgr.validated_dict = MagicMock(return_value=MagicMock())

        ctx = SharedDictContext.__new__(SharedDictContext)
        ctx._enable_debug = False
        ctx._manager = mock_mgr
        ctx._address = "/tmp/fake.sock"
        ctx._namespaces = {}

        ns = ctx["ns1"]
        assert isinstance(ns, SharedNamespace)
        assert "ns1" in ctx._namespaces

    def test_getstate_clears_manager_when_pickled(self):
        """场景：__getstate__。预期：_manager 置 None。"""
        ctx = SharedDictContext.__new__(SharedDictContext)
        ctx._manager = MagicMock()
        ctx._address = "/tmp/x.sock"
        ctx._namespaces = {}
        ctx._enable_debug = True
        state = ctx.__getstate__()
        assert state["_manager"] is None

    def test_ensure_manager_reconnects_when_manager_none(self):
        """场景：_manager 为 None 时 _ensure_manager。预期：重建 PeercredManager。"""
        ctx = SharedDictContext.__new__(SharedDictContext)
        ctx._manager = None
        ctx._address = "/tmp/recon.sock"
        with patch("msmodelslim.core.context.shared_dict_context.context.PeercredManager") as mock_cls:
            mock_cls.return_value = MagicMock()
            mgr = ctx._ensure_manager()
            mock_cls.assert_called_once_with(address="/tmp/recon.sock")
            assert mgr is mock_cls.return_value

    def test_repr_shows_context_type_when_called(self):
        """场景：SharedDictContext.__repr__。预期：含 SharedContext。"""
        ctx = SharedDictContext.__new__(SharedDictContext)
        ctx._namespaces = MagicMock()
        ctx._namespaces.keys.return_value = ["a"]
        assert "SharedContext" in repr(ctx)

    def test_setstate_restores_fields_when_unpickled(self):
        """场景：__setstate__。预期：_manager 为 None 且保留 enable_debug。"""
        ctx = SharedDictContext.__new__(SharedDictContext)
        ctx.__setstate__({"_enable_debug": True, "_address": "/tmp/x", "_namespaces": {}})
        assert ctx._manager is None
        assert ctx._enable_debug is True
