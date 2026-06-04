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

import socket
import struct
import sys
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.core.context.shared_dict_context.peercred_manager import (
    PeercredConnection,
    PeercredManager,
    PeercredProxy,
    PeercredServer,
    PeercredSharedDict,
    PeercredValidatedDict,
)
from msmodelslim.utils.exception import (
    MisbehaviorError,
    SchemaValidateError,
    SecurityError,
    SpecError,
    UnsupportedError,
)


class TestPeercredConnection:
    """Tests for PeercredConnection framing."""

    def test_send_recv_roundtrip_object_when_local_socket_pair(self):
        """场景：本地 socket 对发送 pickle 对象。预期：recv 还原对象。"""
        server_sock, client_sock = socket.socketpair()
        try:
            server = PeercredConnection(server_sock)
            client = PeercredConnection(client_sock)
            client.send({"key": "value"})
            assert server.recv() == {"key": "value"}
        finally:
            server_sock.close()
            client_sock.close()

    def test_send_raises_security_error_when_message_too_large(self):
        """场景：发送超大消息。预期：SecurityError。"""
        from msmodelslim.core.context.shared_dict_context.peercred_manager import MAX_MESSAGE_SIZE

        server_sock, client_sock = socket.socketpair()
        try:
            client = PeercredConnection(client_sock)
            with pytest.raises(SecurityError, match="exceeds maximum"):
                client.send(b"x" * (MAX_MESSAGE_SIZE + 1))
        finally:
            server_sock.close()
            client_sock.close()

    def test_recv_raises_security_error_when_size_header_too_large(self):
        """场景：对端发送超大长度头。预期：SecurityError。"""
        server_sock, client_sock = socket.socketpair()
        try:
            server = PeercredConnection(server_sock)
            with patch(
                "msmodelslim.core.context.shared_dict_context.peercred_manager.MAX_MESSAGE_SIZE",
                64,
            ):
                client_sock.send(struct.pack("!I", 128))
                with pytest.raises(SecurityError, match="exceeds maximum"):
                    server.recv()
        finally:
            server_sock.close()
            client_sock.close()


@pytest.mark.skipif(sys.platform != "linux", reason="SO_PEERCRED requires Linux")
class TestPeercredListener:
    """Tests for PeercredListener on Linux."""

    def test_accept_returns_peer_credentials_when_local_socket(self):
        """场景：本地 Unix socket 连接。预期：accept 返回 pid/uid/gid。"""
        import os
        import tempfile

        from msmodelslim.core.context.shared_dict_context.peercred_manager import PeercredListener

        with tempfile.TemporaryDirectory() as tmpdir:
            addr = os.path.join(tmpdir, "test.sock")
            listener = PeercredListener(addr)
            listener.start()
            try:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # pylint: disable=no-member
                client.connect(addr)
                server_conn, (pid, uid, gid) = listener.accept()
                assert pid > 0
                assert uid == os.getuid()  # pylint: disable=no-member
                assert gid == os.getgid()  # pylint: disable=no-member
                server_conn.close()
                client.close()
            finally:
                listener.close()


class TestPeercredSharedDict:
    """Tests for PeercredSharedDict mapping API."""

    def test_setitem_getitem_roundtrip_when_key_set(self):
        """场景：写入并读取 key。预期：值一致。"""
        d = PeercredSharedDict()
        d["a"] = 1
        assert d["a"] == 1

    def test_delitem_removes_key_when_deleted(self):
        """场景：删除 key。预期：KeyError。"""
        d = PeercredSharedDict({"x": 1})
        del d["x"]
        with pytest.raises(KeyError):
            _ = d["x"]

    def test_update_and_items_when_bulk_update(self):
        """场景：update 批量写入。预期：items 包含新键值。"""
        d = PeercredSharedDict()
        d.update({"k": "v"})
        assert ("k", "v") in d.items()

    def test_pop_returns_value_when_key_exists(self):
        """场景：pop 已有 key。预期：返回值并删除。"""
        d = PeercredSharedDict({"n": 42})
        assert d.pop("n") == 42
        assert "n" not in d

    def test_clear_empties_dict_when_called(self):
        """场景：clear。预期：长度为 0。"""
        d = PeercredSharedDict({"a": 1})
        d.clear()
        assert len(d) == 0

    def test_get_set_del_and_iter_when_used(self):
        """场景：完整 dict 操作。预期：keys/len/contains 正确。"""
        d = PeercredSharedDict({"a": 1})
        d["b"] = 2
        assert d["a"] == 1
        assert "b" in d
        assert len(d) == 2
        assert list(d.keys()) == ["a", "b"]
        del d["a"]
        assert "a" not in d


class TestPeercredValidatedDict:
    """Tests for PeercredValidatedDict validation."""

    def test_setitem_accepts_int_when_whitelist_type(self):
        """场景：写入 int。预期：可读取。"""
        d = PeercredValidatedDict()
        d["n"] = 7
        assert d["n"] == 7

    def test_setitem_raises_schema_error_when_invalid_type(self):
        """场景：写入非白名单对象。预期：SchemaValidateError。"""

        class _Bad:
            pass

        d = PeercredValidatedDict()
        with pytest.raises(SchemaValidateError):
            d["bad"] = _Bad()


class TestPeercredServer:
    """Tests for PeercredServer RPC dispatch."""

    def test_handle_create_dict_registers_object_when_valid_type(self):
        """场景：create dict 请求。预期：registry 注册对象。"""
        server = PeercredServer("/tmp/fake.sock", allowed_uids=None)
        conn = MagicMock()
        server._handle_create_dict(
            conn,
            {"action": "create", "obj_id": "d1", "obj_type": "dict", "args": (), "kwargs": {}},
        )
        conn.send.assert_called_once_with(None)
        assert ("dict", "d1") in server.registry

    def test_handle_method_call_rejects_disallowed_method_when_private(self):
        """场景：调用不允许的方法。预期：返回 SecurityError。"""
        server = PeercredServer("/tmp/fake.sock", allowed_uids=None)
        server.registry[("dict", "d1")] = {}
        conn = MagicMock()
        server._handle_method_call(
            conn,
            {"obj_id": "d1", "obj_type": "dict", "method": "__class__", "args": (), "kwargs": {}},
        )
        sent = conn.send.call_args[0][0]
        assert isinstance(sent, SecurityError)

    def test_handle_method_call_returns_spec_error_when_object_missing(self):
        """场景：对象不存在。预期：返回 SpecError。"""
        server = PeercredServer("/tmp/fake.sock", allowed_uids=None)
        conn = MagicMock()
        server._handle_method_call(
            conn,
            {"obj_id": "missing", "obj_type": "dict", "method": "__getitem__", "args": ("k",), "kwargs": {}},
        )
        sent = conn.send.call_args[0][0]
        assert isinstance(sent, SpecError)


class TestPeercredProxy:
    """Tests for PeercredProxy RPC with mocked socket."""

    @patch("msmodelslim.core.context.shared_dict_context.peercred_manager.PeercredConnection")
    @patch("msmodelslim.core.context.shared_dict_context.peercred_manager.socket")
    def test_call_sends_request_and_returns_result_when_invoked(self, mock_socket_mod, mock_conn_cls):
        """场景：调用 _call。预期：发送请求并返回结果。"""
        mock_socket_mod.AF_UNIX = 1
        mock_socket_mod.SOCK_STREAM = 1
        mock_sock = MagicMock()
        mock_socket_mod.socket.return_value = mock_sock
        mock_conn = MagicMock()
        mock_conn.recv.return_value = 42
        mock_conn_cls.return_value = mock_conn
        proxy = PeercredProxy("/tmp/s.sock", "obj1", "dict")
        assert proxy._call("__len__") == 42
        mock_conn.send.assert_called_once()

    def test_getattr_raises_attribute_error_when_private_name(self):
        """场景：访问私有属性名。预期：AttributeError。"""
        proxy = PeercredProxy("/tmp/s.sock", "obj1")
        with pytest.raises(AttributeError):
            _ = proxy._private


@pytest.mark.skipif(sys.platform != "linux", reason="SO_PEERCRED requires Linux")
class TestPeercredManager:
    """Tests for PeercredManager on Linux."""

    def test_start_creates_validated_dict_proxy_when_started(self):
        """场景：PeercredManager.start 后创建 validated_dict。预期：可读写代理 dict。"""
        manager = PeercredManager()
        try:
            manager.start()
            proxy = manager.validated_dict()
            proxy["key"] = "value"
            assert proxy["key"] == "value"
        finally:
            manager.shutdown()

    @patch(
        "msmodelslim.core.context.shared_dict_context.peercred_manager.os.getuid",
        return_value=1000,
        create=True,
    )
    def test_check_server_exists_returns_false_when_no_socket_file(self, _mock_getuid):
        """场景：socket 文件不存在。预期：非 client 模式。"""
        with patch(
            "msmodelslim.core.context.shared_dict_context.peercred_manager.os.path.exists",
            return_value=False,
        ):
            mgr = PeercredManager(address="/nonexistent/path/peercred.sock")
        assert mgr._is_client is False

    @patch(
        "msmodelslim.core.context.shared_dict_context.peercred_manager.os.getuid",
        return_value=1000,
        create=True,
    )
    @patch("msmodelslim.core.context.shared_dict_context.peercred_manager.os.path.exists", return_value=True)
    @patch("msmodelslim.core.context.shared_dict_context.peercred_manager.socket")
    def test_check_server_exists_returns_true_when_connect_succeeds(self, mock_socket_mod, _mock_exists, _mock_getuid):
        """场景：socket 存在且可连接。预期：client 模式。"""
        mock_socket_mod.AF_UNIX = 1
        mock_socket_mod.SOCK_STREAM = 1
        mock_sock = MagicMock()
        mock_socket_mod.socket.return_value = mock_sock
        mgr = PeercredManager(address="/tmp/running.sock")
        assert mgr._is_client is True

    @patch(
        "msmodelslim.core.context.shared_dict_context.peercred_manager.os.getuid",
        return_value=1000,
        create=True,
    )
    def test_start_raises_misbehavior_error_when_client_mode(self, _mock_getuid):
        """场景：client 模式调用 start。预期：MisbehaviorError。"""
        mgr = PeercredManager.__new__(PeercredManager)
        mgr._is_client = True
        mgr.address = "/tmp/x.sock"
        mgr.allowed_uids = {0}
        mgr._process = None
        mgr._shutdown_called = False
        mgr._lock = __import__("threading").Lock()
        with pytest.raises(MisbehaviorError, match="Client mode"):
            mgr.start()

    @patch(
        "msmodelslim.core.context.shared_dict_context.peercred_manager.os.getuid",
        return_value=1000,
        create=True,
    )
    def test_shutdown_noop_when_client_mode(self, _mock_getuid):
        """场景：client 模式重复 shutdown。预期：不抛异常。"""
        mgr = PeercredManager.__new__(PeercredManager)
        mgr._is_client = True
        mgr._shutdown_called = False
        mgr._lock = __import__("threading").Lock()
        mgr.shutdown()
        mgr.shutdown()


@pytest.mark.skipif(sys.platform == "linux", reason="Only run platform guard test off Linux")
class TestPeercredPlatformCheck:
    """Tests for platform guard on non-Linux."""

    def test_check_platform_support_raises_unsupported_error_when_not_linux(self):
        """场景：非 Linux 调用 _check_platform_support。预期：UnsupportedError。"""
        from msmodelslim.core.context.shared_dict_context import peercred_manager as pm

        pm._platform_checked = False
        try:
            with pytest.raises(UnsupportedError, match="not supported"):
                pm._check_platform_support()
        finally:
            pm._platform_checked = False
