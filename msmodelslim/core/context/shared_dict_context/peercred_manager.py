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
"""

import os
import socket
import struct
import sys
import threading
import time
import uuid
import pickle  # nosec B403
from collections.abc import MutableMapping
from typing import Any, Dict, Optional, Set, Tuple
from multiprocessing import Process, util

from msmodelslim.utils.exception import (
    EnvError,
    MisbehaviorError,
    SchemaValidateError,
    SecurityError,
    SpecError,
    UnsupportedError,
)
from msmodelslim.utils.logging import get_logger
from msmodelslim.core.context.base import ValidatedDict, _validate_namespace_value
from msmodelslim.core.context.interface import IValidatedState, NAMESPACE_VALUE_WHITELIST


# Global registry for container types: name -> class
_container_registry: Dict[str, type] = {}

# Lazy platform check - only performed when SO_PEERCRED is actually used
_platform_checked = False


def _check_platform_support() -> None:
    """Check if the current platform supports SO_PEERCRED.

    This function is called lazily - only when actual IPC communication
    is attempted. Importing the module alone will not trigger this check.

    Raises:
        UnsupportedError: If platform is not Linux (SO_PEERCRED is Linux-only)
    """
    global _platform_checked
    if _platform_checked:
        return

    if sys.platform != 'linux':
        platform_name = 'Windows' if sys.platform == 'win32' else sys.platform
        raise UnsupportedError(
            f"Multi-process shared context is not supported on {platform_name}. "
            f"The underlying mechanism relies on Linux-specific SO_PEERCRED for secure IPC.",
            action="Please run the quantization task on a Linux system, "
            "or use single-process mode (LocalContextFactory) instead.",
        )

    _platform_checked = True


# Protocol constants
MAX_MESSAGE_SIZE = 4 * 1024 * 1024 * 1024  # 4GB max to prevent DoS
PEERCRED_STRUCT_SIZE = 12  # sizeof(struct ucred) on Linux: pid_t + uid_t + gid_t
SIZE_HEADER_BYTES = 4  # uint32 for message size prefix
SOCKET_BACKLOG = 5  # Unix socket listen backlog
SERVER_STARTUP_DELAY = 0.2  # Seconds to wait for server process startup
SERVER_SOCKET_WAIT_TIMEOUT = 10.0  # Max seconds to wait for server socket to accept connections
CONNECTION_TIMEOUT = 30.0  # Socket timeout for blocking operations
ABSTRACT_SOCKET_PREFIX = '\0'
ABSTRACT_SOCKET_NAME_PREFIX = 'peercred_manager'
_WRITE_METHODS_REQUIRING_VALIDATION = frozenset({'__setitem__', 'update', 'setdefault', 'append', 'extend', 'insert'})
_READ_METHODS_REQUIRING_VALIDATION = frozenset({'__getitem__', 'get', 'pop', 'keys', 'values', 'items'})


def _validate_mapping_like_items(mapping: Any, path_prefix: str) -> None:
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            _validate_namespace_value(key, whitelist=NAMESPACE_VALUE_WHITELIST, path=f"{path_prefix}.key")
            _validate_namespace_value(value, whitelist=NAMESPACE_VALUE_WHITELIST, path=f"{path_prefix}[{repr(key)}]")
        return
    if isinstance(mapping, (list, tuple)):
        for idx, item in enumerate(mapping):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                _validate_namespace_value(
                    item[0], whitelist=NAMESPACE_VALUE_WHITELIST, path=f"{path_prefix}[{idx}].key"
                )
                _validate_namespace_value(
                    item[1], whitelist=NAMESPACE_VALUE_WHITELIST, path=f"{path_prefix}[{idx}].value"
                )
            else:
                raise SchemaValidateError(
                    f"Invalid mapping item at {path_prefix}[{idx}]: expected a length-2 key-value pair, "
                    f"got {type(item).__name__}."
                )


def _validate_validated_dict_init(args: tuple, kwargs: dict) -> None:
    if args:
        _validate_mapping_like_items(args[0], "args[0]")
    _validate_mapping_like_items(kwargs, "kwargs")


def _validate_rpc_write_args(method: str, args: tuple, kwargs: dict) -> None:
    if method == '__setitem__':
        if len(args) >= 2:
            _validate_namespace_value(args[0], whitelist=NAMESPACE_VALUE_WHITELIST, path="key")
            _validate_namespace_value(args[1], whitelist=NAMESPACE_VALUE_WHITELIST, path="value")
        return
    if method == 'setdefault':
        if len(args) >= 2:
            _validate_namespace_value(args[0], whitelist=NAMESPACE_VALUE_WHITELIST, path="key")
            _validate_namespace_value(args[1], whitelist=NAMESPACE_VALUE_WHITELIST, path="default")
        elif len(args) >= 1:
            _validate_namespace_value(args[0], whitelist=NAMESPACE_VALUE_WHITELIST, path="key")
            if 'default' in kwargs:
                _validate_namespace_value(kwargs['default'], whitelist=NAMESPACE_VALUE_WHITELIST, path="default")
        return
    if method == 'update':
        if args:
            _validate_mapping_like_items(args[0], "update")
        _validate_mapping_like_items(kwargs, "update_kwargs")
        return
    if method in {'append', 'extend', 'insert'} and args:
        if method == 'insert' and len(args) >= 2:
            _validate_namespace_value(args[1], whitelist=NAMESPACE_VALUE_WHITELIST, path="value")
        elif method == 'extend' and isinstance(args[0], (list, tuple)):
            for idx, item in enumerate(args[0]):
                _validate_namespace_value(item, whitelist=NAMESPACE_VALUE_WHITELIST, path=f"extend[{idx}]")
        else:
            _validate_namespace_value(args[0], whitelist=NAMESPACE_VALUE_WHITELIST, path="value")


def _validate_rpc_read_result(method: str, result: Any) -> None:
    if method in {'keys', 'values', 'items'}:
        _validate_namespace_value(list(result), whitelist=NAMESPACE_VALUE_WHITELIST, path=method)
        return
    _validate_namespace_value(result, whitelist=NAMESPACE_VALUE_WHITELIST, path="result")


def _validate_ipc_payload(obj: Any) -> None:
    """Validate IPC request payload against namespace whitelist before pickle serialization."""
    if obj is None or isinstance(obj, BaseException):
        return

    if not isinstance(obj, dict):
        return

    if obj.get('action') == 'create' and obj.get('obj_type') == 'validated_dict':
        _validate_validated_dict_init(obj.get('args', ()), obj.get('kwargs', {}))
        return

    if 'method' in obj and 'obj_id' in obj:
        if obj.get('obj_type') == 'validated_dict':
            method = obj.get('method')
            if method in _WRITE_METHODS_REQUIRING_VALIDATION:
                _validate_rpc_write_args(method, obj.get('args', ()), obj.get('kwargs', {}))


def make_abstract_address(name: str) -> str:
    """Wrap a logical socket name in the Linux abstract namespace."""
    if name.startswith(ABSTRACT_SOCKET_PREFIX):
        return name
    return ABSTRACT_SOCKET_PREFIX + name


def generate_default_address() -> str:
    """Generate a unique abstract socket address using PID and nanosecond timestamp."""
    name = f"{ABSTRACT_SOCKET_NAME_PREFIX}_{os.getpid()}_{time.time_ns()}"
    return make_abstract_address(name)


def format_address_for_log(address: str) -> str:
    """Format socket address for logging (abstract names omit leading null byte)."""
    if address.startswith(ABSTRACT_SOCKET_PREFIX):
        return f"@{address[1:]}"
    return address


def _try_connect_unix(address: str, timeout: float = 0.1) -> bool:
    """Return True if a Unix domain socket at address accepts connections."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(address)
        return True
    except (socket.error, OSError):
        return False
    finally:
        sock.close()


def get_peer_credentials(sock: socket.socket) -> Tuple[int, int, int]:
    """Extract peer process credentials from Unix socket using SO_PEERCRED.

    Requires Linux and appropriate permissions. Returns (pid, uid, gid).

    Raises:
        UnsupportedError: If platform is not Linux
    """
    _check_platform_support()
    creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, PEERCRED_STRUCT_SIZE)
    pid, uid, gid = struct.unpack('III', creds)
    return pid, uid, gid


class PeercredListener:
    """Unix socket listener that authenticates clients via SO_PEERCRED."""

    def __init__(self, address: str, allowed_uids: Optional[Set] = None):
        self.address = address
        self.allowed_uids = allowed_uids if allowed_uids is not None else {os.getuid()}
        self._socket = None
        self._closed = False

    def start(self):
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(self.address)
        self._socket.listen(SOCKET_BACKLOG)
        get_logger().debug("PeercredListener started on %s", format_address_for_log(self.address))

    def accept(self):
        """Accept connection and authenticate via peer credentials.

        Raises:
            SecurityError: If connecting UID is not in allowed_uids
        """
        conn, addr = self._socket.accept()
        try:
            pid, uid, gid = get_peer_credentials(conn)
            get_logger().debug("Connection from PID=%s, UID=%s, GID=%s", pid, uid, gid)
            if self.allowed_uids is not None and uid not in self.allowed_uids:
                get_logger().warning("Rejecting connection from UID %s", uid)
                try:
                    conn.close()
                except OSError as close_err:
                    get_logger().debug("Ignoring error closing rejected connection: %s", close_err)
                raise SecurityError(
                    f"UID {uid} not authorized for shared context access.",
                    action="Please run all processes under the same user UID.",
                )
            return PeercredConnection(conn), (pid, uid, gid)
        except Exception:
            conn.close()
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            if self._socket:
                self._socket.close()


class PeercredConnection:
    """Thread-safe socket wrapper with framing protocol.

    Note: The lock protects against concurrent send/recv from multiple threads.
    In typical usage (one connection per request), the lock is not needed,
    but it's provided for safety in case of connection reuse.
    """

    def __init__(self, sock: socket.socket, timeout: Optional[float] = CONNECTION_TIMEOUT):
        self._socket = sock
        self._socket.settimeout(timeout)
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send(self, obj):
        """Send object using length-prefixed framing. Format: [4-byte size][data]"""
        _validate_ipc_payload(obj)
        data = pickle.dumps(obj)
        size = len(data)
        if size > MAX_MESSAGE_SIZE:
            raise SecurityError(
                f"Message size {size} exceeds maximum allowed {MAX_MESSAGE_SIZE}",
                action="Please reduce the size of data stored in the shared context.",
            )
        size_bytes = struct.pack('!I', size)
        with self._send_lock:
            self._socket.sendall(size_bytes + data)

    def recv(self):
        with self._recv_lock:
            size_bytes = self._recvall(SIZE_HEADER_BYTES)
            if not size_bytes:
                raise EOFError()
            size = struct.unpack('!I', size_bytes)[0]
            if size > MAX_MESSAGE_SIZE:
                raise SecurityError(
                    f"Message size {size} exceeds maximum allowed {MAX_MESSAGE_SIZE}",
                    action="Please reduce the size of data stored in the shared context.",
                )
            data = self._recvall(size)
        return pickle.loads(data)  # nosec B301

    def _recvall(self, n: int) -> bytes:
        """Receive exactly n bytes. Must be called with _recv_lock held."""
        data = b''
        while len(data) < n:
            chunk = self._socket.recv(n - len(data))
            if not chunk:
                raise EOFError()
            data += chunk
        return data

    def close(self):
        self._socket.close()


# Method whitelist for shared objects - prevents calling dangerous methods
ALLOWED_METHODS = frozenset(
    {
        # dict-like methods
        '__getitem__',
        '__setitem__',
        '__delitem__',
        '__contains__',
        '__len__',
        '__repr__',
        '__str__',
        'get',
        'keys',
        'values',
        'items',
        'update',
        'pop',
        'clear',
        'setdefault',
        # list-like methods (for future extension)
        'append',
        'extend',
        'insert',
        'remove',
        'index',
        'count',
    }
)


class PeercredSharedDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._lock = threading.Lock()

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def __iter__(self):
        with self._lock:
            return iter(list(self._dict.keys()))

    def __len__(self):
        with self._lock:
            return len(self._dict)

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        with self._lock:
            return list(self._dict.values())

    def items(self):
        with self._lock:
            return list(self._dict.items())

    def update(self, other=(), /, **kwargs):
        with self._lock:
            self._dict.update(other, **kwargs)

    def pop(self, key, *args):
        with self._lock:
            return self._dict.pop(key, *args)

    def clear(self):
        with self._lock:
            self._dict.clear()

    def __repr__(self):
        with self._lock:
            return repr(self._dict)


class PeercredValidatedDict(PeercredSharedDict, IValidatedState):
    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
        self._dict = ValidatedDict(*args, **kwargs)
        self._lock = threading.Lock()


_container_registry['dict'] = PeercredSharedDict
_container_registry['validated_dict'] = PeercredValidatedDict


class PeercredProxy:
    """Client-side proxy for remote shared objects.

    Uses dynamic method dispatch via __getattr__ to forward any method
    call to the server. Magic methods (__getitem__, etc.) are explicitly
    defined as they cannot be intercepted by __getattr__.
    """

    def __init__(self, address: str, obj_id: str, obj_type: str = 'dict'):
        self.address = address
        self.obj_id = obj_id
        self.obj_type = obj_type

    def _call(self, method: str, *args, **kwargs):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.address)
            conn = PeercredConnection(sock)
            try:
                request = {
                    'obj_id': self.obj_id,
                    'obj_type': self.obj_type,
                    'method': method,
                    'args': args,
                    'kwargs': kwargs,
                }
                conn.send(request)
                result = conn.recv()
                if isinstance(result, Exception):
                    raise result
                return result
            finally:
                conn.close()
        except Exception:
            # Ensure socket is closed even if connect() fails
            sock.close()
            raise

    def __getattr__(self, name: str):
        """Intercept any method call and forward to remote object.

        This enables the proxy to support all methods of the wrapped
        object without explicitly defining them.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        def method(*args, **kwargs):
            return self._call(name, *args, **kwargs)

        method.__name__ = name
        return method

    def __getitem__(self, key):
        return self._call('__getitem__', key)

    def __setitem__(self, key, value):
        return self._call('__setitem__', key, value)

    def __delitem__(self, key):
        return self._call('__delitem__', key)

    def __contains__(self, key):
        return self._call('__contains__', key)

    def __len__(self):
        return self._call('__len__')

    def __repr__(self):
        return f"{self.__class__.__name__}({self._call('__repr__')})"


class PeercredServer:
    """Multi-threaded server managing shared objects and dispatching RPC calls.

    Uses a registry pattern where objects are identified by (type, id) tuples,
    enabling support for multiple container types through a unified dispatch
    mechanism.
    """

    def __init__(self, address: str, allowed_uids: Optional[Set]):
        self.address = address
        self.allowed_uids = allowed_uids
        # Registry maps (obj_type, obj_id) -> shared object instance
        self.registry: Dict[Tuple[str, str], Any] = {}
        self._listener = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

    def serve_forever(self):
        self._listener = PeercredListener(self.address, self.allowed_uids)
        self._listener.start()
        # Set socket to non-blocking with timeout for clean shutdown
        self._listener._socket.settimeout(1.0)
        get_logger().debug("PeercredServer serving at %s", format_address_for_log(self.address))

        while not self._shutdown_event.is_set():
            try:
                conn, (pid, uid, gid) = self._listener.accept()
                handler = threading.Thread(target=self._handle_client, args=(conn, pid, uid, gid))
                handler.daemon = True
                handler.start()
            except socket.timeout:
                # Normal timeout, check shutdown flag and continue
                continue
            except OSError as e:
                # Socket closed during shutdown
                if self._shutdown_event.is_set():
                    break
                get_logger().error("Error accepting connection: %s", e)
            except Exception as e:
                if not self._shutdown_event.is_set():
                    get_logger().error("Error accepting connection: %s", e)

    def _handle_client(self, conn: PeercredConnection, pid: int, uid: int, gid: int):
        get_logger().debug("Handling client PID=%s, UID=%s", pid, uid)

        while True:
            try:
                request = conn.recv()
                if request is None:
                    break

                self._dispatch_request(conn, request)

            except EOFError:
                break
            except Exception as e:
                get_logger().error("Error handling client request: %s", e)
                break

        conn.close()
        get_logger().debug("Client PID=%s disconnected", pid)

    def _dispatch_request(self, conn: PeercredConnection, request: Dict):
        action = request.get('action')
        if action == 'create':
            self._handle_create_dict(conn, request)
        else:
            self._handle_method_call(conn, request)

    def _handle_create_dict(self, conn: PeercredConnection, request: Dict):
        """Create new shared object in registry."""
        obj_id = request.get('obj_id')
        obj_type = request.get('obj_type', 'dict')
        args = request.get('args', ())
        kwargs = request.get('kwargs', {})

        constructor = _container_registry.get(obj_type)
        if constructor is None:
            conn.send(SpecError(f"Unknown object type: {obj_type}"))
            return

        with self._lock:
            self.registry[(obj_type, obj_id)] = constructor(*args, **kwargs)
        conn.send(None)

    def _handle_method_call(self, conn: PeercredConnection, request: Dict):
        """Dispatch method call to shared object via generic getattr.

        Security: Only methods in ALLOWED_METHODS whitelist can be called.
        This prevents calling dangerous methods like __reduce__, __class__, etc.
        """
        obj_id = request.get('obj_id')
        obj_type = request.get('obj_type', 'dict')
        method = request.get('method')
        args = request.get('args', ())
        kwargs = request.get('kwargs', {})

        # Security: Check method whitelist
        if method not in ALLOWED_METHODS:
            conn.send(
                SecurityError(
                    f"Method '{method}' is not allowed. Allowed methods: {sorted(ALLOWED_METHODS)}",
                    action="Please use only the supported dict-like methods for shared context access.",
                )
            )
            return

        with self._lock:
            obj = self.registry.get((obj_type, obj_id))

        if obj is None:
            conn.send(SpecError(f"Object {obj_type}:{obj_id} not found"))
            return

        try:
            result = getattr(obj, method)(*args, **kwargs)
            if obj_type == 'validated_dict' and method in _READ_METHODS_REQUIRING_VALIDATION:
                _validate_rpc_read_result(method, result)
            conn.send(result)
        except Exception as e:
            self._send_error(conn, e)

    def _send_error(self, conn: PeercredConnection, error: Exception):
        try:
            conn.send(error)
        except Exception:
            get_logger().warning("Failed to send error to client: %s", error)

    def shutdown(self):
        """Signal server to stop accepting connections and close listener."""
        self._shutdown_event.set()
        if self._listener:
            self._listener.close()


def _run_server_process(address: str, allowed_uids: Optional[Set]):
    server = PeercredServer(address, allowed_uids)
    server.serve_forever()


class PeercredManager:
    """High-level manager that spawns a server process and provides shared objects.

    Can operate in server mode (spawns child process) or client mode (connects to existing).
    """

    def __init__(self, address: Optional[str] = None, allowed_uids: Optional[Set] = None):
        if address is None:
            address = generate_default_address()
            self._is_client = False
        else:
            self._is_client = self._check_server_exists(address)

        self.address = address
        self.allowed_uids = set(allowed_uids) if allowed_uids else {os.getuid()}
        self._process = None
        self._dict_counter = 0
        self._lock = threading.Lock()
        self._shutdown_called = False  # Guard against multiple cleanup calls

    def _check_server_exists(self, address: str) -> bool:
        return _try_connect_unix(address, timeout=0.1)

    def start(self):
        if self._is_client:
            raise MisbehaviorError(
                "Client mode manager cannot be started.",
                action="Please create a new manager without address, or use it as a client.",
            )
        self._process = Process(target=_run_server_process, args=(self.address, self.allowed_uids))
        self._process.start()
        time.sleep(SERVER_STARTUP_DELAY)
        # Wait for abstract socket to accept connections (may be slow with spawn child process or in CI)
        deadline = time.monotonic() + SERVER_SOCKET_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if _try_connect_unix(self.address, timeout=1.0):
                break
            time.sleep(0.05)
        else:
            self._finalize_manager(self._process)
            raise EnvError(
                f"Server socket did not become ready at {format_address_for_log(self.address)} "
                f"within {SERVER_SOCKET_WAIT_TIMEOUT}s",
                action="Please ensure the manager server process can start and bind its abstract socket.",
            )
        get_logger().debug("PeercredManager started at %s", format_address_for_log(self.address))

        # 注册 GC/atexit 时清理；不覆盖 self.shutdown，保留显式 shutdown() 供 __exit__ 调用
        util.Finalize(self, PeercredManager._finalize_manager, args=(self._process,), exitpriority=0)

    def shutdown(self):
        """Explicitly shut down the manager: terminate the server process.

        Safe to call multiple times. Only server mode performs actual cleanup.
        Client mode does nothing to avoid affecting the running server.
        """
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True

        # Client mode should not terminate the server or delete socket
        if self._is_client:
            get_logger().debug("Client mode manager shutdown - no cleanup needed")
            return

        PeercredManager._finalize_manager(self._process)

    @staticmethod
    def _finalize_manager(process):
        """Static cleanup method for use by Finalize and shutdown().

        Note: This method may be called multiple times (by shutdown() and Finalize),
        but each operation is idempotent. Abstract sockets are released by the kernel
        when the server process exits.
        """
        if process is not None and process.is_alive():
            get_logger().debug('Sending shutdown to manager')
            try:
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            except Exception as err:
                get_logger().debug("Caught error during manager process cleanup: %s", err)

    def _create_object(self, obj_type: str, *args, **kwargs) -> PeercredProxy:
        obj_id = f"{obj_type}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.address)
            conn = PeercredConnection(sock)
            try:
                conn.send({'action': 'create', 'obj_id': obj_id, 'obj_type': obj_type, 'args': args, 'kwargs': kwargs})
                response = conn.recv()

                if isinstance(response, Exception):
                    raise response

                return PeercredProxy(self.address, obj_id, obj_type)
            finally:
                conn.close()
        except Exception:
            # Ensure socket is closed even if connect() fails
            sock.close()
            raise

    def dict(self, *args, **kwargs) -> PeercredProxy:
        return self._create_object('dict', *args, **kwargs)

    def validated_dict(self, *args, **kwargs) -> PeercredProxy:
        return self._create_object('validated_dict', *args, **kwargs)
