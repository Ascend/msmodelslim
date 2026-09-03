#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap and CLI helpers for msmodelslim skill scripts."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any


def ensure_msmodelslim() -> None:
    import msmodelslim  # noqa: F401 — trigger Ascend / package patches


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def emit_result(result: dict[str, Any]) -> int:
    print(json.dumps(result, ensure_ascii=False, default=_json_default))
    if result.get("ok") is False:
        return 1
    if result.get("valid") is False:
        return 1
    return 0


def parse_optional_json(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    stripped = value.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("device_indices must be a JSON array")
        return [int(x) for x in parsed]
    return [int(part.strip()) for part in stripped.split(",") if part.strip()]


def with_plugin_timeout_retry(fn, attempts: int = 3, backoff_seconds: float = 2.0):
    """Retry `fn` when msmodelslim 插件冷启动加载超时（Code: 303 / timeout）。

    背景：msmodelslim 插件加载在慢环境上可能超过框架硬编码的 5s 超时
    （Code: 303, Message: Plugin load/execute timed out），属冷启动慢而非逻辑错误；
    同进程重试通常在进程热身后成功。脚本不应把一次可自愈的超时直接当失败上报。
    """
    import time

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            is_timeout = ("Code: 303" in msg) or ("timed out after" in msg) or ("Timeout loading plugin" in msg)
            if not is_timeout:
                raise
            if attempt < attempts - 1:
                time.sleep(backoff_seconds)
    if last_exc is None:
        raise RuntimeError("plugin timeout retry exhausted without capturing an exception")
    raise last_exc
