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

import contextlib

from collections.abc import Iterable
from typing import Any, Optional, Protocol

import torch.distributed as dist

from msmodelslim.core.context.interface import get_current_context

SUMMARY_NAMESPACE = "quant_summary"
SUMMARY_LOGGER_NAME = "msmodelslim.core.context.summary"


class SummaryStateProtocol(Protocol):
    """统计容器的结构化契约。

    单进程 ``ValidatedDict``、分布式 RPC 代理与测试用 ``dict`` 均满足该协议；
    仅声明观察者实际使用的映射操作。
    """

    def __getitem__(self, key: str) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...

    def __delitem__(self, key: str) -> None: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def keys(self) -> Iterable[str]: ...


def current_rank_key() -> str:
    rank = 0
    with contextlib.suppress(Exception):
        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
    return "rank_%d" % rank


def ranked_key(key: str) -> str:
    """维度在该 rank 上的独占顶层 key。

    共享 dict 的读改写跨多次 RPC，无原子性；每个 rank 只写自己的 key，
    避免并发整段回写互相覆盖。
    """
    return "%s@%s" % (key, current_rank_key())


def iter_sections(container: Optional[SummaryStateProtocol], key: str) -> Iterable[dict[str, Any]]:
    """遍历某维度的全部 rank 分片，聚合时按内容去重合并。"""
    if container is None:
        return
    for name in list(container.keys()):
        if name == key or name.startswith(key + "@"):
            section = container.get(name)
            if isinstance(section, dict):
                yield section


class SummaryContext:
    """统计结果承载位置，仅负责按段读写，不感知具体统计维度。

    ``state`` 由外部注入：生产环境经 ``from_context`` 取自
    ``ctx["quant_summary"].state``，注入 ``None`` 时读写退化为空操作。分布式
    共享 context 下嵌套值读取为副本，写入须整体重写本 rank 独占的段；读写异常
    仅告警。
    """

    def __init__(self, state: Optional[SummaryStateProtocol] = None) -> None:
        self._state = state

    @classmethod
    def from_context(cls, context: Any) -> "SummaryContext":
        try:
            namespace = context[SUMMARY_NAMESPACE]
            return cls(getattr(namespace, "state", None))
        except Exception:
            return cls(None)

    @classmethod
    def current(cls) -> "SummaryContext":
        """绑定当前线程 context 的统计段；命名空间按需惰性创建，初始为空。"""
        try:
            return cls.from_context(get_current_context())
        except Exception:
            return cls(None)

    @property
    def enabled(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> Optional[SummaryStateProtocol]:
        return self._state
