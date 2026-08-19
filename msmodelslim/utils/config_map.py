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

import fnmatch
from collections import OrderedDict
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Generic, TypeVar, Set, List, Tuple

T = TypeVar('T')


def _find_brace_start(pattern: str) -> int:
    """返回第一个未转义 ``{`` 的下标；不存在时返回 -1。"""
    i = 0
    while i < len(pattern):
        if pattern[i] == '\\':
            i += 2
            continue
        if pattern[i] == '{':
            return i
        i += 1
    return -1


def _find_brace_end(pattern: str, start: int) -> int:
    """从 ``start`` 处的 ``{`` 起查找匹配的 ``}``（支持嵌套与转义）；未闭合返回 -1。"""
    depth = 0
    i = start
    while i < len(pattern):
        if pattern[i] == '\\':
            i += 2
            continue
        if pattern[i] == '{':
            depth += 1
        elif pattern[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_brace_alternatives(inner: str) -> Tuple[str, ...]:
    """按顶层逗号切分花括号内容，忽略嵌套花括号与转义逗号。"""
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == '\\':
            current.append(ch)
            if i + 1 < len(inner):
                current.append(inner[i + 1])
                i += 1
        elif ch == '{':
            depth += 1
            current.append(ch)
        elif ch == '}':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    parts.append(''.join(current))
    return tuple(parts)


@lru_cache(maxsize=None)
def _expand_braces(pattern: str) -> Tuple[str, ...]:
    """
    展开 ``{a,b,c}`` 形式的 brace 交替（wcmatch BRACE 语义的轻量替代）。

    支持嵌套花括号与反斜杠转义；无逗号或未闭合的花括号保持字面量不变。
    """
    start = _find_brace_start(pattern)
    if start < 0:
        return (pattern,)
    end = _find_brace_end(pattern, start)
    if end < 0:
        return (pattern,)
    alternatives = _split_brace_alternatives(pattern[start + 1 : end])
    if len(alternatives) <= 1:
        return (pattern,)
    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    results: List[str] = []
    for alternative in alternatives:
        for expanded_suffix in _expand_braces(suffix):
            for expanded_prefix in _expand_braces(prefix + alternative):
                results.append(expanded_prefix + expanded_suffix)
    # 去重并保持展开顺序稳定
    return tuple(dict.fromkeys(results))


def _brace_fnmatch(name: str, pattern: str) -> bool:
    """``fnmatch`` 匹配，pattern 先做 ``{a,b,c}`` brace 展开。"""
    return any(fnmatch.fnmatch(name, expanded) for expanded in _expand_braces(pattern))


class ConfigMap(Generic[T], Mapping):
    def __init__(self, cfg_map: OrderedDict[str, T]):
        self.cfg_map: OrderedDict[str, Any] = cfg_map

    def __getitem__(self, key: str) -> T:
        if key in self.cfg_map:
            return self.cfg_map[key]
        for pattern in self.cfg_map:
            if _brace_fnmatch(key, pattern):
                return self.cfg_map[pattern]
        raise KeyError(f"Key '{key}' not found in config map")

    def __contains__(self, key: str) -> bool:
        if key in self.cfg_map:
            return True
        for pattern in self.cfg_map:
            if _brace_fnmatch(key, pattern):
                return True
        return False

    def __iter__(self):
        return iter(self.cfg_map)

    def __len__(self):
        return len(self.cfg_map)


class ConfigSet(Generic[T], Set):
    def __init__(self, cfg_list: List[T]):
        self.cfg_set = OrderedDict().fromkeys(cfg_list)
        self.matched_patterns = set()

    def __contains__(self, key: T) -> bool:
        if key in self.cfg_set:
            self.matched_patterns.add(key)
            return True
        for pattern in self.cfg_set:
            if _brace_fnmatch(key, pattern):
                self.matched_patterns.add(pattern)
                return True
        return False

    def __iter__(self):
        return iter(self.cfg_set)

    def __len__(self):
        return len(self.cfg_set)

    def unmatched_keys(self) -> Set[str]:
        unmatched_keys = set(self.cfg_set.keys()) - self.matched_patterns
        return unmatched_keys
