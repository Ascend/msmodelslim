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

from typing import Any, Callable, Dict, Optional

# 全局缓存
_cache: Dict[str, Any] = {}


def load_cached(key: str, init_func: Callable, args=None, kwargs=None) -> Any:
    args = args or tuple()
    kwargs = kwargs or dict()
    if key not in _cache:
        _cache[key] = init_func(*args, **kwargs)
    return _cache[key]


def clear_cache(key: Optional[str] = None):
    if key is None:
        _cache.clear()
    else:
        _cache.pop(key, None)
