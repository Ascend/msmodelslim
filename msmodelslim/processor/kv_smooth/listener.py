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

from typing import Any, Dict, Optional, Tuple, Set, Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

try:
    from transformers import Cache
except ImportError:
    Cache = object

from msmodelslim.utils.exception import SpecError

CONST_PAST_KEY_VALUES = 'past_key_values'
CONST_PAST_KEY_VALUE = 'past_key_value'


class KVCacheListener:
    def __init__(self, listen_helper: Callable[[int, torch.Tensor, torch.Tensor], None], cache: Cache):
        if cache is None:
            raise SpecError(
                "Cache cannot be None. KVCacheListener requires a valid Cache instance.",
                action="Please provide a valid Cache instance.",
            )
        self.cache = cache
        self.listen_helper = listen_helper

    def __getattribute__(self, name: str):
        """
        除了 update/self 内部字段与必要的内置属性外，其余属性与方法一律从 self.cache 获取，
        以确保父类 Cache 的方法也走底层 cache 的实现。
        """
        # 这些属性必须从当前实例直接获取，避免递归与功能错误
        if name in (
            'update',
            'cache',
            'listen_helper',
        ) or name.startswith('__'):
            return object.__getattribute__(self, name)

        # 优先从底层 cache 获取（实现“除了 update 都用 self.cache 的逻辑”）
        try:
            cache = object.__getattribute__(self, 'cache')
            return getattr(cache, name)
        except AttributeError:
            # 底层没有该属性，退回当前实例/父类
            return object.__getattribute__(self, name)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.listen_helper(layer_idx, key_states, value_states)
        if not hasattr(self.cache, 'update'):
            raise SpecError(
                "Cache does not have update method. KVCacheListener requires a valid Cache instance.",
                action="Please provide a valid Cache instance "
                "with update(key_states, value_states, layer_idx, cache_kwargs) method.",
            )
        return key_states, value_states


class KVCacheListenerManager:
    def __init__(self):
        super().__init__()
        self.remove_handlers: Set[RemovableHandle] = set()

    def attach_listener_to_module(
        self, module: nn.Module, listen_helper: Callable[[int, torch.Tensor, torch.Tensor], None]
    ) -> None:
        def pre_hook(_: nn.Module, args: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
            if CONST_PAST_KEY_VALUES in kwargs and kwargs[CONST_PAST_KEY_VALUES] is not None:
                kwargs[CONST_PAST_KEY_VALUES] = KVCacheListener(listen_helper, cache=kwargs[CONST_PAST_KEY_VALUES])
            elif CONST_PAST_KEY_VALUE in kwargs and kwargs[CONST_PAST_KEY_VALUE] is not None:
                kwargs[CONST_PAST_KEY_VALUE] = KVCacheListener(listen_helper, cache=kwargs[CONST_PAST_KEY_VALUE])
            else:
                raise SpecError(
                    f"{CONST_PAST_KEY_VALUES} and {CONST_PAST_KEY_VALUE} both are None or missing. "
                    f"KVCacheListener requires a valid Cache instance.",
                    action="Please pass a valid transformers.Cache instance via past_key_values/past_key_value.",
                )
            return args, kwargs

        remove_handler = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        self.remove_handlers.add(remove_handler)

    def remove_listeners(self) -> None:
        for handler in self.remove_handlers:
            handler.remove()
        self.remove_handlers.clear()
