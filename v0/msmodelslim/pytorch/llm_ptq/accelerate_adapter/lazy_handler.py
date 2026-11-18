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


from typing import Callable, Mapping, MutableMapping

import torch


def handle_lazy_tensor(dic: MutableMapping) -> None:
    for key in dic:
        if isinstance(dic[key], LazyTensor):
            dic[key] = dic[key].value


def get_tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class LazyTensor:
    def __init__(self, func: Callable[..., torch.Tensor], tensor: torch.Tensor = None, **kwargs):
        self._func = func
        self._kwargs = kwargs

        if tensor is not None:
            self._size = get_tensor_size(tensor)
            return

        tensor = self._func(**self._kwargs)
        self._size = get_tensor_size(tensor)

    @property
    def value(self):
        return self._func(**self._kwargs).cpu().contiguous()

    @property
    def size(self):
        return self._size
