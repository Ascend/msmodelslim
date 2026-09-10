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

ascendV1_format 目录测试共享: 极简 TensorStore 替身（decoders/format 测试复用）。
"""

import pytest


class FakeStore:
    """与 AscendV1TensorStore 同接口的内存替身，按需注入 shape/真实张量。"""

    def __init__(self):
        self._tensors = {}
        self._shapes = {}

    def put(self, key, tensor):
        self._tensors[key] = tensor

    def put_shape(self, key, shape):
        self._shapes[key] = tuple(shape)

    def has(self, key):
        return key in self._tensors or key in self._shapes

    def get_shape(self, key):
        if key in self._shapes:
            return self._shapes[key]
        return tuple(self._tensors[key].shape)

    def get(self, key):
        if key in self._tensors:
            return self._tensors[key]
        raise KeyError(key)

    def clone_with(self, **tensors):
        store = FakeStore()
        store._tensors.update(self._tensors)
        store._tensors.update(tensors)
        return store


@pytest.fixture
def fake_store_cls():
    return FakeStore
