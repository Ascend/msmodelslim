#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

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

import pytest
from torch import nn

from msmodelslim.ir.wrapper import HookIR, WrapperIR


class _RecordingHandle:
    def __init__(self):
        self.removed = 0

    def remove(self):
        self.removed += 1


class _PassThroughHook(HookIR):
    def __call__(self, module, args):
        return args

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        return WrapperIR(module)


class _DummyWrapper(WrapperIR):
    pass


def test_is_atomic_returns_false_for_wrapper_ir():
    assert WrapperIR.is_atomic() is False


def test_init_stores_wrapped_module_as_submodule():
    linear = nn.Linear(3, 2, bias=False)

    wrapper = WrapperIR(linear)

    assert wrapper.wrapped_module is linear
    assert dict(wrapper.named_children())["wrapped_module"] is linear


def test_named_modules_yields_only_self_for_wrapper():
    wrapper = _DummyWrapper(nn.Linear(2, 2, bias=False))

    entries = list(wrapper.named_modules())

    assert entries == [("", wrapper)]


def test_hook_ir_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        HookIR()


def test_remove_hook_delegates_to_stored_handle_once():
    hook = _PassThroughHook()
    handle = _RecordingHandle()

    hook.set_hook_handle(handle)
    hook.remove_hook()

    assert handle.removed == 1


def test_subclass_wrapper_module_wraps_given_module():
    hook = _PassThroughHook()
    linear = nn.Linear(2, 2, bias=False)

    wrapper = hook.wrapper_module(linear)

    assert isinstance(wrapper, WrapperIR)
    assert wrapper.wrapped_module is linear
