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

import pytest
from torch import nn

from msmodelslim.core.runner.model_hook_interface import ModelHookInterface


class _HookImpl(ModelHookInterface):
    def load_state_dict_hook(self, key: str, module: nn.Module) -> None:
        module._hook_key = key


class TestModelHookInterface:
    """Tests for ModelHookInterface."""

    def test_load_state_dict_hook_stores_key_when_implemented(self):
        """场景：子类实现 hook。预期：可对模块写入标记。"""
        module = nn.Linear(2, 2)
        hook = _HookImpl()
        hook.load_state_dict_hook("layer0", module)
        assert module._hook_key == "layer0"

    def test_instantiate_raises_type_error_when_abstract(self):
        """场景：直接实例化接口。预期：TypeError。"""
        with pytest.raises(TypeError):
            ModelHookInterface()  # pylint: disable=abstract-class-instantiated
