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

from pathlib import Path

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.core.tune_strategy.standing_high_with_experience.standing_high_with_experience_interface import (
    StandingHighWithExperienceInterface,
)


class _StandingHighWithExperienceModel(StandingHighWithExperienceInterface):
    @property
    def model_type(self):
        return "test"

    @property
    def model_path(self):
        return Path("/tmp/test")

    @property
    def trust_remote_code(self):
        return False

    def handle_dataset(self, dataset, device=DeviceType.NPU):
        return list(dataset) if dataset else []

    def load_model(self, device=DeviceType.NPU):
        return pytest.importorskip("torch").nn.Linear(2, 2)


class TestStandingHighWithExperienceInterface:
    """Tests for StandingHighWithExperienceInterface abstract contract."""

    def test_handle_dataset_returns_list_when_implemented(self):
        """场景：子类实现 handle_dataset。预期：返回处理后的列表。"""
        model = _StandingHighWithExperienceModel()
        assert model.handle_dataset([1, 2]) == [1, 2]

    def test_load_model_returns_module_when_implemented(self):
        """场景：子类实现 load_model。预期：返回 nn.Module。"""
        model = _StandingHighWithExperienceModel()
        loaded = model.load_model(device=DeviceType.NPU)
        assert isinstance(loaded, pytest.importorskip("torch").nn.Module)

    def test_instantiate_raises_type_error_when_abstract_methods_missing(self):
        """场景：未实现抽象方法。预期：TypeError。"""

        class IncompleteModel(StandingHighWithExperienceInterface):
            @property
            def model_type(self):
                return "test"

            @property
            def model_path(self):
                return Path("/tmp/test")

            @property
            def trust_remote_code(self):
                return False

        with pytest.raises(TypeError):
            IncompleteModel()  # pylint: disable=abstract-class-instantiated

    def test_declares_handle_dataset_and_load_model_as_abstract(self):
        """场景：检查接口定义。预期：handle_dataset 与 load_model 为抽象方法。"""
        assert "handle_dataset" in StandingHighWithExperienceInterface.__abstractmethods__
        assert "load_model" in StandingHighWithExperienceInterface.__abstractmethods__
