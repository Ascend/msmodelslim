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

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from msmodelslim.model.minimax_m2 import loader as target
from msmodelslim.model.minimax_m2.loader import MiniMaxM2AdapterLoader


def test_minimax_m2_adapter_loader_given_class_when_inspected_then_inherits_from_base_loader():
    from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader

    assert issubclass(MiniMaxM2AdapterLoader, BaseModelAdapterLoader)


def test_minimax_m2_adapter_loader_given_class_when_inspected_then_has_adapter_class_path_attribute():
    assert hasattr(MiniMaxM2AdapterLoader, "ADAPTER_CLASS_PATH")
    assert (
        target.MiniMaxM2AdapterLoader.ADAPTER_CLASS_PATH
        == "msmodelslim.model.minimax_m2.model_adapter:MiniMaxM2ModelAdapter"
    )


def test_minimax_m2_adapter_loader_given_class_path_format_when_inspected_then_contains_colon_separator():
    assert ":" in MiniMaxM2AdapterLoader.ADAPTER_CLASS_PATH
    module_path, class_name = MiniMaxM2AdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)
    assert module_path == "msmodelslim.model.minimax_m2.model_adapter"
    assert class_name == "MiniMaxM2ModelAdapter"


def test_minimax_m2_adapter_loader_given_module_path_when_imported_then_resolves_to_model_adapter_module():
    module_path, class_name = MiniMaxM2AdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)
    adapter_module = importlib.import_module(module_path)
    assert hasattr(adapter_module, class_name)
    assert adapter_module.MiniMaxM2ModelAdapter is not None


def test_minimax_m2_adapter_loader_given_loader_instance_when_get_loader_requirements_called_then_return_dict():
    loader = MiniMaxM2AdapterLoader()
    requirements = loader.get_loader_requirements()
    assert isinstance(requirements, dict)


def test_minimax_m2_adapter_loader_given_loader_instance_when_load_called_with_invalid_path_when_no_colon_then_raise():
    from msmodelslim.utils.exception import UnsupportedError

    class _BadLoader(MiniMaxM2AdapterLoader):
        ADAPTER_CLASS_PATH = "invalid_no_colon"

    with pytest.raises(UnsupportedError):
        _BadLoader().load("minimax_m2", Path("."))


def test_minimax_m2_adapter_loader_given_loader_instance_when_load_called_then_import_adapter_class():
    from msmodelslim.model.plugin_factory import base_loader

    loader = MiniMaxM2AdapterLoader()

    fake_adapter_class = type(
        "FakeAdapter",
        (),
        {
            "__init__": lambda self, **kwargs: setattr(self, "kwargs", kwargs),
        },
    )

    def fake_get_require_packages(obj):
        return {}

    with (
        patch.object(base_loader, "get_require_packages", side_effect=fake_get_require_packages),
        patch.object(
            base_loader, "import_module", return_value=SimpleNamespace(MiniMaxM2ModelAdapter=fake_adapter_class)
        ),
        patch.object(loader, "check_requirements") as mock_check,
    ):
        out = loader.load("minimax_m2", Path("/tmp/model"), trust_remote_code=True)

    assert isinstance(out, fake_adapter_class)
    assert mock_check.called


def test_minimax_m2_adapter_loader_given_loader_instance_when_precheck_called_then_skip_path_validation(monkeypatch):
    loader = MiniMaxM2AdapterLoader()
    monkeypatch.setattr(loader, "check_requirements", lambda **kwargs: None)
    loader.precheck("minimax_m2", Path("."))
