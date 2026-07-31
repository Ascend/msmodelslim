#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from msmodelslim.model.minimax_m3 import loader as target
from msmodelslim.model.minimax_m3.loader import MiniMaxM3AdapterLoader


def test_minimax_m3_adapter_loader_given_class_when_inspected_then_inherits_from_base_loader():
    from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader

    assert issubclass(MiniMaxM3AdapterLoader, BaseModelAdapterLoader)


def test_minimax_m3_adapter_loader_given_class_when_inspected_then_has_adapter_class_path_attribute():
    assert hasattr(MiniMaxM3AdapterLoader, "ADAPTER_CLASS_PATH")
    assert (
        target.MiniMaxM3AdapterLoader.ADAPTER_CLASS_PATH
        == "msmodelslim.model.minimax_m3.model_adapter:MiniMaxM3ModelAdapter"
    )


def test_minimax_m3_adapter_loader_given_class_path_format_when_inspected_then_contains_colon_separator():
    assert ":" in MiniMaxM3AdapterLoader.ADAPTER_CLASS_PATH
    module_path, class_name = MiniMaxM3AdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)
    assert module_path == "msmodelslim.model.minimax_m3.model_adapter"
    assert class_name == "MiniMaxM3ModelAdapter"


def test_minimax_m3_adapter_loader_given_module_path_when_imported_then_resolves_to_model_adapter_module():
    module_path, class_name = MiniMaxM3AdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)
    adapter_module = importlib.import_module(module_path)
    assert hasattr(adapter_module, class_name)
    assert adapter_module.MiniMaxM3ModelAdapter is not None


def test_minimax_m3_adapter_loader_given_loader_instance_when_get_loader_requirements_called_then_return_dict():
    loader = MiniMaxM3AdapterLoader()
    requirements = loader.get_loader_requirements()
    assert isinstance(requirements, dict)


def test_minimax_m3_adapter_loader_given_loader_instance_when_load_called_with_invalid_path_when_no_colon_then_raise():
    from msmodelslim.utils.exception import UnsupportedError

    class _BadLoader(MiniMaxM3AdapterLoader):
        ADAPTER_CLASS_PATH = "invalid_no_colon"

    with pytest.raises(UnsupportedError):
        _BadLoader().load("minimax_m3", Path("."))


def test_minimax_m3_adapter_loader_given_loader_instance_when_load_called_then_import_adapter_class():
    from msmodelslim.model.plugin_factory import base_loader

    loader = MiniMaxM3AdapterLoader()

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
            base_loader, "import_module", return_value=SimpleNamespace(MiniMaxM3ModelAdapter=fake_adapter_class)
        ),
        patch.object(loader, "check_requirements") as mock_check,
    ):
        out = loader.load("minimax_m3", Path("/tmp/model"), trust_remote_code=True)

    assert isinstance(out, fake_adapter_class)
    assert mock_check.called


def test_minimax_m3_adapter_loader_given_loader_instance_when_precheck_called_then_skip_path_validation(monkeypatch):
    loader = MiniMaxM3AdapterLoader()
    monkeypatch.setattr(loader, "check_requirements", lambda **kwargs: None)
    loader.precheck("minimax_m3", Path("."))
